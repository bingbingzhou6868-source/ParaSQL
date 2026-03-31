"""
ParaSQL 强化学习奖励函数

Ground Truth 奖励（不可替代）：
  - 执行正确性：F1 软评分（部分匹配也能得分）

代理奖励（辅助平滑梯度）：
  - 格式规范：输出是否按 <clause_*> + <summary> XML 标签格式
  - 语法正确：SQL 能否被 parser 解析
  - Schema 匹配：用到的表名列名与标准答案的 Jaccard 相似度
  - N-gram 相似度：与标准 SQL 的 bigram 重合度

总奖励 = w_exec * R_exec + w_fmt * R_fmt + w_syn * R_syn + w_schema * R_schema + w_ngram * R_ngram
"""

import re
import sqlite3
import sqlparse
from typing import Dict, List, Optional, Tuple
from func_timeout import func_timeout, FunctionTimedOut


# ── 奖励权重 ──────────────────────────────────────────────────────────────
REWARD_WEIGHTS = {
    "exec":   1.0,   # 执行正确性（最重要）
    "format": 0.1,   # 格式规范
    "syntax": 0.1,   # 语法正确
    "schema": 0.2,   # Schema 匹配
    "ngram":  0.1,   # N-gram 相似度
}

SQL_CLAUSES = ["select", "from", "join", "where", "group_by", "having", "order_by", "limit"]


# ══════════════════════════════════════════════════════════════════════════
# Ground Truth 奖励：执行正确性 F1 软评分
# ══════════════════════════════════════════════════════════════════════════

def execute_sql(db_path: str, sql: str, timeout: float = 10.0):
    """执行 SQL，返回 (success, result_set)"""
    def _run():
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows
    try:
        rows = func_timeout(timeout, _run)
        return True, rows
    except FunctionTimedOut:
        return False, []
    except Exception:
        return False, []


def result_f1_score(pred_rows: List, gold_rows: List) -> float:
    """
    执行结果 F1 软评分。
    将结果集视为多重集合，计算 token 级别的 F1。
    完全匹配 -> 1.0；部分匹配 -> 0~1；完全不匹配 -> 0.0
    """
    if not gold_rows:
        return 1.0 if not pred_rows else 0.0

    def rows_to_tokens(rows):
        tokens = []
        for row in rows:
            for cell in row:
                tokens.extend(str(cell).lower().split())
        return tokens

    pred_tokens = rows_to_tokens(pred_rows)
    gold_tokens = rows_to_tokens(gold_rows)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    # 多重集合交集
    from collections import Counter
    pred_c = Counter(pred_tokens)
    gold_c = Counter(gold_tokens)
    common = sum((pred_c & gold_c).values())

    precision = common / len(pred_tokens)
    recall    = common / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reward_execution(pred_sql: str, gold_sql: str, db_path: str) -> float:
    """
    Ground Truth 奖励：执行正确性 F1 软评分。
    完全匹配执行结果 -> 1.0
    部分匹配 -> 0~1
    执行失败 -> 0.0
    """
    ok_pred, pred_rows = execute_sql(db_path, pred_sql)
    ok_gold, gold_rows = execute_sql(db_path, gold_sql)

    if not ok_gold:
        return 0.0
    if not ok_pred:
        return 0.0

    # 排序消除顺序差异（除非有 ORDER BY）
    has_order = "order by" in pred_sql.lower()
    if not has_order:
        pred_rows = sorted(pred_rows, key=lambda x: str(x))
        gold_rows = sorted(gold_rows, key=lambda x: str(x))

    if pred_rows == gold_rows:
        return 1.0
    return result_f1_score(pred_rows, gold_rows)


# ══════════════════════════════════════════════════════════════════════════
# 代理奖励 1：格式规范
# ══════════════════════════════════════════════════════════════════════════

def reward_format(model_output: str) -> float:
    """
    检查输出是否包含所有必要的 XML 标签格式：
    - 所有 8 个 <clause_*> 开闭标签
    - <summary> 开闭标签
    每缺少一个标签扣分，全部存在得 1.0
    """
    required_tags = (
        [f"<clause_{c}>" for c in SQL_CLAUSES] +
        [f"</clause_{c}>" for c in SQL_CLAUSES] +
        ["<summary>", "</summary>"]
    )
    present = sum(1 for tag in required_tags if tag in model_output)
    return present / len(required_tags)


# ══════════════════════════════════════════════════════════════════════════
# 代理奖励 2：语法正确
# ══════════════════════════════════════════════════════════════════════════

def reward_syntax(pred_sql: str) -> float:
    """
    检查 SQL 能否被 sqlparse 解析（无语法错误）。
    能解析 -> 1.0；解析失败 -> 0.0
    """
    if not pred_sql or not pred_sql.strip():
        return 0.0
    try:
        parsed = sqlparse.parse(pred_sql.strip())
        if not parsed or not parsed[0].tokens:
            return 0.0
        # 进一步检查：尝试用 sqlite3 的 explain 验证
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(f"EXPLAIN {pred_sql}")
            conn.close()
            return 1.0
        except Exception:
            conn.close()
            # sqlparse 能解析但 sqlite 报错，给部分分
            return 0.5
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════
# 代理奖励 3：Schema 匹配（Jaccard 相似度）
# ══════════════════════════════════════════════════════════════════════════

def extract_schema_tokens(sql: str) -> set:
    """从 SQL 中提取表名和列名 token"""
    sql_lower = sql.lower()
    # 去掉字符串字面量
    sql_clean = re.sub(r"'[^']*'", "", sql_lower)
    sql_clean = re.sub(r'"[^"]*"', "", sql_clean)
    # 提取标识符（字母数字下划线，长度>1）
    tokens = set(re.findall(r'\b[a-z_][a-z0-9_]{1,}\b', sql_clean))
    # 过滤 SQL 关键字
    sql_keywords = {
        'select', 'from', 'where', 'join', 'on', 'and', 'or', 'not',
        'in', 'is', 'null', 'like', 'between', 'group', 'by', 'having',
        'order', 'limit', 'distinct', 'count', 'sum', 'avg', 'max', 'min',
        'inner', 'left', 'right', 'outer', 'as', 'case', 'when', 'then',
        'else', 'end', 'union', 'all', 'exists', 'asc', 'desc', 'with',
    }
    return tokens - sql_keywords


def reward_schema(pred_sql: str, gold_sql: str) -> float:
    """
    Schema 匹配：用到的表名列名与标准答案的 Jaccard 相似度。
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    """
    pred_tokens = extract_schema_tokens(pred_sql)
    gold_tokens = extract_schema_tokens(gold_sql)

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.5
    if not pred_tokens:
        return 0.0

    intersection = len(pred_tokens & gold_tokens)
    union        = len(pred_tokens | gold_tokens)
    return intersection / union if union > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════
# 代理奖励 4：N-gram 相似度（bigram）
# ══════════════════════════════════════════════════════════════════════════

def get_ngrams(text: str, n: int = 2) -> List[tuple]:
    """提取 n-gram"""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def reward_ngram(pred_sql: str, gold_sql: str, n: int = 2) -> float:
    """
    N-gram 相似度：与标准 SQL 的 bigram 重合度（F1）。
    """
    pred_ngrams = get_ngrams(pred_sql, n)
    gold_ngrams = get_ngrams(gold_sql, n)

    if not gold_ngrams:
        return 1.0 if not pred_ngrams else 0.0
    if not pred_ngrams:
        return 0.0

    from collections import Counter
    pred_c = Counter(pred_ngrams)
    gold_c = Counter(gold_ngrams)
    common = sum((pred_c & gold_c).values())

    precision = common / len(pred_ngrams)
    recall    = common / len(gold_ngrams)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ══════════════════════════════════════════════════════════════════════════
# 综合奖励
# ══════════════════════════════════════════════════════════════════════════

def compute_reward(
    model_output: str,
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    weights: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    计算综合奖励，返回各分项和总分。

    Args:
        model_output: 模型完整输出（含 XML 标签）
        pred_sql:     从 model_output 中解析出的最终 SQL
        gold_sql:     标准答案 SQL
        db_path:      SQLite 数据库路径
        weights:      各奖励权重（默认使用 REWARD_WEIGHTS）

    Returns:
        {
            "exec":   float,  # 执行正确性 F1
            "format": float,  # 格式规范
            "syntax": float,  # 语法正确
            "schema": float,  # Schema Jaccard
            "ngram":  float,  # Bigram F1
            "total":  float,  # 加权总分
        }
    """
    if weights is None:
        weights = REWARD_WEIGHTS

    r_exec   = reward_execution(pred_sql, gold_sql, db_path)
    r_format = reward_format(model_output)
    r_syntax = reward_syntax(pred_sql)
    r_schema = reward_schema(pred_sql, gold_sql)
    r_ngram  = reward_ngram(pred_sql, gold_sql)

    total = (
        weights["exec"]   * r_exec   +
        weights["format"] * r_format +
        weights["syntax"] * r_syntax +
        weights["schema"] * r_schema +
        weights["ngram"]  * r_ngram
    )
    # 归一化到 [0, 1]
    total_max = sum(weights.values())
    total_normalized = total / total_max

    return {
        "exec":   r_exec,
        "format": r_format,
        "syntax": r_syntax,
        "schema": r_schema,
        "ngram":  r_ngram,
        "total":  total_normalized,
    }
