"""
Text2SQL 评测指标

- EX (Execution Accuracy)：执行结果是否一致
- EM (Exact Match)：SQL 字符串是否完全匹配（归一化后）
- VES (Valid Efficiency Score)：BIRD 特有，考虑执行效率
"""

import re
import sqlite3
from typing import Any, List, Optional, Tuple
from func_timeout import func_timeout, FunctionTimedOut


def normalize_sql(sql: str) -> str:
    """归一化 SQL 字符串用于 EM 比较"""
    sql = sql.strip().rstrip(";").lower()
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'\(\s+', '(', sql)
    sql = re.sub(r'\s+\)', ')', sql)
    return sql.strip()


def execute_sql(db_path: str, sql: str, timeout: float = 30.0) -> Tuple[bool, Any]:
    """
    执行 SQL，返回 (success, result)。
    result 为排序后的结果集（用于比较）。
    """
    def _run():
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return result

    try:
        result = func_timeout(timeout, _run)
        # 排序以消除顺序差异（除非有 ORDER BY）
        if "order by" not in sql.lower():
            result = sorted(result, key=lambda x: str(x))
        return True, result
    except FunctionTimedOut:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def execution_accuracy(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    timeout: float = 30.0,
) -> bool:
    """计算单条 EX"""
    ok_pred, res_pred = execute_sql(db_path, pred_sql, timeout)
    ok_gold, res_gold = execute_sql(db_path, gold_sql, timeout)

    if not ok_pred or not ok_gold:
        return False
    return res_pred == res_gold


def exact_match(pred_sql: str, gold_sql: str) -> bool:
    """计算单条 EM（归一化后字符串匹配）"""
    return normalize_sql(pred_sql) == normalize_sql(gold_sql)


def batch_execution_accuracy(
    predictions: List[str],
    gold_sqls: List[str],
    db_paths: List[str],
    timeout: float = 30.0,
) -> Tuple[float, List[bool]]:
    """批量计算 EX，返回 (accuracy, per_sample_results)"""
    assert len(predictions) == len(gold_sqls) == len(db_paths)
    results = []
    for pred, gold, db in zip(predictions, gold_sqls, db_paths):
        results.append(execution_accuracy(pred, gold, db, timeout))
    acc = sum(results) / len(results) if results else 0.0
    return acc, results


def batch_exact_match(
    predictions: List[str],
    gold_sqls: List[str],
) -> Tuple[float, List[bool]]:
    """批量计算 EM"""
    results = [exact_match(p, g) for p, g in zip(predictions, gold_sqls)]
    acc = sum(results) / len(results) if results else 0.0
    return acc, results
