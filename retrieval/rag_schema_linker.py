"""
两层 RAG Schema Linking 模块

第一层：业务路由（Layer1Router）
  - 根据用户问题识别业务领域
  - 关键词匹配 + Embedding 相似度
  - 将候选数据库从几十个缩小到 1-2 个目标库

第二层：Schema 检索（Layer2SchemaRetriever）
  - 在目标数据库内检索最相关的表和字段
  - 从历史日志中找到相似查询作为 few-shot 示例
  - 将精准 schema + few-shot 示例拼进 prompt

最终 prompt 只包含相关 schema 和几个参考示例，
而不是整个数据库的全量信息。
"""

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def get_embedding(text: str, model=None) -> np.ndarray:
    """
    获取文本的 embedding 向量。
    优先使用传入的 model（sentence-transformers），
    fallback 到简单的 TF-IDF 风格词袋向量。
    """
    if model is not None:
        return np.array(model.encode(text, normalize_embeddings=True))
    # Fallback: 简单词袋 embedding（仅用于测试）
    words = re.findall(r'\w+', text.lower())
    vocab = sorted(set(words))
    vec = np.zeros(max(len(vocab), 1))
    for i, w in enumerate(vocab):
        vec[i] = words.count(w)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ══════════════════════════════════════════════════════════════════════════
# 第一层：业务路由
# ══════════════════════════════════════════════════════════════════════════

class Layer1Router:
    """
    第一层：业务路由
    根据用户问题识别业务领域，将候选数据库缩小到 1-2 个目标库。

    策略：关键词匹配 + Embedding 相似度融合打分
      score(db) = α * keyword_score(q, db) + (1-α) * embedding_sim(q, db)
    """

    def __init__(
        self,
        db_descriptions: Dict[str, str],   # {db_id: 业务描述文本}
        keyword_map: Dict[str, List[str]],  # {db_id: [关键词列表]}
        embedding_model=None,
        alpha: float = 0.4,                 # 关键词权重
        top_k: int = 2,                     # 返回 top-k 个数据库
    ):
        self.db_descriptions = db_descriptions
        self.keyword_map = keyword_map
        self.embedding_model = embedding_model
        self.alpha = alpha
        self.top_k = top_k

        # 预计算数据库描述的 embedding
        self.db_embeddings: Dict[str, np.ndarray] = {}
        for db_id, desc in db_descriptions.items():
            self.db_embeddings[db_id] = get_embedding(desc, embedding_model)

    def _keyword_score(self, question: str, db_id: str) -> float:
        """关键词匹配得分：命中关键词数 / 总关键词数"""
        keywords = self.keyword_map.get(db_id, [])
        if not keywords:
            return 0.0
        q_lower = question.lower()
        hits = sum(1 for kw in keywords if kw.lower() in q_lower)
        return hits / len(keywords)

    def route(self, question: str) -> List[Tuple[str, float]]:
        """
        路由问题到最相关的数据库。

        Returns:
            [(db_id, score), ...] 按分数降序排列，取 top_k
        """
        q_emb = get_embedding(question, self.embedding_model)
        scores = {}
        for db_id in self.db_descriptions:
            kw_score  = self._keyword_score(question, db_id)
            emb_score = cosine_similarity(q_emb, self.db_embeddings[db_id])
            scores[db_id] = self.alpha * kw_score + (1 - self.alpha) * emb_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:self.top_k]

    @classmethod
    def from_db_dir(cls, db_dir: str, **kwargs) -> "Layer1Router":
        """
        从数据库目录自动构建路由器。
        每个子目录对应一个数据库，从表名和列名自动生成描述和关键词。
        """
        db_dir = Path(db_dir)
        descriptions = {}
        keyword_map  = {}

        for db_path in db_dir.glob("*/*.sqlite"):
            db_id = db_path.parent.name
            try:
                conn = sqlite3.connect(str(db_path))
                conn.text_factory = lambda b: b.decode(errors="ignore")
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [r[0] for r in cur.fetchall()]

                all_cols = []
                for tbl in tables:
                    cur.execute(f"PRAGMA table_info(`{tbl}`)")
                    cols = [r[1] for r in cur.fetchall()]
                    all_cols.extend(cols)
                conn.close()

                desc = f"Database {db_id}. Tables: {', '.join(tables)}. "
                desc += f"Columns include: {', '.join(all_cols[:20])}."
                descriptions[db_id] = desc
                keyword_map[db_id]  = tables + all_cols[:30]
            except Exception:
                descriptions[db_id] = f"Database {db_id}"
                keyword_map[db_id]  = [db_id]

        return cls(descriptions, keyword_map, **kwargs)


# ══════════════════════════════════════════════════════════════════════════
# 第二层：Schema 检索
# ══════════════════════════════════════════════════════════════════════════

class Layer2SchemaRetriever:
    """
    第二层：Schema 检索
    在目标数据库内检索最相关的表和字段，
    并从历史日志中找到相似查询作为 few-shot 示例。

    策略：
      1. 表级相关性：问题与表名/列名的 embedding 相似度
      2. Few-shot 检索：从历史 (question, SQL) 对中找最相似的 top-k 示例
    """

    def __init__(
        self,
        db_base_dir: str,
        history_log: Optional[List[Dict]] = None,  # [{question, sql, db_id}, ...]
        embedding_model=None,
        top_tables: int = 5,      # 每个数据库最多保留几张表
        top_cols: int = 10,       # 每张表最多保留几列
        top_fewshot: int = 3,     # few-shot 示例数量
    ):
        self.db_base_dir    = Path(db_base_dir)
        self.history_log    = history_log or []
        self.embedding_model = embedding_model
        self.top_tables     = top_tables
        self.top_cols       = top_cols
        self.top_fewshot    = top_fewshot

        # 预计算历史问题的 embedding
        self._history_embeddings: List[np.ndarray] = []
        for item in self.history_log:
            emb = get_embedding(item["question"], embedding_model)
            self._history_embeddings.append(emb)

    def _get_full_schema(self, db_id: str) -> Dict[str, List[str]]:
        """获取数据库完整 schema：{table: [col1, col2, ...]}"""
        db_path = self.db_base_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            return {}
        try:
            conn = sqlite3.connect(str(db_path))
            conn.text_factory = lambda b: b.decode(errors="ignore")
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            schema = {}
            for tbl in tables:
                cur.execute(f"PRAGMA table_info(`{tbl}`)")
                cols = [r[1] for r in cur.fetchall()]
                schema[tbl] = cols
            conn.close()
            return schema
        except Exception:
            return {}

    def _score_table(self, question: str, table_name: str, cols: List[str]) -> float:
        """计算问题与表（名+列名）的相关性得分"""
        table_text = f"{table_name} {' '.join(cols)}"
        q_emb = get_embedding(question, self.embedding_model)
        t_emb = get_embedding(table_text, self.embedding_model)
        return cosine_similarity(q_emb, t_emb)

    def retrieve_schema(self, question: str, db_id: str) -> Dict[str, List[str]]:
        """
        检索与问题最相关的表和列。

        Returns:
            {table_name: [relevant_cols, ...]}
        """
        full_schema = self._get_full_schema(db_id)
        if not full_schema:
            return {}

        # 对每张表打分
        table_scores = {}
        for tbl, cols in full_schema.items():
            table_scores[tbl] = self._score_table(question, tbl, cols)

        # 取 top_tables 张表
        top_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        top_tables = top_tables[:self.top_tables]

        # 对每张表的列也做相关性筛选
        result = {}
        q_emb = get_embedding(question, self.embedding_model)
        for tbl, _ in top_tables:
            cols = full_schema[tbl]
            if len(cols) <= self.top_cols:
                result[tbl] = cols
            else:
                col_scores = []
                for col in cols:
                    c_emb = get_embedding(col, self.embedding_model)
                    col_scores.append((col, cosine_similarity(q_emb, c_emb)))
                col_scores.sort(key=lambda x: x[1], reverse=True)
                result[tbl] = [c for c, _ in col_scores[:self.top_cols]]

        return result

    def retrieve_fewshot(self, question: str, db_id: str) -> List[Dict]:
        """
        从历史日志中检索最相似的 few-shot 示例。

        Returns:
            [{question, sql}, ...] top-k 最相似示例
        """
        if not self.history_log:
            return []

        q_emb = get_embedding(question, self.embedding_model)
        scored = []
        for i, (item, emb) in enumerate(zip(self.history_log, self._history_embeddings)):
            if item.get("db_id") != db_id:
                continue
            sim = cosine_similarity(q_emb, emb)
            scored.append((sim, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:self.top_fewshot]]

    def build_prompt_context(
        self,
        question: str,
        db_id: str,
        evidence: str = "",
    ) -> str:
        """
        构建精准的 prompt 上下文：
        相关 schema + few-shot 示例（而非全量数据库信息）
        """
        # 检索相关 schema
        schema = self.retrieve_schema(question, db_id)
        schema_lines = [f"Database: {db_id}"]
        for tbl, cols in schema.items():
            schema_lines.append(f"  Table {tbl}: ({', '.join(cols)})")
        schema_str = "\n".join(schema_lines)

        # 检索 few-shot 示例
        fewshots = self.retrieve_fewshot(question, db_id)
        fewshot_str = ""
        if fewshots:
            fewshot_lines = ["\n### Similar Examples:"]
            for i, ex in enumerate(fewshots, 1):
                fewshot_lines.append(f"Example {i}:")
                fewshot_lines.append(f"  Q: {ex['question']}")
                fewshot_lines.append(f"  SQL: {ex['sql']}")
            fewshot_str = "\n".join(fewshot_lines)

        # 拼接 prompt 上下文
        parts = [f"### Relevant Schema:\n{schema_str}"]
        if evidence:
            parts.append(f"### External Knowledge:\n{evidence}")
        if fewshot_str:
            parts.append(fewshot_str)
        parts.append(f"### Question:\n{question}")

        return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# 完整两层 RAG Schema Linker
# ══════════════════════════════════════════════════════════════════════════

class RAGSchemaLinker:
    """
    两层 RAG Schema Linking 完整流程：
      Layer 1: 业务路由 → 目标数据库 (1-2个)
      Layer 2: Schema 检索 → 精准表/列 + few-shot 示例
    """

    def __init__(
        self,
        db_base_dir: str,
        db_descriptions: Optional[Dict[str, str]] = None,
        keyword_map: Optional[Dict[str, List[str]]] = None,
        history_log: Optional[List[Dict]] = None,
        embedding_model=None,
        router_top_k: int = 2,
        top_tables: int = 5,
        top_cols: int = 10,
        top_fewshot: int = 3,
        alpha: float = 0.4,
    ):
        # 第一层：业务路由
        if db_descriptions is None or keyword_map is None:
            self.router = Layer1Router.from_db_dir(
                db_base_dir,
                embedding_model=embedding_model,
                alpha=alpha,
                top_k=router_top_k,
            )
        else:
            self.router = Layer1Router(
                db_descriptions, keyword_map,
                embedding_model=embedding_model,
                alpha=alpha,
                top_k=router_top_k,
            )

        # 第二层：Schema 检索
        self.retriever = Layer2SchemaRetriever(
            db_base_dir=db_base_dir,
            history_log=history_log,
            embedding_model=embedding_model,
            top_tables=top_tables,
            top_cols=top_cols,
            top_fewshot=top_fewshot,
        )

    def link(
        self,
        question: str,
        evidence: str = "",
        db_id: Optional[str] = None,
    ) -> Tuple[str, str, List[Tuple[str, float]]]:
        """
        完整两层 RAG 流程。

        Args:
            question: 自然语言问题
            evidence: 外部知识（BIRD 特有）
            db_id:    已知数据库 ID（如果已知则跳过第一层路由）

        Returns:
            (prompt_context, selected_db_id, routing_scores)
        """
        # 第一层：路由
        if db_id is None:
            routing = self.router.route(question)
            selected_db_id = routing[0][0]
        else:
            routing = [(db_id, 1.0)]
            selected_db_id = db_id

        # 第二层：Schema 检索 + prompt 构建
        prompt_context = self.retriever.build_prompt_context(
            question, selected_db_id, evidence
        )

        return prompt_context, selected_db_id, routing
