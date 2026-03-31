"""
ParaSQL 的 Prompt 模板

并行子句生成的 prompt 设计，仿照 ParaThinker 的控制 token 机制。
"""

from typing import Dict, Optional

SQL_CLAUSES = ["select", "from", "join", "where", "group_by", "having", "order_by", "limit"]

SYSTEM_PROMPT = """You are a SQL expert. Given a database schema and a natural language question, generate each SQL clause independently within its tags, then synthesize them into a complete SQL query.

Rules:
- Generate each clause independently and in parallel
- Leave empty tags if a clause is not needed
- The <summary> must contain the complete, executable SQL query
- Use table aliases for clarity"""


def build_user_prompt(
    schema: str,
    question: str,
    evidence: str = "",
    db_id: str = "",
) -> str:
    """构建用户输入 prompt"""
    parts = []
    if db_id:
        parts.append(f"Database: {db_id}")
    parts.append(f"### Schema:\n{schema}")
    if evidence:
        parts.append(f"### External Knowledge:\n{evidence}")
    parts.append(f"### Question:\n{question}")
    parts.append(
        "\nGenerate each SQL clause in parallel within the corresponding tags, "
        "then provide the complete SQL in <summary></summary>."
    )
    return "\n\n".join(parts)


def build_assistant_prefix() -> str:
    """推理时的 assistant 前缀，触发并行子句生成"""
    lines = []
    for clause in SQL_CLAUSES:
        lines.append(f"<clause_{clause}>")
    return "\n".join(lines)


def format_clause_output(clauses: Dict[str, str], full_sql: str) -> str:
    """格式化完整的 assistant 输出（训练数据用）"""
    lines = []
    for clause in SQL_CLAUSES:
        val = clauses.get(clause, "").strip()
        lines.append(f"<clause_{clause}> {val} </clause_{clause}>")
    lines.append(f"<summary>\n{full_sql.strip()}\n</summary>")
    return "\n".join(lines)


def build_chat_messages(
    schema: str,
    question: str,
    evidence: str = "",
    db_id: str = "",
) -> list:
    """构建 chat 格式的消息列表（用于 tokenizer.apply_chat_template）"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(schema, question, evidence, db_id)},
    ]
