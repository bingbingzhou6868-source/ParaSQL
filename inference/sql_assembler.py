"""
SQL 子句汇总器（SQL Assembler）

从模型输出中解析各子句 tag，汇总为完整 SQL。
对应 ParaThinker 的 summarization 阶段。
"""

import re
from typing import Dict, Optional, Tuple

SQL_CLAUSES = ["select", "from", "join", "where", "group_by", "having", "order_by", "limit"]


def extract_clause(text: str, clause: str) -> str:
    """从模型输出中提取指定子句的内容"""
    open_tag = f"<clause_{clause}>"
    close_tag = f"</clause_{clause}>"
    pattern = re.compile(
        re.escape(open_tag) + r"\s*(.*?)\s*" + re.escape(close_tag),
        re.DOTALL | re.IGNORECASE
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def extract_summary(text: str) -> str:
    """从模型输出中提取 <summary> 内的完整 SQL"""
    pattern = re.compile(r"<summary>\s*(.*?)\s*</summary>", re.DOTALL | re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def parse_all_clauses(text: str) -> Dict[str, str]:
    """解析所有子句"""
    return {clause: extract_clause(text, clause) for clause in SQL_CLAUSES}


def assemble_sql_from_clauses(clauses: Dict[str, str]) -> str:
    """
    从各子句字典拼接完整 SQL。
    当 <summary> 不可用时的 fallback 方案。
    """
    parts = []

    select = clauses.get("select", "").strip()
    if select:
        if not select.upper().startswith("SELECT"):
            select = "SELECT " + select
        parts.append(select)

    from_ = clauses.get("from", "").strip()
    if from_:
        if not from_.upper().startswith("FROM"):
            from_ = "FROM " + from_
        parts.append(from_)

    join = clauses.get("join", "").strip()
    if join:
        parts.append(join)

    where = clauses.get("where", "").strip()
    if where:
        if not where.upper().startswith("WHERE"):
            where = "WHERE " + where
        parts.append(where)

    group_by = clauses.get("group_by", "").strip()
    if group_by:
        if not group_by.upper().startswith("GROUP"):
            group_by = "GROUP BY " + group_by
        parts.append(group_by)

    having = clauses.get("having", "").strip()
    if having:
        if not having.upper().startswith("HAVING"):
            having = "HAVING " + having
        parts.append(having)

    order_by = clauses.get("order_by", "").strip()
    if order_by:
        if not order_by.upper().startswith("ORDER"):
            order_by = "ORDER BY " + order_by
        parts.append(order_by)

    limit = clauses.get("limit", "").strip()
    if limit:
        if not limit.upper().startswith("LIMIT"):
            limit = "LIMIT " + limit
        parts.append(limit)

    return "\n".join(parts)


def parse_model_output(output: str) -> Tuple[str, Dict[str, str]]:
    """
    解析模型完整输出，返回 (final_sql, clauses_dict)。
    优先使用 <summary> 中的 SQL，fallback 到子句拼接。
    """
    clauses = parse_all_clauses(output)
    summary_sql = extract_summary(output)

    if summary_sql:
        final_sql = summary_sql
    else:
        final_sql = assemble_sql_from_clauses(clauses)

    return final_sql, clauses
