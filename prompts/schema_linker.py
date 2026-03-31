"""
Schema Linking：将 question 中的实体链接到 schema 中的表/列。
用于在 prompt 中提供更精简的 schema，减少 token 消耗。
"""

import re
from typing import Dict, List, Tuple


def serialize_schema(
    tables: Dict[str, List[str]],
    foreign_keys: List[Tuple[str, str, str, str]] = None,
    primary_keys: Dict[str, str] = None,
    compact: bool = False,
) -> str:
    """
    将 schema 序列化为字符串。

    Args:
        tables: {table_name: [col1, col2, ...]}
        foreign_keys: [(table1, col1, table2, col2), ...]
        primary_keys: {table_name: pk_col}
        compact: 是否使用紧凑格式

    Returns:
        schema 字符串
    """
    lines = []
    for table, cols in tables.items():
        pk = primary_keys.get(table, "") if primary_keys else ""
        col_strs = []
        for col in cols:
            marker = " [PK]" if col == pk else ""
            col_strs.append(f"{col}{marker}")

        if compact:
            lines.append(f"{table}({', '.join(col_strs)})")
        else:
            lines.append(f"Table {table}:")
            for col in col_strs:
                lines.append(f"  - {col}")

    if foreign_keys:
        lines.append("\nForeign Keys:")
        for t1, c1, t2, c2 in foreign_keys:
            lines.append(f"  {t1}.{c1} = {t2}.{c2}")

    return "\n".join(lines)


def simple_schema_link(question: str, schema_str: str) -> str:
    """
    简单的 schema linking：保留 question 中提到的表/列名相关的 schema 行。
    生产环境可替换为更复杂的 embedding-based linking。
    """
    question_lower = question.lower()
    lines = schema_str.split("\n")
    relevant = []
    for line in lines:
        # 保留包含 question 中词汇的行
        words = re.findall(r'\w+', line.lower())
        if any(w in question_lower for w in words if len(w) > 2):
            relevant.append(line)
        elif line.startswith("Table") or line.startswith("Foreign"):
            relevant.append(line)

    return "\n".join(relevant) if relevant else schema_str
