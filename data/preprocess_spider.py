"""
Spider 数据集预处理
将原始 Spider 数据转换为统一格式：
{
    "db_id": str,
    "question": str,
    "query": str,
    "schema": str,          # 序列化的 schema 字符串
    "clauses": {            # SQL 各子句拆分
        "select": str,
        "from": str,
        "join": str,
        "where": str,
        "group_by": str,
        "having": str,
        "order_by": str,
        "limit": str
    }
}
"""

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML


SPIDER_DIR = Path("data/raw/spider")
OUTPUT_DIR = Path("data/processed/spider")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_schema_str(db_path: str) -> str:
    """从 SQLite 数据库提取 schema 字符串"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema_parts = []
    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        cols = cursor.fetchall()
        col_strs = [f"{col[1]} {col[2]}" for col in cols]
        schema_parts.append(f"Table {table_name}: ({', '.join(col_strs)})")

    conn.close()
    return "\n".join(schema_parts)


def parse_sql_clauses(sql: str) -> Dict[str, str]:
    """将 SQL 拆分为各子句"""
    sql = sql.strip().rstrip(";")
    clauses = {
        "select": "",
        "from": "",
        "join": "",
        "where": "",
        "group_by": "",
        "having": "",
        "order_by": "",
        "limit": ""
    }

    # 用正则按关键字切分（大小写不敏感）
    pattern = re.compile(
        r'\b(SELECT|FROM|(?:LEFT\s+|RIGHT\s+|INNER\s+|OUTER\s+|FULL\s+)?JOIN|WHERE|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT)\b',
        re.IGNORECASE
    )

    tokens = pattern.split(sql)
    # tokens: ['', 'SELECT', '...', 'FROM', '...', ...]
    i = 1
    while i < len(tokens):
        kw = tokens[i].strip().upper().replace("  ", " ")
        val = tokens[i + 1].strip() if i + 1 < len(tokens) else ""
        if kw == "SELECT":
            clauses["select"] = val
        elif kw == "FROM":
            clauses["from"] = val
        elif "JOIN" in kw:
            clauses["join"] = (clauses["join"] + " " + kw + " " + val).strip()
        elif kw == "WHERE":
            clauses["where"] = val
        elif kw == "GROUP BY":
            clauses["group_by"] = val
        elif kw == "HAVING":
            clauses["having"] = val
        elif kw == "ORDER BY":
            clauses["order_by"] = val
        elif kw == "LIMIT":
            clauses["limit"] = val
        i += 2

    return clauses


def process_split(split: str) -> List[Dict]:
    """处理 train/dev 分割"""
    data_file = SPIDER_DIR / f"{split}.json"
    if not data_file.exists():
        print(f"[警告] 找不到 {data_file}，跳过")
        return []

    with open(data_file) as f:
        raw_data = json.load(f)

    processed = []
    for item in raw_data:
        db_id = item["db_id"]
        db_path = str(SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite")

        try:
            schema = get_schema_str(db_path)
        except Exception as e:
            print(f"[警告] 无法读取 {db_id} schema: {e}")
            schema = ""

        query = item.get("query", "")
        clauses = parse_sql_clauses(query)

        processed.append({
            "db_id": db_id,
            "question": item["question"],
            "query": query,
            "schema": schema,
            "clauses": clauses,
            "source": "spider"
        })

    return processed


def main():
    for split in ["train", "dev"]:
        data = process_split(split)
        out_file = OUTPUT_DIR / f"{split}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Spider {split}: {len(data)} 条 -> {out_file}")


if __name__ == "__main__":
    main()
