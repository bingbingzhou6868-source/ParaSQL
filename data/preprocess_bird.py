"""
BIRD 数据集预处理
BIRD 目录结构：
  bird/
    train/
      train.json
      train_databases/
    dev/
      dev.json
      dev_databases/
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List

# 复用 Spider 的子句解析
import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocess_spider import get_schema_str, parse_sql_clauses

BIRD_DIR = Path("data/raw/bird")
OUTPUT_DIR = Path("data/processed/bird")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_split(split: str) -> List[Dict]:
    split_dir = BIRD_DIR / split
    data_file = split_dir / f"{split}.json"

    if not data_file.exists():
        print(f"[警告] 找不到 {data_file}，跳过")
        return []

    with open(data_file, encoding="utf-8") as f:
        raw_data = json.load(f)

    db_dir = split_dir / f"{split}_databases"
    processed = []

    for item in raw_data:
        db_id = item["db_id"]
        db_path = str(db_dir / db_id / f"{db_id}.sqlite")

        try:
            schema = get_schema_str(db_path)
        except Exception as e:
            print(f"[警告] 无法读取 {db_id} schema: {e}")
            schema = ""

        # BIRD 的 SQL 字段名可能是 SQL 或 query
        query = item.get("SQL", item.get("query", ""))
        clauses = parse_sql_clauses(query)

        processed.append({
            "db_id": db_id,
            "question": item["question"],
            "query": query,
            "schema": schema,
            "clauses": clauses,
            "evidence": item.get("evidence", ""),  # BIRD 特有的外部知识
            "difficulty": item.get("difficulty", ""),
            "source": "bird"
        })

    return processed


def main():
    for split in ["train", "dev"]:
        data = process_split(split)
        out_file = OUTPUT_DIR / f"{split}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"BIRD {split}: {len(data)} 条 -> {out_file}")


if __name__ == "__main__":
    main()
