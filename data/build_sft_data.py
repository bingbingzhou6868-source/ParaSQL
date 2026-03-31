"""
构建 ParaSQL 的 SFT 训练数据。

核心格式（仿照 ParaThinker）：
<|User|> [schema] [question]
<|Assistant|>
<clause_select> SELECT t1.name </clause_select>
<clause_from> FROM employees AS t1 </clause_from>
<clause_join> JOIN departments AS t2 ON t1.dept_id = t2.id </clause_join>
<clause_where> WHERE t2.name = 'Engineering' </clause_where>
<clause_group_by> </clause_group_by>
<clause_having> </clause_having>
<clause_order_by> ORDER BY t1.salary DESC </clause_order_by>
<clause_limit> LIMIT 5 </clause_limit>
<summary>
SELECT t1.name FROM employees AS t1
JOIN departments AS t2 ON t1.dept_id = t2.id
WHERE t2.name = 'Engineering'
ORDER BY t1.salary DESC LIMIT 5
</summary>
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

OUTPUT_DIR = Path("data/sft")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLAUSE_ORDER = ["select", "from", "join", "where", "group_by", "having", "order_by", "limit"]
CLAUSE_TOKEN_MAP = {
    "select":   ("<clause_select>",   "</clause_select>"),
    "from":     ("<clause_from>",     "</clause_from>"),
    "join":     ("<clause_join>",     "</clause_join>"),
    "where":    ("<clause_where>",    "</clause_where>"),
    "group_by": ("<clause_group_by>", "</clause_group_by>"),
    "having":   ("<clause_having>",   "</clause_having>"),
    "order_by": ("<clause_order_by>", "</clause_order_by>"),
    "limit":    ("<clause_limit>",    "</clause_limit>"),
}

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural language question, "
    "generate each SQL clause independently and in parallel, then synthesize them into "
    "a complete, executable SQL query.\n"
    "Generate each clause within its corresponding tags, then provide the final SQL "
    "within <summary></summary> tags."
)


def build_schema_prompt(schema: str, evidence: str = "") -> str:
    prompt = f"### Database Schema:\n{schema}\n"
    if evidence:
        prompt += f"\n### External Knowledge:\n{evidence}\n"
    return prompt


def build_clause_output(clauses: Dict[str, str], full_sql: str) -> str:
    """构建并行子句输出格式"""
    lines = []
    for key in CLAUSE_ORDER:
        open_tag, close_tag = CLAUSE_TOKEN_MAP[key]
        val = clauses.get(key, "").strip()
        # 即使为空也保留 tag，让模型学会"此子句不需要"
        lines.append(f"{open_tag} {val} {close_tag}")

    lines.append(f"<summary>\n{full_sql.strip()}\n</summary>")
    return "\n".join(lines)


def build_sft_sample(item: Dict) -> Optional[Dict]:
    """将一条预处理数据转为 SFT 格式"""
    schema = item.get("schema", "")
    question = item.get("question", "")
    query = item.get("query", "")
    clauses = item.get("clauses", {})
    evidence = item.get("evidence", "")

    if not query or not question:
        return None

    user_content = build_schema_prompt(schema, evidence) + f"\n### Question:\n{question}"
    assistant_content = build_clause_output(clauses, query)

    return {
        "system": SYSTEM_PROMPT,
        "conversations": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "db_id": item.get("db_id", ""),
        "source": item.get("source", ""),
        "query": query,
    }


def process_dataset(dataset: str, split: str) -> List[Dict]:
    in_file = Path(f"data/processed/{dataset}/{split}.json")
    if not in_file.exists():
        print(f"[跳过] {in_file} 不存在")
        return []

    with open(in_file, encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for item in raw:
        s = build_sft_sample(item)
        if s:
            samples.append(s)

    print(f"{dataset}/{split}: {len(raw)} -> {len(samples)} 条 SFT 样本")
    return samples


def main():
    all_train, all_dev = [], []

    for dataset in ["spider", "bird"]:
        all_train.extend(process_dataset(dataset, "train"))
        all_dev.extend(process_dataset(dataset, "dev"))

    random.shuffle(all_train)

    for split, data in [("train", all_train), ("dev", all_dev)]:
        out = OUTPUT_DIR / f"{split}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"SFT {split}: {len(data)} 条 -> {out}")

    # 同时输出 jsonl 格式（兼容 LLaMA-Factory）
    for split, data in [("train", all_train), ("dev", all_dev)]:
        out = OUTPUT_DIR / f"{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"SFT {split} (jsonl): {len(data)} 条 -> {out}")


if __name__ == "__main__":
    main()
