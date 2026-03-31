"""
Spider 数据集评测脚本

用法：
  python eval/eval_spider.py \
    --model_path ./output/parasql-1.5b \
    --backend vllm \
    --split dev \
    --batch_size 16
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.engine import build_engine
from eval.metrics import batch_execution_accuracy, batch_exact_match


SPIDER_DB_DIR = Path("data/raw/spider/database")
PROCESSED_DIR = Path("data/processed/spider")


def load_data(split: str):
    data_file = PROCESSED_DIR / f"{split}.json"
    with open(data_file, encoding="utf-8") as f:
        return json.load(f)


def get_db_path(db_id: str) -> str:
    return str(SPIDER_DB_DIR / db_id / f"{db_id}.sqlite")


def run_eval(args):
    print(f"加载模型: {args.model_path}")
    engine = build_engine(
        args.model_path,
        backend=args.backend,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    data = load_data(args.split)
    print(f"评测 Spider {args.split}: {len(data)} 条")

    predictions = []
    gold_sqls = []
    db_paths = []

    # 批量推理
    for i in tqdm(range(0, len(data), args.batch_size), desc="推理中"):
        batch = data[i:i + args.batch_size]
        batch_preds = engine.batch_generate(batch)
        predictions.extend(batch_preds)
        gold_sqls.extend([item["query"] for item in batch])
        db_paths.extend([get_db_path(item["db_id"]) for item in batch])

    # 计算指标
    ex_acc, ex_results = batch_execution_accuracy(predictions, gold_sqls, db_paths)
    em_acc, em_results = batch_exact_match(predictions, gold_sqls)

    print(f"\n=== Spider {args.split} 结果 ===")
    print(f"EX (Execution Accuracy): {ex_acc:.4f} ({sum(ex_results)}/{len(ex_results)})")
    print(f"EM (Exact Match):        {em_acc:.4f} ({sum(em_results)}/{len(em_results)})")

    # 保存详细结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, item in enumerate(data):
        results.append({
            "db_id": item["db_id"],
            "question": item["question"],
            "gold_sql": gold_sqls[i],
            "pred_sql": predictions[i],
            "ex": ex_results[i],
            "em": em_results[i],
        })

    out_file = output_dir / f"spider_{args.split}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"ex": ex_acc, "em": em_acc, "details": results}, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存到 {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_dir", default="./eval_results")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
