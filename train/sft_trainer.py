"""
ParaSQL SFT 训练脚本

使用 HuggingFace Trainer + PEFT (LoRA) 进行监督微调。
训练数据格式见 data/build_sft_data.py。
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts.parallel_clause import SYSTEM_PROMPT, SQL_CLAUSES


# ── 特殊 token ──────────────────────────────────────────────────────────────
ALL_SPECIAL_TOKENS = (
    [f"<clause_{c}>" for c in SQL_CLAUSES]
    + [f"</clause_{c}>" for c in SQL_CLAUSES]
    + ["<summary>", "</summary>"]
)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    use_lora: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="q_proj,v_proj,k_proj,o_proj")


@dataclass
class DataArguments:
    train_file: str = field(default="data/sft/train.jsonl")
    dev_file: str = field(default="data/sft/dev.jsonl")
    max_length: int = field(default=4096)


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def tokenize_sample(
    sample: Dict,
    tokenizer,
    max_length: int,
) -> Dict:
    """将一条 SFT 样本 tokenize，并构建 labels（只对 assistant 部分计算 loss）"""
    system = sample.get("system", SYSTEM_PROMPT)
    convs = sample["conversations"]
    user_content = convs[0]["content"]
    assistant_content = convs[1]["content"]

    # 构建完整文本
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        # 只有 prompt 部分（不含 assistant）
        prompt_messages = messages[:2]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = f"{system}\n\nUser: {user_content}\nAssistant:"
        full_text = prompt_text + assistant_content + tokenizer.eos_token

    full_ids = tokenizer(full_text, truncation=True, max_length=max_length)["input_ids"]
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]
    prompt_len = len(prompt_ids)

    labels = [-100] * prompt_len + full_ids[prompt_len:]
    labels = labels[:max_length]
    full_ids = full_ids[:max_length]

    # padding 到 max_length
    pad_len = max_length - len(full_ids)
    attention_mask = [1] * len(full_ids) + [0] * pad_len
    full_ids = full_ids + [tokenizer.pad_token_id or 0] * pad_len
    labels = labels + [-100] * pad_len

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ALL_SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ──────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    if model_args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules.split(","),
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ── Dataset ────────────────────────────────────────────────────────────
    train_raw = load_jsonl(data_args.train_file)
    dev_raw = load_jsonl(data_args.dev_file)

    def tokenize_fn(sample):
        return tokenize_sample(sample, tokenizer, data_args.max_length)

    train_dataset = Dataset.from_list(train_raw).map(
        tokenize_fn, remove_columns=train_raw[0].keys() if train_raw else []
    )
    dev_dataset = Dataset.from_list(dev_raw).map(
        tokenize_fn, remove_columns=dev_raw[0].keys() if dev_raw else []
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, padding=True, pad_to_multiple_of=8
        ),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"模型已保存到 {training_args.output_dir}")


if __name__ == "__main__":
    main()
