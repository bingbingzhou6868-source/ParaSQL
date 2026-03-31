"""
ParaSQL 反思阶段 RL 训练器

流程：
  Stage 1 (已有): 并行子句生成 → 汇总 SQL
  Stage 2 (本模块): 反思修正
    - 将 Stage 1 生成的 SQL + 执行结果 + 错误信息 输入模型
    - 模型输出修正后的 SQL（带 <reasoning> 推理过程 + <answer> 最终 SQL）
    - 用 GRPO 算法训练，奖励 = 执行F1 + 格式 + 语法 + Schema Jaccard + N-gram

奖励设计（对应图片）：
  Ground Truth 奖励（不可替代）：
    - 执行正确性：F1 软评分（部分匹配也得分，非 0/1）
  代理奖励（辅助平滑梯度）：
    - 格式规范：<reasoning> + <answer> 标签是否完整
    - 语法正确：SQL 能否被 parser 解析
    - Schema 匹配：表名列名 Jaccard 相似度
    - N-gram 相似度：bigram 重合度
"""

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from tqdm import tqdm

from .reward import compute_reward
from ...inference.sql_assembler import parse_model_output


# ── 反思 prompt 模板 ──────────────────────────────────────────────────────
REFLECTION_SYSTEM = """You are a SQL expert. You previously generated a SQL query, but it may contain errors.
Given the original question, database schema, the SQL you generated, and its execution result,
reflect on potential mistakes and generate a corrected SQL query.

Output format (strictly follow):
<reasoning>
[Your step-by-step analysis of what went wrong and how to fix it]
</reasoning>
<answer>
[The corrected SQL query]
</answer>"""


def build_reflection_prompt(
    question: str,
    schema: str,
    pred_sql: str,
    exec_result: str,
    exec_error: str = "",
    evidence: str = "",
) -> str:
    """构建反思阶段的输入 prompt"""
    parts = [f"### Database Schema:\n{schema}"]
    if evidence:
        parts.append(f"### External Knowledge:\n{evidence}")
    parts.append(f"### Question:\n{question}")
    parts.append(f"### Your Previous SQL:\n```sql\n{pred_sql}\n```")

    if exec_error:
        parts.append(f"### Execution Error:\n{exec_error}")
    else:
        result_str = exec_result[:500] if len(exec_result) > 500 else exec_result
        parts.append(f"### Execution Result:\n{result_str}")

    parts.append(
        "\nReflect on the SQL and provide a corrected version. "
        "Use <reasoning>...</reasoning> for your analysis and <answer>...</answer> for the final SQL."
    )
    return "\n\n".join(parts)


def extract_reflection_output(text: str) -> Tuple[str, str]:
    """
    从反思输出中提取 reasoning 和 answer。
    返回 (reasoning, corrected_sql)
    """
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
    answer_match    = re.search(r'<answer>(.*?)</answer>',    text, re.DOTALL)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer    = answer_match.group(1).strip()    if answer_match    else ""

    # fallback: 如果没有 <answer> 标签，尝试直接提取 SQL
    if not answer:
        sql_match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        answer = sql_match.group(1).strip() if sql_match else text.strip()

    return reasoning, answer


# ══════════════════════════════════════════════════════════════════════════
# 数据集
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RLReflectionSample:
    """单条 RL 训练样本"""
    question:    str
    schema:      str
    evidence:    str
    pred_sql:    str      # Stage 1 生成的 SQL（可能有错）
    gold_sql:    str      # 标准答案
    db_path:     str
    exec_result: str      # pred_sql 的执行结果
    exec_error:  str      # 执行错误信息（如有）


class ReflectionRLDataset(Dataset):
    """
    反思 RL 训练数据集。
    从 Stage 1 的推理结果中筛选出需要反思的样本（执行结果不正确的）。
    """

    def __init__(
        self,
        stage1_results_path: str,   # Stage 1 推理结果 JSON
        processed_data_path: str,   # 预处理后的数据（含 schema、gold_sql）
        db_base_dir: str,
        dataset: str = "bird",      # "bird" or "spider"
        only_wrong: bool = True,    # 只训练错误样本
    ):
        self.samples: List[RLReflectionSample] = []
        self._load(stage1_results_path, processed_data_path, db_base_dir, dataset, only_wrong)

    def _get_db_path(self, db_id: str, db_base_dir: str, dataset: str) -> str:
        if dataset == "bird":
            return os.path.join(db_base_dir, db_id, f"{db_id}.sqlite")
        else:
            return os.path.join(db_base_dir, db_id, f"{db_id}.sqlite")

    def _execute_sql(self, db_path: str, sql: str) -> Tuple[str, str]:
        """执行 SQL，返回 (result_str, error_str)"""
        try:
            conn = sqlite3.connect(db_path)
            conn.text_factory = lambda b: b.decode(errors="ignore")
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchmany(10)  # 最多取 10 行
            conn.close()
            return str(rows), ""
        except Exception as e:
            return "", str(e)

    def _load(self, stage1_path, data_path, db_base_dir, dataset, only_wrong):
        with open(stage1_path, encoding="utf-8") as f:
            stage1 = json.load(f)
        with open(data_path, encoding="utf-8") as f:
            raw_data = json.load(f)

        # 建立 question -> raw_data 的映射
        data_map = {item["question"]: item for item in raw_data}

        for item in tqdm(stage1.get("details", stage1), desc="加载反思数据"):
            question = item.get("question", "")
            pred_sql = item.get("pred_sql", "")
            gold_sql = item.get("gold_sql", "")
            db_id    = item.get("db_id", "")
            ex_correct = item.get("ex", False)

            if only_wrong and ex_correct:
                continue  # 跳过已经正确的样本

            raw = data_map.get(question, {})
            schema   = raw.get("schema", "")
            evidence = raw.get("evidence", "")
            db_path  = self._get_db_path(db_id, db_base_dir, dataset)

            exec_result, exec_error = self._execute_sql(db_path, pred_sql)

            self.samples.append(RLReflectionSample(
                question=question,
                schema=schema,
                evidence=evidence,
                pred_sql=pred_sql,
                gold_sql=gold_sql,
                db_path=db_path,
                exec_result=exec_result,
                exec_error=exec_error,
            ))

        print(f"反思 RL 数据集: {len(self.samples)} 条样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ══════════════════════════════════════════════════════════════════════════
# GRPO 训练器
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class GRPOConfig:
    """GRPO 训练配置"""
    model_name_or_path: str = field(default="./output/parasql-sft")
    output_dir:         str = field(default="./output/parasql-rl")

    # 数据
    stage1_results_path:  str = field(default="eval_results/bird_dev_results.json")
    processed_data_path:  str = field(default="data/processed/bird/train.json")
    db_base_dir:          str = field(default="data/raw/bird/train/train_databases")
    dataset:              str = field(default="bird")

    # GRPO 超参
    num_generations:    int   = field(default=8)     # 每个 prompt 采样 G 个输出
    temperature:        float = field(default=0.8)
    max_new_tokens:     int   = field(default=512)
    learning_rate:      float = field(default=5e-6)
    num_train_epochs:   int   = field(default=2)
    per_device_batch:   int   = field(default=1)
    grad_accum:         int   = field(default=8)
    kl_coef:            float = field(default=0.04)  # KL 散度惩罚系数
    clip_ratio:         float = field(default=0.2)   # PPO clip ratio

    # 奖励权重
    w_exec:   float = field(default=1.0)
    w_format: float = field(default=0.1)
    w_syntax: float = field(default=0.1)
    w_schema: float = field(default=0.2)
    w_ngram:  float = field(default=0.1)

    # 其他
    save_steps:   int  = field(default=100)
    logging_steps: int = field(default=10)
    bf16:         bool = field(default=True)
    use_lora:     bool = field(default=True)
    lora_r:       int  = field(default=16)
    lora_alpha:   int  = field(default=32)


class GRPOReflectionTrainer:
    """
    GRPO (Group Relative Policy Optimization) 反思训练器。

    GRPO 核心思想（来自 DeepSeek-R1）：
      对每个 prompt，采样 G 个输出 {o₁,...,oG}
      计算每个输出的奖励 {r₁,...,rG}
      用组内相对奖励作为优势估计：Aᵢ = (rᵢ - mean(r)) / std(r)
      用 PPO-clip 目标更新策略，同时加 KL 惩罚防止偏离参考策略
    """

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.reward_weights = {
            "exec":   config.w_exec,
            "format": config.w_format,
            "syntax": config.w_syntax,
            "schema": config.w_schema,
            "ngram":  config.w_ngram,
        }
        self._setup_model()

    def _setup_model(self):
        cfg = self.config
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        # 参考策略（冻结，用于 KL 惩罚）
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        if cfg.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            self.model.print_trainable_parameters()

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
        )

    def _build_prompt(self, sample: RLReflectionSample) -> str:
        """构建反思 prompt"""
        messages = [
            {"role": "system",  "content": REFLECTION_SYSTEM},
            {"role": "user",    "content": build_reflection_prompt(
                sample.question, sample.schema, sample.pred_sql,
                sample.exec_result, sample.exec_error, sample.evidence
            )},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{REFLECTION_SYSTEM}\n\nUser: {messages[1]['content']}\nAssistant:"

    @torch.no_grad()
    def _sample_outputs(self, prompt: str) -> List[str]:
        """对单个 prompt 采样 G 个输出"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=3072).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            num_return_sequences=self.config.num_generations,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        texts = []
        for out in outputs:
            new_ids = out[prompt_len:]
            texts.append(self.tokenizer.decode(new_ids, skip_special_tokens=False))
        return texts

    def _compute_rewards(
        self, outputs: List[str], sample: RLReflectionSample
    ) -> List[float]:
        """计算每个输出的综合奖励"""
        rewards = []
        for output in outputs:
            _, corrected_sql = extract_reflection_output(output)
            r = compute_reward(
                model_output=output,
                pred_sql=corrected_sql,
                gold_sql=sample.gold_sql,
                db_path=sample.db_path,
                weights=self.reward_weights,
            )
            rewards.append(r["total"])
        return rewards

    def _compute_log_probs(self, model, prompt: str, output: str) -> torch.Tensor:
        """计算模型对 output 的 log 概率（用于 PPO 目标）"""
        full_text = prompt + output
        inputs = self.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=4096
        ).to(model.device)
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3072
        )["input_ids"]
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad() if model is self.ref_model else torch.enable_grad():
            logits = model(**inputs).logits  # (1, seq_len, vocab)

        # 只计算 completion 部分的 log prob
        log_probs = torch.log_softmax(logits[0, prompt_len-1:-1], dim=-1)
        target_ids = inputs["input_ids"][0, prompt_len:]
        if len(target_ids) == 0:
            return torch.tensor(0.0, device=model.device)
        token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        return token_log_probs.sum()

    def _grpo_loss(
        self,
        prompt: str,
        outputs: List[str],
        rewards: List[float],
    ) -> torch.Tensor:
        """
        GRPO 损失计算。

        L_GRPO = -E[ A_i * clip(π/π_old, 1-ε, 1+ε) ] + β * KL(π || π_ref)

        其中优势 A_i = (r_i - mean(r)) / (std(r) + 1e-8)
        """
        import numpy as np
        rewards_arr = np.array(rewards, dtype=np.float32)

        # 组内相对优势
        mean_r = rewards_arr.mean()
        std_r  = rewards_arr.std() + 1e-8
        advantages = (rewards_arr - mean_r) / std_r

        total_loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)

        for output, advantage in zip(outputs, advantages):
            if abs(advantage) < 1e-6:
                continue

            # 当前策略 log prob
            log_prob_new = self._compute_log_probs(self.model, prompt, output)
            # 参考策略 log prob（用于 KL）
            log_prob_ref = self._compute_log_probs(self.ref_model, prompt, output)

            # PPO-clip 目标
            ratio = torch.exp(log_prob_new - log_prob_ref.detach())
            adv_t = torch.tensor(advantage, device=self.model.device, dtype=torch.float32)
            clipped = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
            policy_loss = -torch.min(ratio * adv_t, clipped * adv_t)

            # KL 惩罚
            kl = log_prob_new - log_prob_ref.detach()
            kl_loss = self.config.kl_coef * kl

            total_loss = total_loss + policy_loss + kl_loss

        return total_loss / max(len(outputs), 1)

    def train(self, dataset: ReflectionRLDataset):
        """主训练循环"""
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        self.model.train()
        global_step = 0
        total_reward_log = []

        for epoch in range(cfg.num_train_epochs):
            print(f"\n=== Epoch {epoch+1}/{cfg.num_train_epochs} ===")

            for step, sample in enumerate(tqdm(dataset, desc=f"Epoch {epoch+1}")):
                prompt = self._build_prompt(sample)

                # 1. 采样 G 个输出
                self.model.eval()
                with torch.no_grad():
                    outputs = self._sample_outputs(prompt)
                self.model.train()

                # 2. 计算奖励
                rewards = self._compute_rewards(outputs, sample)
                mean_reward = sum(rewards) / len(rewards)
                total_reward_log.append(mean_reward)

                # 3. 如果所有输出奖励相同，跳过（无法计算优势）
                if max(rewards) - min(rewards) < 1e-6:
                    continue

                # 4. GRPO 损失 + 反向传播
                loss = self._grpo_loss(prompt, outputs, rewards)

                loss = loss / cfg.grad_accum
                loss.backward()

                if (step + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if global_step % cfg.logging_steps == 0:
                        avg_r = sum(total_reward_log[-cfg.logging_steps:]) / cfg.logging_steps
                        print(f"  Step {global_step} | loss={loss.item()*cfg.grad_accum:.4f} "
                              f"| avg_reward={avg_r:.4f}")

                    if global_step % cfg.save_steps == 0:
                        save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        self.model.save_pretrained(save_path)
                        self.tokenizer.save_pretrained(save_path)
                        print(f"  已保存: {save_path}")

        # 保存最终模型
        self.model.save_pretrained(cfg.output_dir)
        self.tokenizer.save_pretrained(cfg.output_dir)
        print(f"\n训练完成，模型已保存到: {cfg.output_dir}")


def main():
    parser = HfArgumentParser(GRPOConfig)
    config, = parser.parse_args_into_dataclasses()

    dataset = ReflectionRLDataset(
        stage1_results_path=config.stage1_results_path,
        processed_data_path=config.processed_data_path,
        db_base_dir=config.db_base_dir,
        dataset=config.dataset,
        only_wrong=True,
    )

    trainer = GRPOReflectionTrainer(config)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
