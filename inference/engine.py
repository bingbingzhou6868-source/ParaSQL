"""
ParaSQL 推理引擎

支持两种模式：
1. vLLM 模式（高效并行推理，推荐）
2. HuggingFace 模式（调试用）

对应 ParaThinker Section 4.3 的 Inference Engine。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .sql_assembler import parse_model_output
from ..prompts.parallel_clause import build_chat_messages, SYSTEM_PROMPT


class ParaSQLEngine:
    """
    ParaSQL 推理引擎基类
    """

    def generate(self, schema: str, question: str, evidence: str = "", db_id: str = "") -> str:
        raise NotImplementedError

    def batch_generate(self, items: List[Dict]) -> List[str]:
        raise NotImplementedError


class VLLMParaSQLEngine(ParaSQLEngine):
    """
    基于 vLLM 的高效推理引擎。
    利用 PagedAttention 实现 KV cache 复用（对应 ParaThinker 的 KV-cache reuse）。
    """

    def __init__(
        self,
        model_path: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("请安装 vllm: pip install vllm")

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["</summary>"],
        )

        from vllm import LLM
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _build_prompt(self, schema: str, question: str, evidence: str = "", db_id: str = "") -> str:
        messages = build_chat_messages(schema, question, evidence, db_id)
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{SYSTEM_PROMPT}\n\nUser: {messages[1]['content']}\nAssistant:"
        return prompt

    def generate(self, schema: str, question: str, evidence: str = "", db_id: str = "") -> str:
        prompt = self._build_prompt(schema, question, evidence, db_id)
        outputs = self.llm.generate([prompt], self.sampling_params)
        raw_output = outputs[0].outputs[0].text
        final_sql, _ = parse_model_output(raw_output)
        return final_sql

    def batch_generate(self, items: List[Dict]) -> List[str]:
        """批量推理，充分利用 vLLM 的并行能力"""
        prompts = [
            self._build_prompt(
                item["schema"], item["question"],
                item.get("evidence", ""), item.get("db_id", "")
            )
            for item in items
        ]
        outputs = self.llm.generate(prompts, self.sampling_params)
        results = []
        for out in outputs:
            raw = out.outputs[0].text
            sql, _ = parse_model_output(raw)
            results.append(sql)
        return results


class HFParaSQLEngine(ParaSQLEngine):
    """
    基于 HuggingFace Transformers 的推理引擎（调试/小规模用）。
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        device: str = "cuda",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

    def _build_prompt(self, schema: str, question: str, evidence: str = "", db_id: str = "") -> str:
        messages = build_chat_messages(schema, question, evidence, db_id)
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{SYSTEM_PROMPT}\n\nUser: {messages[1]['content']}\nAssistant:"

    def generate(self, schema: str, question: str, evidence: str = "", db_id: str = "") -> str:
        import torch
        prompt = self._build_prompt(schema, question, evidence, db_id)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        do_sample = self.temperature > 0
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        final_sql, _ = parse_model_output(raw_output)
        return final_sql

    def batch_generate(self, items: List[Dict]) -> List[str]:
        return [
            self.generate(
                item["schema"], item["question"],
                item.get("evidence", ""), item.get("db_id", "")
            )
            for item in items
        ]


def build_engine(model_path: str, backend: str = "vllm", **kwargs) -> ParaSQLEngine:
    """工厂函数，根据 backend 创建推理引擎"""
    if backend == "vllm":
        return VLLMParaSQLEngine(model_path, **kwargs)
    elif backend == "hf":
        return HFParaSQLEngine(model_path, **kwargs)
    else:
        raise ValueError(f"未知 backend: {backend}，支持 'vllm' 或 'hf'")
