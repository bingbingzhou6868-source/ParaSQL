"""
ParaSQL 模型封装

在 HuggingFace Transformers 基础上，添加：
1. 子句特定位置嵌入（ClauseEmbedding）
2. 两阶段注意力掩码
3. 并行子句生成 + 汇总的 forward 逻辑
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from typing import Dict, List, Optional, Tuple

from .thought_embedding import ClauseEmbedding, SQL_CLAUSES
from .attention_mask import build_full_training_mask


# 特殊控制 token
CLAUSE_TOKENS = {
    clause: (f"<clause_{clause}>", f"</clause_{clause}>")
    for clause in SQL_CLAUSES
}
SUMMARY_TOKENS = ("<summary>", "</summary>")

ALL_SPECIAL_TOKENS = (
    [t for pair in CLAUSE_TOKENS.values() for t in pair]
    + list(SUMMARY_TOKENS)
)


class ParaSQLModel(nn.Module):
    """
    ParaSQL：并行 SQL 子句生成模型

    基于任意 CausalLM，添加子句嵌入层。
    训练时使用两阶段注意力掩码。
    """

    def __init__(
        self,
        base_model_name: str,
        num_clauses: int = len(SQL_CLAUSES),
        use_clause_embedding: bool = True,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # 添加特殊 token
        self.tokenizer.add_special_tokens({"additional_special_tokens": ALL_SPECIAL_TOKENS})
        self.model.resize_token_embeddings(len(self.tokenizer))

        # 子句嵌入
        self.use_clause_embedding = use_clause_embedding
        if use_clause_embedding:
            hidden_size = self.model.config.hidden_size
            self.clause_embedding = ClauseEmbedding(num_clauses, hidden_size)

        self.num_clauses = num_clauses

    def get_special_token_ids(self) -> Dict[str, int]:
        """返回所有特殊 token 的 id"""
        ids = {}
        for clause, (open_t, close_t) in CLAUSE_TOKENS.items():
            ids[f"open_{clause}"] = self.tokenizer.convert_tokens_to_ids(open_t)
            ids[f"close_{clause}"] = self.tokenizer.convert_tokens_to_ids(close_t)
        ids["open_summary"] = self.tokenizer.convert_tokens_to_ids(SUMMARY_TOKENS[0])
        ids["close_summary"] = self.tokenizer.convert_tokens_to_ids(SUMMARY_TOKENS[1])
        return ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        clause_segment_ids: Optional[torch.Tensor] = None,  # 每个 token 属于哪个子句（-1=prompt/summary）
        **kwargs,
    ):
        """
        训练 forward。
        clause_segment_ids: (batch, seq_len)，值为 0~num_clauses-1 或 -1
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
            **kwargs,
        )
        return outputs

    def save_pretrained(self, save_dir: str):
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        if self.use_clause_embedding:
            torch.save(self.clause_embedding.state_dict(), f"{save_dir}/clause_embedding.pt")

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        import os
        obj = cls(save_dir, **kwargs)
        emb_path = os.path.join(save_dir, "clause_embedding.pt")
        if os.path.exists(emb_path) and obj.use_clause_embedding:
            obj.clause_embedding.load_state_dict(torch.load(emb_path, map_location="cpu"))
        return obj
