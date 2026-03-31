"""
子句特定位置嵌入（Clause-Specific Positional Embedding）

对应 ParaThinker 论文 Section 3.4 的 Thought-Specific Positional Embedding。
每个 SQL 子句有独立的可学习嵌入，注入到 key/value 中，
解决多子句并行时的位置歧义问题。

公式（同论文 Eq.4/5）：
  k̃_t^(j) = R_t(k_t^(j) + T^(j))
  ṽ_t^(j) = v_t^(j) + T^(j)
"""

import torch
import torch.nn as nn
from typing import List


# SQL 子句顺序（与 build_sft_data.py 保持一致）
SQL_CLAUSES = ["select", "from", "join", "where", "group_by", "having", "order_by", "limit"]
NUM_CLAUSES = len(SQL_CLAUSES)  # 8


class ClauseEmbedding(nn.Module):
    """
    为每个 SQL 子句维护一个可学习的嵌入向量。
    在 summarization 阶段注入到 KV cache，帮助模型区分各子句来源。
    """

    def __init__(self, num_clauses: int = NUM_CLAUSES, hidden_size: int = 2048):
        super().__init__()
        self.num_clauses = num_clauses
        self.hidden_size = hidden_size
        # T^(j)，j=0 保留给 prompt/summary 本身
        self.embeddings = nn.Embedding(num_clauses + 1, hidden_size)
        nn.init.normal_(self.embeddings.weight, std=0.02)

    def get(self, clause_idx: int) -> torch.Tensor:
        """返回第 clause_idx 个子句的嵌入向量，shape: (hidden_size,)"""
        idx = torch.tensor(clause_idx, dtype=torch.long, device=self.embeddings.weight.device)
        return self.embeddings(idx)

    def inject_kv(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        clause_idx: int
    ):
        """
        将子句嵌入注入到 key/value 张量中。
        keys/values shape: (batch, heads, seq_len, head_dim)
        注意：RoPE 旋转在注入之后应用（见论文 Eq.4）
        """
        T = self.get(clause_idx)  # (hidden_size,)
        head_dim = keys.shape[-1]
        # 将 hidden_size 维嵌入投影到 head_dim（简单截断或线性投影）
        if T.shape[0] != head_dim:
            T = T[:head_dim]
        T = T.view(1, 1, 1, head_dim)
        keys_injected = keys + T
        values_injected = values + T
        return keys_injected, values_injected

    def forward(self, clause_indices: torch.Tensor) -> torch.Tensor:
        """batch 查询，clause_indices: (batch,) -> (batch, hidden_size)"""
        return self.embeddings(clause_indices)
