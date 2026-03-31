"""
两阶段注意力掩码（Two-Phase Attention Mask）

对应 ParaThinker 论文 Section 4.2。

阶段一（并行子句生成）：
  每个子句 token 只能 attend 到：
    - 输入 prompt（schema + question）
    - 本子句自身的历史 token
  不能 attend 到其他子句（强制独立）

阶段二（汇总生成完整 SQL）：
  summary token 可以 attend 到：
    - 输入 prompt
    - 所有子句的 token
    - 已生成的 summary token
"""

import torch
from typing import List, Tuple


def build_parallel_clause_mask(
    prompt_len: int,
    clause_lens: List[int],
    device: str = "cpu"
) -> torch.Tensor:
    """
    构建并行子句生成阶段的注意力掩码。

    Args:
        prompt_len: 输入 prompt 的 token 数
        clause_lens: 每个子句的 token 数列表，长度 = num_clauses
        device: 设备

    Returns:
        mask: (total_len, total_len) 的 float 掩码，
              0 表示可以 attend，-inf 表示不可 attend
    """
    num_clauses = len(clause_lens)
    total_len = prompt_len + sum(clause_lens)
    mask = torch.full((total_len, total_len), float("-inf"), device=device)

    # prompt 部分：因果掩码（prompt tokens 互相可见）
    for i in range(prompt_len):
        mask[i, :i + 1] = 0.0

    # 每个子句：只能看 prompt + 自身历史
    offset = prompt_len
    for clause_idx, clause_len in enumerate(clause_lens):
        for t in range(clause_len):
            pos = offset + t
            # 可以 attend 到整个 prompt
            mask[pos, :prompt_len] = 0.0
            # 可以 attend 到本子句的历史（含自身）
            mask[pos, offset:offset + t + 1] = 0.0
        offset += clause_len

    return mask


def build_summary_mask(
    prompt_len: int,
    clause_lens: List[int],
    summary_len: int,
    device: str = "cpu"
) -> torch.Tensor:
    """
    构建汇总阶段的注意力掩码。

    summary token 可以 attend 到 prompt + 所有子句 + 已生成 summary。
    """
    total_clause_len = sum(clause_lens)
    context_len = prompt_len + total_clause_len
    total_len = context_len + summary_len
    mask = torch.full((summary_len, total_len), float("-inf"), device=device)

    for t in range(summary_len):
        # attend 到 prompt
        mask[t, :prompt_len] = 0.0
        # attend 到所有子句
        mask[t, prompt_len:context_len] = 0.0
        # attend 到已生成的 summary（因果）
        mask[t, context_len:context_len + t + 1] = 0.0

    return mask


def build_full_training_mask(
    prompt_len: int,
    clause_lens: List[int],
    summary_len: int,
    device: str = "cpu"
) -> torch.Tensor:
    """
    训练时的完整注意力掩码（并行子句 + 汇总）。

    Returns:
        mask: (total_len, total_len)
    """
    total_clause_len = sum(clause_lens)
    total_len = prompt_len + total_clause_len + summary_len

    mask = torch.full((total_len, total_len), float("-inf"), device=device)

    # --- 阶段一：prompt 因果掩码 ---
    for i in range(prompt_len):
        mask[i, :i + 1] = 0.0

    # --- 阶段一：并行子句掩码 ---
    offset = prompt_len
    for clause_len in clause_lens:
        for t in range(clause_len):
            pos = offset + t
            mask[pos, :prompt_len] = 0.0
            mask[pos, offset:offset + t + 1] = 0.0
        offset += clause_len

    # --- 阶段二：汇总掩码 ---
    context_len = prompt_len + total_clause_len
    for t in range(summary_len):
        pos = context_len + t
        mask[pos, :prompt_len] = 0.0
        mask[pos, prompt_len:context_len] = 0.0
        mask[pos, context_len:context_len + t + 1] = 0.0

    return mask
