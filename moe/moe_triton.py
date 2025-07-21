import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import triton
import triton.language as tl
from moe.moe_torch import SwiGLUMoETorch
import pdb

# class SwiGLUExpert(nn.Module):
#     def __init__(self, hidden_dim, moe_inner_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(hidden_dim, moe_inner_dim * 2)
#         self.fc2 = nn.Linear(moe_inner_dim, hidden_dim)

#     def forward(self, x):
#         gate, value = self.fc1(x).chunk(2, dim=-1)
#         gate = gate * torch.sigmoid(gate)
#         fc1_output = gate * value
#         fc2_output = self.fc2(fc1_output)
#         return fc2_output


def get_flat_token_and_score(top_k_indices, top_k_scores, num_experts, top_k, device):

    # To get tokens per expert:
    expert_to_tokens = [[] for _ in range(num_experts)]
    expert_to_scores = [[] for _ in range(num_experts)]

    # For each token position
    for token_idx in range(top_k_indices.size(0)):
        # For each selected expert for this token
        for k in range(top_k):
            expert_idx = top_k_indices[token_idx, k].item()
            score = top_k_scores[token_idx, k].item()

            # Add this token to the expert's list
            expert_to_tokens[expert_idx].append(token_idx)
            expert_to_scores[expert_idx].append(score)

    # Create cumulative sequence lengths for easy indexing
    cu_seq_lens = [0]
    running_sum = 0

    # Convert lists to tensors for each expert
    for i in range(num_experts):
        if expert_to_tokens[i]:  # Check if this expert has any tokens
            expert_to_tokens[i] = torch.tensor(
                expert_to_tokens[i], dtype=torch.int64, device=device
            )
            expert_to_scores[i] = torch.tensor(
                expert_to_scores[i], dtype=torch.float32, device=device
            )
        else:
            # Handle empty expert lists with empty tensors of proper type
            expert_to_tokens[i] = torch.tensor([], dtype=torch.int64, device=device)
            expert_to_scores[i] = torch.tensor([], dtype=torch.float32, device=device)

        # Update cumulative lengths
        running_sum += len(expert_to_tokens[i])
        cu_seq_lens.append(running_sum)

    # Convert cu_seq_lens to tensor
    cu_seq_lens = torch.tensor(cu_seq_lens, dtype=torch.int64, device=device)

    # Flatten the lists using torch.cat
    expert_to_tokens_flat = torch.cat(
        [tensor for tensor in expert_to_tokens if tensor.numel() > 0], dim=0
    )
    expert_to_scores_flat = torch.cat(
        [tensor for tensor in expert_to_scores if tensor.numel() > 0], dim=0
    )
    return cu_seq_lens, expert_to_tokens_flat, expert_to_scores_flat


# Define a list of configurations to try
configs = [
    # triton.Config({'BLOCK_SIZE_T': 32, 'BLOCK_SIZE_D': 32, 'BLOCK_SIZE_dD': 32, 'BLOCK_SIZE_d': 32}, num_warps=4),
    # triton.Config({'BLOCK_SIZE_T': 64, 'BLOCK_SIZE_D': 64, 'BLOCK_SIZE_dD': 64, 'BLOCK_SIZE_d': 64}, num_warps=8),
    triton.Config(
        {
            "BLOCK_SIZE_T": 128,
            "BLOCK_SIZE_D": 64,
            "BLOCK_SIZE_dD": 128,
            "BLOCK_SIZE_d": 64,
        },
        num_warps=4,
    ),
    # triton.Config({'BLOCK_SIZE_T': 32, 'BLOCK_SIZE_D': 128, 'BLOCK_SIZE_dD': 32, 'BLOCK_SIZE_d': 32}, num_warps=4),
]


@triton.autotune(configs=configs, key=["d", "D", "dD", "num_experts"])
@triton.jit
def fused_fwd_kernel(
    x,
    fc11,
    fc12,
    fc2,
    o,
    cu_seq_lens,
    n_tokens,
    n_score,
    stride_xt,
    stride_xd,
    stride_fc11_n,
    stride_fc11_d,
    stride_fc11_D,
    stride_fc12_n,
    stride_fc12_d,
    stride_fc12_D,
    stride_fc2_n,
    stride_fc2_D,
    stride_fc2_dD,
    stride_o_t,
    stride_o_dD,
    d: tl.constexpr,
    D: tl.constexpr,
    dD: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_d: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_dD: tl.constexpr,
):
    pid_nt = tl.program_id(0)
    pid_n = pid_nt % num_experts
    pid_t = pid_nt // num_experts
    pid_D = tl.program_id(1)
    pid_dD = tl.program_id(2)
    n_start = tl.load(cu_seq_lens + pid_n)
    n_end = tl.load(cu_seq_lens + pid_n + 1)
    n_len = n_end - n_start

    if pid_t * BLOCK_SIZE_T > n_len:
        return

    t_id_ptrs = tl.make_block_ptr(
        base=n_tokens + n_start,
        shape=(n_len,),
        strides=(1,),
        offsets=(pid_t * BLOCK_SIZE_T,),
        block_shape=(BLOCK_SIZE_T,),
        order=(0,),
    )
    s_ptrs = tl.make_block_ptr(
        base=n_score + n_start,
        shape=(n_len,),
        strides=(1,),
        offsets=(pid_t * BLOCK_SIZE_T,),
        block_shape=(BLOCK_SIZE_T,),
        order=(0,),
    )

    fc11_ptrs = tl.make_block_ptr(
        base=fc11 + pid_n * stride_fc11_n,
        shape=(d, D),
        strides=(stride_fc11_d, stride_fc11_D),
        offsets=(0, pid_D * BLOCK_SIZE_D),
        block_shape=(BLOCK_SIZE_d, BLOCK_SIZE_D),
        order=(1, 0),
    )
    fc12_ptrs = tl.make_block_ptr(
        base=fc12 + pid_n * stride_fc12_n,
        shape=(d, D),
        strides=(stride_fc12_d, stride_fc12_D),
        offsets=(0, pid_D * BLOCK_SIZE_D),
        block_shape=(BLOCK_SIZE_d, BLOCK_SIZE_D),
        order=(1, 0),
    )
    fc2_ptrs = tl.make_block_ptr(
        base=fc2 + pid_n * stride_fc2_n,
        shape=(D, dD),
        strides=(stride_fc2_D, stride_fc2_dD),
        offsets=(pid_D * BLOCK_SIZE_D, pid_dD * BLOCK_SIZE_dD),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_dD),
        order=(1, 0),
    )

    t_id = tl.load(t_id_ptrs, boundary_check=(0,), padding_option="zero")
    b_s = tl.load(s_ptrs, boundary_check=(0,), padding_option="zero")
    x_ptrs_t = x + t_id * stride_xt
    o_ptrs = (
        o
        + (t_id * stride_o_t)[:, None]
        + (
            pid_dD * BLOCK_SIZE_dD * stride_o_dD
            + tl.arange(0, BLOCK_SIZE_dD) * stride_o_dD
        )[None, :]
    )
    o_mask = ((pid_dD * BLOCK_SIZE_dD + tl.arange(0, BLOCK_SIZE_dD)) < dD)[None, :]

    # Initialize separate accumulators for gate and value
    b_g_acc = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_D), dtype=tl.float32)
    b_v_acc = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_D), dtype=tl.float32)

    for i in range(0, d, BLOCK_SIZE_d):
        x_ptrs = (
            x_ptrs_t[:, None] + ((i + tl.arange(0, BLOCK_SIZE_d)) * stride_xd)[None, :]
        )
        x_mask = (i + tl.arange(0, BLOCK_SIZE_d)) < d
        b_x = tl.load(x_ptrs, mask=x_mask[None, :], other=0)
        # b_x : [BLOCK_SIZE_T, BLOCK_SIZE_d]

        b_fc11 = tl.load(fc11_ptrs, boundary_check=(0, 1), padding_option="zero")
        # b_fc11: [BLOCK_SIZE_d, moe_inner_dim]

        b_fc12 = tl.load(fc12_ptrs, boundary_check=(0, 1), padding_option="zero")
        # b_fc12: [BLOCK_SIZE_d, moe_inner_dim]

        # Accumulate partial gate and value calculations
        b_g_acc += tl.dot(b_x, b_fc11)
        b_v_acc += tl.dot(b_x, b_fc12)
        # b_fc1_o += b_v
        fc11_ptrs = tl.advance(fc11_ptrs, (BLOCK_SIZE_d, 0))
        fc12_ptrs = tl.advance(fc12_ptrs, (BLOCK_SIZE_d, 0))

    b_g = b_g_acc * tl.sigmoid(b_g_acc)
    b_fc1_o = b_g * b_v_acc

    b_fc2 = tl.load(fc2_ptrs, boundary_check=(0, 1), padding_option="zero")
    # b_fc2: [moe_inner_dim, BLOCK_SIZE_dD]

    b_o = b_s[:, None] * tl.dot(b_fc1_o.to(b_fc2.dtype), b_fc2)
    # b_o = tl.dot(b_fc1_o, b_fc2)
    # b_o : [BLOCK_SIZE_T, BLOCK_SIZE_dD]

    tl.atomic_add(o_ptrs, b_o.to(o_ptrs.dtype.element_ty), mask=o_mask)


# Define a list of configurations to try
configs = [
    # triton.Config({'BLOCK_SIZE_T': 32, 'BLOCK_SIZE_D': 32, 'BLOCK_SIZE_dD': 32, 'BLOCK_SIZE_d': 32}, num_warps=4),
    # triton.Config({'BLOCK_SIZE_T': 64, 'BLOCK_SIZE_D': 64, 'BLOCK_SIZE_dD': 64, 'BLOCK_SIZE_d': 64}, num_warps=8),
    triton.Config(
        {
            "BLOCK_SIZE_T": 128,
            "BLOCK_SIZE_D": 64,
            "BLOCK_SIZE_dD": 128,
            "BLOCK_SIZE_d": 64,
        },
        num_warps=4,
    ),
    # triton.Config({'BLOCK_SIZE_T': 32, 'BLOCK_SIZE_D': 128, 'BLOCK_SIZE_dD': 32, 'BLOCK_SIZE_d': 32}, num_warps=4),
]


@triton.autotune(configs=configs, key=["d", "D", "dD", "num_experts"])
@triton.jit
def fc2_bwd_kernel(
    x,
    fc11,
    fc12,
    fc1_o_g,
    fc1_o_v,
    cu_seq_lens,
    n_tokens,
    n_score,
    dx,
    dg,
    dfc2,
    do,
    stride_xt,
    stride_xd,
    stride_fc11_n,
    stride_fc11_d,
    stride_fc11_D,
    stride_fc12_n,
    stride_fc12_d,
    stride_fc12_D,
    stride_fc1_og_t,
    stride_fc1_og_n,
    stride_fc1_og_D,
    stride_fc1_ov_t,
    stride_fc1_ov_n,
    stride_fc1_ov_D,
    stride_dfc2_n,
    stride_dfc2_D,
    stride_dfc2_dD,
    stride_do_t,
    stride_do_dD,
    d: tl.constexpr,
    D: tl.constexpr,
    dD: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_d: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_dD: tl.constexpr,
):
    pid_nt = tl.program_id(0)
    pid_n = pid_nt % num_experts
    pid_t = pid_nt // num_experts
    pid_D = tl.program_id(1)
    pid_dD = tl.program_id(2)
    n_start = tl.load(cu_seq_lens + pid_n)
    n_end = tl.load(cu_seq_lens + pid_n + 1)
    n_len = n_end - n_start

    if pid_t * BLOCK_SIZE_T > n_len:
        return

    t_id_ptrs = tl.make_block_ptr(
        base=n_tokens + n_start,
        shape=(n_len,),
        strides=(1,),
        offsets=(pid_t * BLOCK_SIZE_T,),
        block_shape=(BLOCK_SIZE_T,),
        order=(0,),
    )
    s_ptrs = tl.make_block_ptr(
        base=n_score + n_start,
        shape=(n_len,),
        strides=(1,),
        offsets=(pid_t * BLOCK_SIZE_T,),
        block_shape=(BLOCK_SIZE_T,),
        order=(0,),
    )

    fc11_ptrs = tl.make_block_ptr(
        base=fc11 + pid_n * stride_fc11_n,
        shape=(d, D),
        strides=(stride_fc11_d, stride_fc11_D),
        offsets=(0, pid_D * BLOCK_SIZE_D),
        block_shape=(BLOCK_SIZE_d, BLOCK_SIZE_D),
        order=(1, 0),
    )
    fc12_ptrs = tl.make_block_ptr(
        base=fc12 + pid_n * stride_fc12_n,
        shape=(d, D),
        strides=(stride_fc12_d, stride_fc12_D),
        offsets=(0, pid_D * BLOCK_SIZE_D),
        block_shape=(BLOCK_SIZE_d, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dfc2_ptrs = (
        dfc2
        + pid_n * stride_dfc2_n
        + (pid_D * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D))[:, None] * stride_dfc2_D
        + (pid_dD * BLOCK_SIZE_dD + tl.arange(0, BLOCK_SIZE_dD))[None, :]
        * stride_dfc2_dD
    )
    dfc2_mask = ((pid_D * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)) < D)[:, None] & (
        (pid_dD * BLOCK_SIZE_dD + tl.arange(0, BLOCK_SIZE_dD)) < dD
    )[None, :]

    t_id = tl.load(t_id_ptrs, boundary_check=(0,), padding_option="zero")
    b_s = tl.load(s_ptrs, boundary_check=(0,), padding_option="zero")
    x_ptrs_t = x + t_id * stride_xt

    do_ptrs = (
        do
        + (t_id * stride_do_t)[:, None]
        + (
            pid_dD * BLOCK_SIZE_dD * stride_do_dD
            + tl.arange(0, BLOCK_SIZE_dD) * stride_do_dD
        )[None, :]
    )
    do_mask = ((pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)) < n_len)[:, None] & (
        (pid_dD * BLOCK_SIZE_dD + tl.arange(0, BLOCK_SIZE_dD)) < dD
    )[None, :]
    fc1_o_g_ptrs = (
        fc1_o_g
        + pid_n * stride_fc1_og_n
        + (t_id * stride_fc1_og_t)[:, None]
        + (
            pid_D * BLOCK_SIZE_D * stride_fc1_og_D
            + tl.arange(0, BLOCK_SIZE_D) * stride_fc1_og_D
        )[None, :]
    )
    fc1_o_g_mask = ((pid_D * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)) < D)[
        None, :
    ] & ((pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)) < n_len)[:, None]

    fc1_o_v_ptrs = (
        fc1_o_v
        + pid_n * stride_fc1_ov_n
        + (t_id * stride_fc1_ov_t)[:, None]
        + (
            pid_D * BLOCK_SIZE_D * stride_fc1_ov_D
            + tl.arange(0, BLOCK_SIZE_D) * stride_fc1_ov_D
        )[None, :]
    )
    fc1_o_v_mask = ((pid_D * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)) < D)[
        None, :
    ] & ((pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)) < n_len)[:, None]

    # --- Recompute Forward Pass Intermediate Values & Compute Gradients ---
    b_g_acc = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_D), dtype=tl.float32)
    b_v_acc = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_D), dtype=tl.float32)

    for i in range(0, d, BLOCK_SIZE_d):
        x_ptrs = (
            x_ptrs_t[:, None] + ((i + tl.arange(0, BLOCK_SIZE_d)) * stride_xd)[None, :]
        )
        x_mask = ((pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)) < n_len)[
            :, None
        ] & ((i + tl.arange(0, BLOCK_SIZE_d)) < d)[None, :]
        b_x = tl.load(x_ptrs, mask=x_mask, other=0)
        # b_x : [BLOCK_SIZE_T, BLOCK_SIZE_d]

        b_fc11 = tl.load(fc11_ptrs, boundary_check=(0, 1), padding_option="zero")
        # b_fc11: [BLOCK_SIZE_d, moe_inner_dim]

        b_fc12 = tl.load(fc12_ptrs, boundary_check=(0, 1), padding_option="zero")
        # b_fc12: [BLOCK_SIZE_d, moe_inner_dim]

        # Accumulate partial gate and value calculations
        b_g_acc += tl.dot(b_x, b_fc11)
        b_v_acc += tl.dot(b_x, b_fc12)
        # b_fc1_o += b_v
        fc11_ptrs = tl.advance(fc11_ptrs, (BLOCK_SIZE_d, 0))
        fc12_ptrs = tl.advance(fc12_ptrs, (BLOCK_SIZE_d, 0))

    b_g = b_g_acc * tl.sigmoid(b_g_acc)
    b_fc1_o = b_g * b_v_acc
    # [BLOCK_SIZE_T, BLOCK_SIZE_D]

    # --- dfc2 ---
    b_do = tl.load(do_ptrs, mask=do_mask, other=0)
    # [BLOCK_SIZE_T, BLOCK_SIZE_dD]

    b_dfc2 = tl.dot(tl.trans(b_fc1_o), b_do * b_s[:, None])

    tl.store(fc1_o_g_ptrs, b_g_acc.to(fc1_o_g_ptrs.dtype.element_ty), mask=fc1_o_g_mask)
    tl.store(fc1_o_v_ptrs, b_v_acc.to(fc1_o_v_ptrs.dtype.element_ty), mask=fc1_o_v_mask)
    tl.atomic_add(dfc2_ptrs, b_dfc2.to(dfc2_ptrs.dtype.element_ty), mask=dfc2_mask)


# Define a list of configurations to try
configs = [
    # triton.Config({'BLOCK_SIZE_T': 32, 'BLOCK_SIZE_D': 32, 'BLOCK_SIZE_dD': 32, 'BLOCK_SIZE_d': 32}, num_warps=4),
    # triton.Config({'BLOCK_SIZE_T': 64, 'BLOCK_SIZE_D': 64, 'BLOCK_SIZE_dD': 64, 'BLOCK_SIZE_d': 64}, num_warps=8),
    triton.Config(
        {
            "BLOCK_SIZE_T": 128,
            "BLOCK_SIZE_D": 64,
            "BLOCK_SIZE_dD": 128,
            "BLOCK_SIZE_d": 64,
        },
        num_warps=4,
    ),
    # triton.Config({'BLOCK_SIZE_T': 32, 'BLOCK_SIZE_D': 128, 'BLOCK_SIZE_dD': 32, 'BLOCK_SIZE_d': 32}, num_warps=4),
]


@triton.autotune(configs=configs, key=["d", "D", "dD", "num_experts"])
@triton.jit
def fc1_bwd_kernel(
    x,
    fc11,
    fc12,
    fc2,
    fc1_o_g,
    fc1_o_v,
    cu_seq_lens,
    n_tokens,
    n_score,
    dx,
    dg,
    dfc11,
    dfc12,
    do,
    stride_xt,
    stride_xd,
    stride_fc11_n,
    stride_fc11_d,
    stride_fc11_D,
    stride_fc12_n,
    stride_fc12_d,
    stride_fc12_D,
    stride_fc2_n,
    stride_fc2_D,
    stride_fc2_dD,
    stride_fc1_og_t,
    stride_fc1_og_n,
    stride_fc1_og_D,
    stride_fc1_ov_t,
    stride_fc1_ov_n,
    stride_fc1_ov_D,
    stride_dfc11_n,
    stride_dfc11_d,
    stride_dfc11_D,
    stride_dfc12_n,
    stride_dfc12_d,
    stride_dfc12_D,
    stride_do_t,
    stride_do_dD,
    d: tl.constexpr,
    D: tl.constexpr,
    dD: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_d: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_dD: tl.constexpr,
):
    pid_nt = tl.program_id(0)
    pid_n = pid_nt % num_experts
    pid_t = pid_nt // num_experts
    pid_D = tl.program_id(1)
    pid_dD = tl.program_id(2)
    n_start = tl.load(cu_seq_lens + pid_n)
    n_end = tl.load(cu_seq_lens + pid_n + 1)
    n_len = n_end - n_start

    if pid_t * BLOCK_SIZE_T > n_len:
        return

    t_id_ptrs = tl.make_block_ptr(
        base=n_tokens + n_start,
        shape=(n_len,),
        strides=(1,),
        offsets=(pid_t * BLOCK_SIZE_T,),
        block_shape=(BLOCK_SIZE_T,),
        order=(0,),
    )
    s_ptrs = tl.make_block_ptr(
        base=n_score + n_start,
        shape=(n_len,),
        strides=(1,),
        offsets=(pid_t * BLOCK_SIZE_T,),
        block_shape=(BLOCK_SIZE_T,),
        order=(0,),
    )

    fc11_ptrs = tl.make_block_ptr(
        base=fc11 + pid_n * stride_fc11_n,
        shape=(d, D),
        strides=(stride_fc11_d, stride_fc11_D),
        offsets=(0, pid_D * BLOCK_SIZE_D),
        block_shape=(BLOCK_SIZE_d, BLOCK_SIZE_D),
        order=(1, 0),
    )
    fc12_ptrs = tl.make_block_ptr(
        base=fc12 + pid_n * stride_fc12_n,
        shape=(d, D),
        strides=(stride_fc12_d, stride_fc12_D),
        offsets=(0, pid_D * BLOCK_SIZE_D),
        block_shape=(BLOCK_SIZE_d, BLOCK_SIZE_D),
        order=(1, 0),
    )
    fc2_ptrs = tl.make_block_ptr(
        base=fc2 + pid_n * stride_fc2_n,
        shape=(dD, D),
        strides=(stride_fc2_dD, stride_fc2_D),
        offsets=(pid_dD * BLOCK_SIZE_dD, pid_D * BLOCK_SIZE_D),
        block_shape=(BLOCK_SIZE_dD, BLOCK_SIZE_D),
        order=(0, 1),
    )

    t_id = tl.load(t_id_ptrs, boundary_check=(0,), padding_option="zero")
    b_s = tl.load(s_ptrs, boundary_check=(0,), padding_option="zero")
    x_ptrs_t = x + t_id * stride_xt

    do_ptrs = (
        do
        + (t_id * stride_do_t)[:, None]
        + (
            pid_dD * BLOCK_SIZE_dD * stride_do_dD
            + tl.arange(0, BLOCK_SIZE_dD) * stride_do_dD
        )[None, :]
    )
    do_mask = ((pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)) < n_len)[:, None] & (
        (pid_dD * BLOCK_SIZE_dD + tl.arange(0, BLOCK_SIZE_dD)) < dD
    )[None, :]

    fc1_o_g_ptrs = (
        fc1_o_g
        + pid_n * stride_fc1_og_n
        + (t_id * stride_fc1_og_t)[:, None]
        + (
            pid_D * BLOCK_SIZE_D * stride_fc1_og_D
            + tl.arange(0, BLOCK_SIZE_D) * stride_fc1_og_D
        )[None, :]
    )
    fc1_o_g_mask = ((pid_D * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)) < D)[
        None, :
    ] & ((pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)) < n_len)[:, None]

    fc1_o_v_ptrs = (
        fc1_o_v
        + pid_n * stride_fc1_ov_n
        + (t_id * stride_fc1_ov_t)[:, None]
        + (
            pid_D * BLOCK_SIZE_D * stride_fc1_ov_D
            + tl.arange(0, BLOCK_SIZE_D) * stride_fc1_ov_D
        )[None, :]
    )
    fc1_o_v_mask = ((pid_D * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)) < D)[
        None, :
    ] & ((pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)) < n_len)[:, None]

    b_fc1_og = tl.load(fc1_o_g_ptrs, mask=fc1_o_g_mask, other=0)
    b_fc1_ov = tl.load(fc1_o_v_ptrs, mask=fc1_o_v_mask, other=0)
    b_fc1_og_v = b_fc1_og * tl.sigmoid(b_fc1_og)
    # [BLOCK_SIZE_T, BLOCK_SIZE_D]

    b_do = tl.load(do_ptrs, mask=do_mask, other=0)
    b_do = b_do * b_s[:, None]

    # --- dfc1 ---
    b_fc2 = tl.load(fc2_ptrs, boundary_check=(0, 1), padding_option="zero")
    # [BLOCK_SIZE_dD, BLOCK_SIZE_D]

    b_dfc1_o = tl.dot(b_do.to(b_fc2.dtype), b_fc2)
    # [BLOCK_SIZE_T, BLOCK_SIZE_D]

    b_dfc11_o = (
        b_dfc1_o
        * b_fc1_ov
        * tl.sigmoid(b_fc1_og)
        * (1 + b_fc1_og * (1 - tl.sigmoid(b_fc1_og)))
    )
    b_dfc12_o = b_dfc1_o * b_fc1_og_v

    dfc11_ptrs_base = (
        dfc11
        + pid_n * stride_dfc11_n
        + (pid_D * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)) * stride_dfc11_D
    )
    dfc11_ptrs_mask_D = (pid_D * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)) < D

    dfc12_ptrs_base = (
        dfc12
        + pid_n * stride_dfc12_n
        + (pid_D * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)) * stride_dfc12_D
    )
    dfc12_ptrs_mask_D = (pid_D * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)) < D

    for i in range(0, d, BLOCK_SIZE_d):
        x_ptrs = (
            x_ptrs_t[None, :] + ((i + tl.arange(0, BLOCK_SIZE_d)) * stride_xd)[:, None]
        )
        x_mask = ((pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)) < n_len)[
            None, :
        ] & ((i + tl.arange(0, BLOCK_SIZE_d)) < d)[:, None]
        b_x = tl.load(x_ptrs, mask=x_mask, other=0)
        # b_x : [BLOCK_SIZE_d, BLOCK_SIEZ_T]

        b_dfc11 = tl.dot(b_x, b_dfc11_o.to(b_x.dtype))
        # [BLOCK_SIZE_d, BLOCK_SIZE_D]

        b_dfc12 = tl.dot(b_x, b_dfc12_o.to(b_x.dtype))
        # [BLOCK_SIZE_d, BLOCK_SIZE_D]

        dfc11_ptrs = (i + tl.arange(0, BLOCK_SIZE_d))[
            :, None
        ] * stride_dfc11_d + dfc11_ptrs_base[None, :]

        dfc11_ptrs_mask = ((i + tl.arange(0, BLOCK_SIZE_d)) < d)[
            :, None
        ] & dfc11_ptrs_mask_D[None, :]

        dfc12_ptrs = (i + tl.arange(0, BLOCK_SIZE_d))[
            :, None
        ] * stride_dfc12_d + dfc12_ptrs_base[None, :]
        dfc12_ptrs_mask = ((i + tl.arange(0, BLOCK_SIZE_d)) < d)[
            :, None
        ] & dfc12_ptrs_mask_D[None, :]

        tl.atomic_add(
            dfc11_ptrs, b_dfc11.to(dfc11_ptrs.dtype.element_ty), mask=dfc11_ptrs_mask
        )
        tl.atomic_add(
            dfc12_ptrs, b_dfc12.to(dfc12_ptrs.dtype.element_ty), mask=dfc12_ptrs_mask
        )


class SwiGLUMoETritonFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        fc11_weight: torch.Tensor,
        fc12_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
        moe_inner_dim: int,
        num_experts: int,
        top_k: int,
    ) -> torch.Tensor:
        """
        Forward pass for SwiGLUMoE using Triton.

        Args:
            ctx: Context object to save tensors for backward pass.
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            gate_weight (torch.Tensor): Gating network weight of shape (num_experts, hidden_dim).
            fc11_weight (torch.Tensor): Expert fc11 weights of shape (num_experts, moe_inner_dim, hidden_dim).
            fc12_weight (torch.Tensor): Expert fc12 weights of shape (num_experts, moe_inner_dim, hidden_dim).
            fc2_weight (torch.Tensor): Expert fc2 weights of shape (num_experts, hidden_dim, moe_inner_dim).
            moe_inner_dim (int): inner dim of each expert
            num_experts (int): Number of experts.
            top_k (int): Number of experts to select per token.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Reshape x to (batch_size * seq_len, hidden_dim) for processing
        x_flat = x.view(-1, hidden_dim)
        T = batch_size * seq_len
        D = hidden_dim

        # Compute gating scores: x @ gate_weight.T
        gate_scores = F.linear(x_flat, gate_weight)  # [T, num_experts]
        gate_scores = F.softmax(gate_scores, dim=-1)  # [T, num_experts]

        # Select top-k experts
        top_k_scores, top_k_indices = gate_scores.topk(top_k, dim=-1)  # [T, top_k]
        top_k_scores = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # get the total tokens for each moe expert
        cu_seq_lens, expert_to_tokens_flat, expert_to_scores_flat = (
            get_flat_token_and_score(
                top_k_indices, top_k_scores, num_experts, top_k, x.device
            )
        )

        # Allocate output tensor
        output = torch.zeros_like(x_flat, dtype=torch.float32)

        max_seq_len = (cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item()

        def grid(meta):
            BLOCK_SIZE_T = meta["BLOCK_SIZE_T"]
            BLOCK_SIZE_D = meta["BLOCK_SIZE_D"]
            BLOCK_SIZE_dD = meta["BLOCK_SIZE_dD"]
            return (
                num_experts * triton.cdiv(max_seq_len, BLOCK_SIZE_T),
                triton.cdiv(moe_inner_dim, BLOCK_SIZE_D),
                triton.cdiv(hidden_dim, BLOCK_SIZE_dD),
            )

        fused_fwd_kernel[grid](
            x_flat,
            fc11_weight,
            fc12_weight,
            fc2_weight,
            output,
            cu_seq_lens,
            expert_to_tokens_flat,
            expert_to_scores_flat,
            x_flat.stride(0),
            x_flat.stride(1),
            fc11_weight.stride(0),
            fc11_weight.stride(1),
            fc11_weight.stride(2),
            fc12_weight.stride(0),
            fc12_weight.stride(1),
            fc12_weight.stride(2),
            fc2_weight.stride(0),
            fc2_weight.stride(1),
            fc2_weight.stride(2),
            output.stride(0),
            output.stride(1),
            hidden_dim,
            moe_inner_dim,
            hidden_dim,
            num_experts,
        )

        # Reshape output back to original shape
        output = output.view(batch_size, seq_len, hidden_dim).to(x.dtype)

        # Save tensors needed for backward
        ctx.save_for_backward(
            x,
            fc11_weight,
            fc12_weight,
            fc2_weight,
            gate_weight,
            cu_seq_lens,
            expert_to_tokens_flat,
            expert_to_scores_flat,
        )

        ctx.hidden_dim = hidden_dim
        ctx.moe_inner_dim = moe_inner_dim
        ctx.num_experts = num_experts

        return output

    @staticmethod
    def backward(ctx, do: torch.Tensor) -> tuple:
        """
        Backward pass for SwiGLUMoE using Triton.

        Args:
            ctx: Context object containing saved tensors and parameters.
            grad_output (torch.Tensor): Gradient of the loss w.r.t. output.

        Returns:
            tuple: Gradients w.r.t. inputs (x, gate_weight, fc11_weight,
                   fc12_weight, fc2_weight, num_experts, top_k).
        """
        (
            x,
            fc11_weight,
            fc12_weight,
            fc2_weight,
            gate_weight,
            cu_seq_lens,
            expert_to_tokens_flat,
            expert_to_scores_flat,
        ) = ctx.saved_tensors

        hidden_dim = ctx.hidden_dim
        moe_inner_dim = ctx.moe_inner_dim
        num_experts = ctx.num_experts

        batch_size, seq_len, hidden_dim = do.shape
        T = batch_size * seq_len
        fc1_o_g = torch.zeros(
            T, num_experts, moe_inner_dim, dtype=torch.float32, device=x.device
        )
        fc1_o_v = torch.zeros(
            T, num_experts, moe_inner_dim, dtype=torch.float32, device=x.device
        )

        # Reshape tensors
        do_flat = do.view(-1, hidden_dim)
        x_flat = x.view(-1, hidden_dim)

        # Allocate gradient tensors
        dx = torch.zeros_like(x_flat)
        dg = torch.zeros_like(gate_weight)
        d_fc11 = torch.zeros_like(fc11_weight, dtype=torch.float32)
        d_fc12 = torch.zeros_like(fc12_weight, dtype=torch.float32)
        d_fc2 = torch.zeros_like(fc2_weight, dtype=torch.float32)

        def grid(meta):
            BLOCK_SIZE_T = meta["BLOCK_SIZE_T"]
            BLOCK_SIZE_D = meta["BLOCK_SIZE_D"]
            BLOCK_SIZE_dD = meta["BLOCK_SIZE_dD"]
            return (
                num_experts * triton.cdiv(max_seq_len, BLOCK_SIZE_T),
                triton.cdiv(moe_inner_dim, BLOCK_SIZE_D),
                triton.cdiv(hidden_dim, BLOCK_SIZE_dD),
            )

        max_seq_len = (cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item()
        fc2_bwd_kernel[grid](
            x_flat,
            fc11_weight,
            fc12_weight,
            fc1_o_g,
            fc1_o_v,
            cu_seq_lens,
            expert_to_tokens_flat,
            expert_to_scores_flat,
            dx,
            dg,
            d_fc2,
            do,
            x_flat.stride(0),
            x_flat.stride(1),
            fc11_weight.stride(0),
            fc11_weight.stride(1),
            fc11_weight.stride(2),
            fc12_weight.stride(0),
            fc12_weight.stride(1),
            fc12_weight.stride(2),
            fc1_o_g.stride(0),
            fc1_o_g.stride(1),
            fc1_o_g.stride(2),
            fc1_o_v.stride(0),
            fc1_o_v.stride(1),
            fc1_o_v.stride(2),
            d_fc2.stride(0),
            d_fc2.stride(1),
            d_fc2.stride(2),
            do.stride(0),
            do.stride(1),
            hidden_dim,
            moe_inner_dim,
            hidden_dim,
            num_experts,
        )

        fc1_bwd_kernel[grid](
            x_flat,
            fc11_weight,
            fc12_weight,
            fc2_weight,
            fc1_o_g,
            fc1_o_v,
            cu_seq_lens,
            expert_to_tokens_flat,
            expert_to_scores_flat,
            dx,
            dg,
            d_fc11,
            d_fc12,
            do,
            x_flat.stride(0),
            x_flat.stride(1),
            fc11_weight.stride(0),
            fc11_weight.stride(1),
            fc11_weight.stride(2),
            fc12_weight.stride(0),
            fc12_weight.stride(1),
            fc12_weight.stride(2),
            fc2_weight.stride(0),
            fc2_weight.stride(1),
            fc2_weight.stride(2),
            fc1_o_g.stride(0),
            fc1_o_g.stride(1),
            fc1_o_g.stride(2),
            fc1_o_v.stride(0),
            fc1_o_v.stride(1),
            fc1_o_v.stride(2),
            d_fc11.stride(0),
            d_fc11.stride(1),
            d_fc11.stride(2),
            d_fc12.stride(0),
            d_fc12.stride(1),
            d_fc12.stride(2),
            do.stride(0),
            do.stride(1),
            hidden_dim,
            moe_inner_dim,
            hidden_dim,
            num_experts,
        )

        # Reshape grad_x back to original shape
        dx = dx.view(batch_size, seq_len, hidden_dim)

        return (dx, dg, d_fc11, d_fc12, d_fc2, None, None, None)


class SwiGLUMoETriton(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        moe_inner_dim: int,
        num_experts: int,
        top_k: int = 2,
        dtype=torch.bfloat16,
    ):
        """
        Initialize SwiGLUMoETriton module using Triton kernels.

        Args:
            hidden_dim (int): Input hidden dimension.
            moe_inner_dim (int): Inner dimension for expert MLPs.
            num_experts (int): Number of expert networks.
            top_k (int): Number of experts to select per token. Default: 2.
        Note: SwiGLU computes fc2(swish(fc11(x)) * fc12(x)) for each expert.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.moe_inner_dim = moe_inner_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Gating network
        self.moe_gate = nn.Linear(hidden_dim, num_experts, bias=False, dtype=dtype)

        # Expert weights: fc11 and fc12 produce gate and value (hidden_dim -> moe_inner_dim)
        self.fc11_weight = nn.Parameter(
            torch.randn(num_experts, hidden_dim, moe_inner_dim, dtype=dtype)
        )
        self.fc12_weight = nn.Parameter(
            torch.randn(num_experts, hidden_dim, moe_inner_dim, dtype=dtype)
        )
        # fc2 maps back to hidden_dim
        self.fc2_weight = nn.Parameter(
            torch.randn(num_experts, moe_inner_dim, hidden_dim, dtype=dtype)
        )

        # # Initialize weights
        # nn.init.xavier_uniform_(self.fc11_weight)
        # nn.init.xavier_uniform_(self.fc12_weight)
        # nn.init.xavier_uniform_(self.fc2_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SwiGLUMoETriton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        return SwiGLUMoETritonFunction.apply(
            x,
            self.moe_gate.weight,
            self.fc11_weight,
            self.fc12_weight,
            self.fc2_weight,
            self.moe_inner_dim,
            self.num_experts,
            self.top_k,
        )


def run_moe():
    # Set parameters
    batch_size = 1
    seq_len = 128
    hidden_dim = 128
    moe_inner_dim = 256
    num_experts = 8
    top_k = 2

    # Create random input tensor
    x = torch.randn(
        batch_size, seq_len, hidden_dim
    ).cuda()  # Move to GPU if using Triton

    # Instantiate the model
    model = SwiGLUMoETriton(
        hidden_dim=hidden_dim,
        moe_inner_dim=moe_inner_dim,
        num_experts=num_experts,
        top_k=top_k,
    ).cuda()  # Move to GPU

    # Set model to evaluation mode (optional, depending on use case)
    model.eval()

    # Forward pass
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(x)

    # Check output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


def test_swiglu_moe_equivalence():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Model parameters
    batch_size = 1
    seq_len = 128
    hidden_dim = 256
    moe_inner_dim = 512
    num_experts = 128
    top_k = 8
    dtype = torch.bfloat16
    # dtype = torch.float32
    device = "cuda"
    # Initialize models
    torch_model = SwiGLUMoETorch(
        hidden_dim, moe_inner_dim, num_experts, top_k, dtype
    ).cuda()
    triton_model = SwiGLUMoETriton(
        hidden_dim, moe_inner_dim, num_experts, top_k, dtype
    ).cuda()

    # Copy weights from torch_model to triton_model
    with torch.no_grad():
        # Copy gating weights
        triton_model.moe_gate.weight.data.copy_(torch_model.moe_gate.weight.data)

        # Copy expert weights
        for i in range(num_experts):
            # fc11 weights and biases
            triton_model.fc11_weight[i].data.copy_(
                torch_model.experts[i].fc11.weight.data.t()
            )
            # fc12 weights and biases
            triton_model.fc12_weight[i].data.copy_(
                torch_model.experts[i].fc12.weight.data.t()
            )
            # fc2 weights and biases
            triton_model.fc2_weight[i].data.copy_(
                torch_model.experts[i].fc2.weight.data.t()
            )

    # Create input tensor
    x_torch = torch.randn(
        batch_size, seq_len, hidden_dim, dtype=dtype, device=device, requires_grad=True
    )
    x_triton = x_torch.detach().clone().requires_grad_(True)

    # Forward pass
    # torch_gate, torch_scores, torch_indices, torch_output = torch_model(x_torch)
    # triton_gate, triton_scores, triton_indices, triton_output = triton_model(x_triton)
    torch_output = torch_model(x_torch)
    triton_output = triton_model(x_triton)

    # Compare outputs
    torch.testing.assert_close(triton_output, torch_output, rtol=1e-2, atol=1e-2)

    # Backward pass
    torch_output.sum().backward()
    triton_output.sum().backward()
    # Expert weights
    triton_fc11_grad = triton_model.fc11_weight.grad
    triton_fc12_grad = triton_model.fc12_weight.grad
    triton_fc2_grad = triton_model.fc2_weight.grad
    for i in range(num_experts):
        # # fc11 weights
        torch.testing.assert_close(
            triton_fc11_grad[i],
            torch_model.experts[i].fc11.weight.grad.t(),
            rtol=1e-2,
            atol=1e-2,
        )
        # fc12 weights
        torch.testing.assert_close(
            triton_fc12_grad[i],
            torch_model.experts[i].fc12.weight.grad.t(),
            rtol=1e-2,
            atol=1e-2,
        )
        # fc2 weights
        torch.testing.assert_close(
            triton_fc2_grad[i],
            torch_model.experts[i].fc2.weight.grad.t(),
            rtol=1e-2,
            atol=1e-2,
        )


if __name__ == "__main__":
    test_swiglu_moe_equivalence()
