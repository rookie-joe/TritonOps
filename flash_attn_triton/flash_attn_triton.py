from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func
import pdb

from utils.utils import get_num_warps_stages, is_hopper_gpu


IS_HOPPER_GPU = is_hopper_gpu()


# Define configurations for num_warps and num_stages.
# BLOCK_SIZEs are determined by the caller and are part of the autotuner key.
autotune_configs = []
_stages = (
    [2, 3, 4, 5] if IS_HOPPER_GPU else [2, 3, 4]
)  # Extended stages a bit for non-Hopper too
_warps = [2, 4]  # Common values for num_warps. Can add 2 or 16 if needed.

for warps_val in _warps:
    for stages_val in _stages:
        autotune_configs.append(
            triton.Config(
                {"num_warps": warps_val, "num_stages": stages_val},
                num_warps=warps_val,
                num_stages=stages_val,
            )
        )
if (
    not autotune_configs
):  # Fallback if IS_HOPPER_GPU logic somehow fails or lists are empty
    autotune_configs.append(
        triton.Config({"num_warps": 4, "num_stages": 2}, num_warps=4, num_stages=2)
    )


@triton.autotune(
    configs=autotune_configs,
    key=[
        "BLOCK_SIZE_QD",
        "HEAD_DIM_Q",
        "HEAD_DIM_V",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_KN",
        "BLOCK_SIZE_D",
    ],
)
@triton.jit
def flash_attn_fwd_kernel(
    q,
    k,
    v,
    o,
    cu_seq_lens,
    lse,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_on,
    stride_oh,
    stride_od,
    stride_lse_n,
    stride_lse_h,
    BLOCK_SIZE_QD: tl.constexpr,
    HEAD_DIM_Q: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KN: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    sm_scale: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = pid_bh % NUM_HEADS
    pid_kh = pid_h // GQA_RATIO
    pid_b = pid_bh // NUM_HEADS
    seq_start = tl.load(cu_seq_lens + pid_b)
    seq_end = tl.load(cu_seq_lens + pid_b + 1)
    seq_len = seq_end - seq_start
    if pid_n * BLOCK_SIZE_N >= seq_len:
        return

    q_ptrs = tl.make_block_ptr(
        base=q + seq_start * stride_qn + pid_h * stride_qh,
        strides=(stride_qn, stride_qd),
        shape=(seq_len, HEAD_DIM_Q),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_QD),
        order=(1, 0),
    )

    # note that k/v need to loop in a kernel
    k_ptrs = tl.make_block_ptr(
        base=k + seq_start * stride_kn + pid_kh * stride_kh,
        strides=(stride_kd, stride_kn),
        shape=(HEAD_DIM_Q, seq_len),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_QD, BLOCK_SIZE_KN),
        order=(0, 1),
    )

    v_ptrs = tl.make_block_ptr(
        base=v + seq_start * stride_vn + pid_kh * stride_vh,
        strides=(stride_vn, stride_vd),
        shape=(seq_len, HEAD_DIM_V),
        offsets=(0, pid_d * BLOCK_SIZE_D),
        block_shape=(BLOCK_SIZE_KN, BLOCK_SIZE_D),
        order=(1, 0),
    )

    o_ptrs = tl.make_block_ptr(
        base=o + seq_start * stride_on + pid_h * stride_oh,
        strides=(stride_on, stride_od),
        shape=(seq_len, HEAD_DIM_V),
        offsets=(pid_n * BLOCK_SIZE_N, pid_d * BLOCK_SIZE_D),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = tl.make_block_ptr(
        base=lse + seq_start * stride_lse_n + pid_h * stride_lse_h,
        strides=(stride_lse_n,),
        shape=(seq_len,),
        offsets=(pid_n * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    # acc_o = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_D), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_D), 0, dtype=tl.float32)
    # lse_i = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_N,), float("-inf"), dtype=tl.float32)
    m_i = tl.full((BLOCK_SIZE_N,), float("-inf"), dtype=tl.float32)

    for i in range(0, pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N, BLOCK_SIZE_KN):
        q_id = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        k_id = i + tl.arange(0, BLOCK_SIZE_KN)
        b_qk_mask = (
            (q_id[:, None] >= k_id[None, :])
            & (q_id < seq_len)[:, None]
            & (k_id < seq_len)[None, :]
        )

        qk = tl.where(b_qk_mask, 0, float("-inf"))
        b_q = tl.load(q_ptrs, padding_option="zero", boundary_check=(0, 1))
        # b_q: [BLOCK_SIZE_N, BLOCK_SIZE_QD]
        b_k = tl.load(k_ptrs, padding_option="zero", boundary_check=(0, 1))
        # b_k: [BLOCK_SIZE_QD, BLOCK_SIZE_N]
        qk += tl.dot(b_q, b_k) * sm_scale

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))

        qk_exp = tl.exp2(qk - m_ij[:, None])

        b_v = tl.load(v_ptrs, padding_option="zero", boundary_check=(0, 1))
        # b_v: [BLOCK_SIZE_N, BLOCK_SIZE_D]

        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None] + tl.dot(qk_exp.to(b_v.dtype), b_v)

        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + tl.sum(qk_exp, axis=1))

        m_i = m_ij

        k_ptrs = tl.advance(k_ptrs, offsets=(0, BLOCK_SIZE_KN))
        v_ptrs = tl.advance(v_ptrs, offsets=(BLOCK_SIZE_KN, 0))

    acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]

    tl.store(o_ptrs, acc_o.to(o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(lse_ptrs, lse_i.to(lse.dtype.element_ty), boundary_check=(0,))


@triton.jit
def flash_attn_bwd_D_kernel(
    do,
    o,
    D,
    cu_seq_lens,
    stride_do_n,
    stride_do_h,
    stride_do_d,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_D_n,
    stride_D_h,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    seq_start = tl.load(cu_seq_lens + pid_b)
    seq_end = tl.load(cu_seq_lens + pid_b + 1)
    seq_len = seq_end - seq_start
    if pid_n * BLOCK_SIZE_N >= seq_len:
        return

    do_ptrs = tl.make_block_ptr(
        base=do + seq_start * stride_do_n + pid_h * stride_do_h,
        strides=(stride_do_n, stride_do_d),
        shape=(seq_len, HEAD_DIM),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1, 0),
    )
    o_ptrs = tl.make_block_ptr(
        base=o + seq_start * stride_o_n + pid_h * stride_o_h,
        strides=(stride_o_n, stride_o_d),
        shape=(seq_len, HEAD_DIM),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1, 0),
    )
    D_ptrs = tl.make_block_ptr(
        base=D + seq_start * stride_D_n + pid_h * stride_D_h,
        strides=(stride_D_n,),
        shape=(seq_len,),
        offsets=(pid_n * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )

    b_o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    b_do = tl.load(do_ptrs, boundary_check=(0, 1), padding_option="zero")

    b_D = tl.sum(b_do * b_o, axis=1)
    b_D = tl.reshape(b_D, (BLOCK_SIZE_N,))

    tl.store(D_ptrs, b_D.to(D_ptrs.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=autotune_configs,
    key=[
        "HEAD_DIM",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_KN",
        "BLOCK_SIZE_D",
    ],
)
@triton.jit
def flash_attn_bwd_dq_kernel(
    q,
    k,
    v,
    D,
    lse,
    dq,
    do,
    cu_seq_lens,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_D_n,
    stride_D_h,
    stride_lse_n,
    stride_lse_h,
    stride_dq_n,
    stride_dq_h,
    stride_dq_d,
    stride_do_n,
    stride_do_h,
    stride_do_d,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KN: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    sm_scale: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kh = pid_h // GQA_RATIO
    pid_n = tl.program_id(2)
    seq_start = tl.load(cu_seq_lens + pid_b)
    seq_end = tl.load(cu_seq_lens + pid_b + 1)
    seq_len = seq_end - seq_start
    if pid_n * BLOCK_SIZE_N >= seq_len:
        return
    q_ptrs = tl.make_block_ptr(
        base=q + seq_start * stride_q_n + pid_h * stride_q_h,
        strides=(stride_q_n, stride_q_d),
        shape=(seq_len, HEAD_DIM),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1, 0),
    )

    k_ptrs = tl.make_block_ptr(
        base=k + seq_start * stride_k_n + pid_kh * stride_k_h,
        strides=(stride_k_n, stride_k_d),
        shape=(seq_len, HEAD_DIM),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KN, BLOCK_SIZE_D),
        order=(1, 0),
    )

    v_ptrs = tl.make_block_ptr(
        base=v + seq_start * stride_v_n + pid_kh * stride_v_h,
        strides=(stride_v_d, stride_v_n),
        shape=(HEAD_DIM, seq_len),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_KN),
        order=(0, 1),
    )
    D_ptrs = tl.make_block_ptr(
        base=D + seq_start * stride_D_n + pid_h * stride_D_h,
        strides=(stride_D_n,),
        shape=(seq_len,),
        offsets=(pid_n * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    lse_ptrs = tl.make_block_ptr(
        base=lse + seq_start * stride_lse_n + pid_h * stride_lse_h,
        strides=(stride_lse_n,),
        shape=(seq_len,),
        offsets=(pid_n * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )

    dq_ptrs = tl.make_block_ptr(
        base=dq + seq_start * stride_dq_n + pid_h * stride_dq_h,
        strides=(stride_dq_n, stride_dq_d),
        shape=(seq_len, HEAD_DIM),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1, 0),
    )

    do_ptrs = tl.make_block_ptr(
        base=do + seq_start * stride_do_n + pid_h * stride_do_h,
        strides=(stride_do_n, stride_do_d),
        shape=(seq_len, HEAD_DIM),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1, 0),
    )

    b_dq = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_D), value=0, dtype=tl.float32)

    q_id = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    for i in range(0, pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N, BLOCK_SIZE_KN):
        k_id = i + tl.arange(0, BLOCK_SIZE_KN)
        b_qk_mask = (
            (q_id[:, None] >= k_id[None, :])
            & (q_id < seq_len)[:, None]
            & (k_id < seq_len)[None, :]
        )

        b_q = tl.load(q_ptrs, padding_option="zero", boundary_check=(0, 1))
        # b_q: [BLOCK_SIZE_N, BLOCK_SIZE_QD]
        b_k = tl.load(k_ptrs, padding_option="zero", boundary_check=(0, 1))
        # b_k [BLOCK_SIZE_KN, BLOCK_SIZE_D]
        b_v = tl.load(v_ptrs, padding_option="zero", boundary_check=(0, 1))
        # b_v [BLOCK_SIZE_D, BLOCK_SIZE_KN]
        b_lse = tl.load(lse_ptrs, padding_option="zero", boundary_check=(0,))

        b_do = tl.load(do_ptrs, padding_option="zero", boundary_check=(0, 1))
        b_dpij = tl.dot(b_do, b_v)
        # [BLOCK_SIZE_N, BLOCK_SIZE_KN]

        qk = tl.where(b_qk_mask, 0, float("-inf"))
        qk += tl.dot(b_q, tl.trans(b_k)) * sm_scale
        b_lse = tl.load(lse_ptrs, padding_option="zero", boundary_check=(0,))
        b_pij = tl.exp2(qk - b_lse[:, None])
        # [BLOCK_SIZE_N, BLOCK_SIZE_KN]

        b_D = tl.load(D_ptrs, padding_option="zero", boundary_check=(0,))
        # [BLOCK_SIZE_N, ]

        b_dsij = (b_dpij * b_pij - b_pij * b_D[:, None]) * sm_scale / 1.4426950408889634
        # [BLOCK_SIZE_N, BLOCK_SIZE_KN]

        b_dq += tl.dot(b_dsij.to(b_k.dtype), b_k)

        k_ptrs = tl.advance(k_ptrs, (BLOCK_SIZE_KN, 0))
        v_ptrs = tl.advance(v_ptrs, (0, BLOCK_SIZE_KN))

    tl.store(dq_ptrs, b_dq.to(dq_ptrs.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=autotune_configs,
    key=[
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_QN",
        "BLOCK_SIZE_D",
        "HEAD_DIM",
    ],
)
@triton.jit
def flash_attn_bwd_dkdv_kernel(
    q,
    k,
    v,
    cu_seq_lens,
    lse,
    D,
    do,
    dk,
    dv,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_lse_n,
    stride_lse_h,
    stride_D_n,
    stride_D_h,
    stride_do_n,
    stride_do_h,
    stride_do_d,
    stride_dk_n,
    stride_dk_h,
    stride_dk_qga,
    stride_dk_d,
    stride_dv_n,
    stride_dv_h,
    stride_dv_qga,
    stride_dv_d,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_QN: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    sm_scale: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)
    pid_kh = pid_h // GQA_RATIO
    pid_gqa = pid_h % GQA_RATIO
    seq_start = tl.load(cu_seq_lens + pid_b)
    seq_end = tl.load(cu_seq_lens + pid_b + 1)
    seq_len = seq_end - seq_start
    if pid_n * BLOCK_SIZE_N >= seq_len:
        return

    q_ptrs = tl.make_block_ptr(
        base=q + seq_start * stride_q_n + pid_h * stride_q_h,
        strides=(stride_q_n, stride_q_d),
        shape=(seq_len, HEAD_DIM),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_QN, BLOCK_SIZE_D),
        order=(1, 0),
    )

    lse_ptrs = tl.make_block_ptr(
        base=lse + seq_start * stride_lse_n + pid_h * stride_lse_h,
        strides=(stride_lse_n,),
        shape=(seq_len,),
        offsets=(pid_n * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_QN,),
        order=(0,),
    )
    D_ptrs = tl.make_block_ptr(
        base=D + seq_start * stride_D_n + pid_h * stride_D_h,
        strides=(stride_D_n,),
        shape=(seq_len,),
        offsets=(pid_n * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_QN,),
        order=(0,),
    )

    k_ptrs = tl.make_block_ptr(
        base=k + seq_start * stride_k_n + pid_kh * stride_k_h,
        strides=(stride_k_n, stride_k_d),
        shape=(seq_len, HEAD_DIM),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1, 0),
    )

    v_ptrs = tl.make_block_ptr(
        base=v + seq_start * stride_v_n + pid_kh * stride_v_h,
        strides=(stride_v_n, stride_v_d),
        shape=(seq_len, HEAD_DIM),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1, 0),
    )

    do_ptrs = tl.make_block_ptr(
        base=do + seq_start * stride_do_n + pid_h * stride_do_h,
        strides=(stride_do_n, stride_do_d),
        shape=(seq_len, HEAD_DIM),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_QN, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dk_ptrs = tl.make_block_ptr(
        base=dk
        + seq_start * stride_dk_n
        + pid_kh * stride_dk_h
        + pid_gqa * stride_dk_qga,
        strides=(stride_dk_n, stride_dk_d),
        shape=(seq_len, HEAD_DIM),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dv_ptrs = tl.make_block_ptr(
        base=dv
        + seq_start * stride_dv_n
        + pid_kh * stride_dv_h
        + pid_gqa * stride_dv_qga,
        strides=(stride_dv_n, stride_dv_d),
        shape=(seq_len, HEAD_DIM),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1, 0),
    )

    b_dk = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_D), value=0, dtype=tl.float32)
    b_dv = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_D), value=0, dtype=tl.float32)

    k_id = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    for i in range(pid_n * BLOCK_SIZE_N, seq_end, BLOCK_SIZE_QN):
        q_id = i + tl.arange(0, BLOCK_SIZE_QN)
        b_kq_mask = (
            (k_id[:, None] <= q_id[None, :])
            & (k_id < seq_len)[:, None]
            & (q_id < seq_len)[None, :]
        )

        b_q = tl.load(q_ptrs, padding_option="zero", boundary_check=(0, 1))
        # b_q: [BLOCK_SIZE_QN, BLOCK_SIZE_D]
        b_k = tl.load(k_ptrs, padding_option="zero", boundary_check=(0, 1))
        # b_k [BLOCK_SIZE_N, BLOCK_SIZE_D]
        b_v = tl.load(v_ptrs, padding_option="zero", boundary_check=(0, 1))
        # b_v [BLOCK_SIZE_N, BLOCK_SIZE_D]

        b_lse = tl.load(lse_ptrs, padding_option="zero", boundary_check=(0,))
        # b_q: [BLOCK_SIZE_QN, ]

        b_do = tl.load(do_ptrs, padding_option="zero", boundary_check=(0, 1))
        # b_do: [BLOCK_SIZE_QN, BLOCK_SIZE_D]

        b_dpij = tl.dot(b_v, tl.trans(b_do))
        # [BLOCK_SIZE_N, BLOCK_SIZE_QN]

        kq = tl.where(b_kq_mask, 0, float("-inf"))
        # [BLOCK_SIZE_N, BLOCK_SIZE_QN]

        kq += tl.dot(b_k, tl.trans(b_q)) * sm_scale
        b_lse = tl.load(lse_ptrs, padding_option="zero", boundary_check=(0,))
        b_pij = tl.exp2(kq - b_lse[None, :])
        # [BLOCK_SIZE_N, BLOCK_SIZE_QN]

        b_D = tl.load(D_ptrs, padding_option="zero", boundary_check=(0,))
        # [BLOCK_SIZE_QN, ]

        b_dsij = (b_dpij * b_pij - b_pij * b_D[None, :]) * sm_scale / 1.4426950408889634
        # [BLOCK_SIZE_N, BLOCK_SIZE_QN]

        b_dk += tl.dot(b_dsij.to(b_k.dtype), b_q)

        b_dv += tl.dot(b_pij.to(b_do.dtype), b_do)

        q_ptrs = tl.advance(q_ptrs, (BLOCK_SIZE_QN, 0))
        lse_ptrs = tl.advance(lse_ptrs, (BLOCK_SIZE_QN,))
        D_ptrs = tl.advance(D_ptrs, (BLOCK_SIZE_QN,))
        do_ptrs = tl.advance(do_ptrs, (BLOCK_SIZE_QN, 0))

    tl.store(dk_ptrs, b_dk.to(dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dv_ptrs, b_dv.to(dv.dtype.element_ty), boundary_check=(0, 1))


def flash_attn_triton_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    sm_scale,
) -> torch.Tensor:

    _, seq_len, num_heads_q, head_dim_q = q.shape
    batch_size = len(cu_seq_lens) - 1
    _, _, num_heads_v, head_dim_v = v.shape

    # Output tensor
    o = torch.zeros(1, seq_len, num_heads_q, head_dim_q, device=q.device, dtype=q.dtype)
    lse = torch.zeros(1, seq_len, num_heads_q, device=q.device, dtype=torch.float32)

    # Constants
    BLOCK_SIZE_QD = triton.next_power_of_2(head_dim_q)
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_KN = 128
    BLOCK_SIZE_D = 128
    GQA_RATIO = num_heads_q // num_heads_v  # Assuming AABB

    # Grid computation
    grid = (
        batch_size * num_heads_q,
        triton.cdiv((cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item(), BLOCK_SIZE_N),
        triton.cdiv(head_dim_v, BLOCK_SIZE_D),
    )

    # Launch kernel
    flash_attn_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        cu_seq_lens,
        lse,
        q.stride(1),
        q.stride(2),
        q.stride(3),  # q strides: batch, heads, dim
        k.stride(1),
        k.stride(2),
        k.stride(3),  # k strides
        v.stride(1),
        v.stride(2),
        v.stride(3),  # v strides
        o.stride(1),
        o.stride(2),
        o.stride(3),  # o strides
        lse.stride(1),
        lse.stride(2),
        BLOCK_SIZE_QD=BLOCK_SIZE_QD,
        HEAD_DIM_Q=head_dim_q,
        HEAD_DIM_V=head_dim_v,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_KN=BLOCK_SIZE_KN,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        NUM_HEADS=num_heads_q,
        GQA_RATIO=GQA_RATIO,
        sm_scale=sm_scale,
    )

    return o, lse


def flash_attn_triton_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    sm_scale: float,
):

    _, seq_len, num_heads_q, head_dim_q = q.shape
    batch_size = len(cu_seq_lens) - 1
    _, _, num_heads_v, head_dim_v = v.shape

    # firstly calculate the do_o for both dq and dk
    D = torch.zeros(1, seq_len, num_heads_q, device=q.device, dtype=q.dtype)

    BLOCK_SIZE_N = 128
    BLOCK_SIZE_KN = 128
    BLOCK_SIZE_QN = 128
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim_v)
    GQA_RATIO = num_heads_q // num_heads_v  # Assuming AABB

    # Output tensor
    dq = torch.zeros_like(q)
    dk = torch.zeros(
        1, seq_len, num_heads_v, GQA_RATIO, head_dim_v, device=q.device, dtype=q.dtype
    )
    dv = torch.zeros(
        1, seq_len, num_heads_v, GQA_RATIO, head_dim_v, device=q.device, dtype=q.dtype
    )

    # Grid computation
    grid = (
        batch_size,
        num_heads_q,
        triton.cdiv((cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item(), BLOCK_SIZE_N),
    )

    flash_attn_bwd_D_kernel[grid](
        do,
        o,
        D,
        cu_seq_lens,
        do.stride(1),
        do.stride(2),
        do.stride(3),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        D.stride(1),
        D.stride(2),
        BLOCK_SIZE_N,
        BLOCK_SIZE_D,
        head_dim_v,
    )
    # do_o = torch.einsum("bthd,bthd->bth", do, o)

    # Grid computation
    grid = (
        batch_size,
        num_heads_q,
        triton.cdiv((cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item(), BLOCK_SIZE_N),
    )
    # now dq
    flash_attn_bwd_dq_kernel[grid](
        q,
        k,
        v,
        D,
        lse,
        dq,
        do,
        cu_seq_lens,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        D.stride(1),
        D.stride(2),
        lse.stride(1),
        lse.stride(2),
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        do.stride(1),
        do.stride(2),
        do.stride(3),
        BLOCK_SIZE_N,
        BLOCK_SIZE_KN,
        BLOCK_SIZE_D,
        head_dim_q,
        GQA_RATIO,
        sm_scale,
    )

    # Grid computation
    grid = (
        batch_size,
        num_heads_q,
        triton.cdiv((cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item(), BLOCK_SIZE_N),
    )

    # now dk & dv
    flash_attn_bwd_dkdv_kernel[grid](
        q,
        k,
        v,
        cu_seq_lens,
        lse,
        D,
        do,
        dk,
        dv,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        lse.stride(1),
        lse.stride(2),
        D.stride(1),
        D.stride(2),
        do.stride(1),
        do.stride(2),
        do.stride(3),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        dk.stride(4),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        dv.stride(4),
        BLOCK_SIZE_N,
        BLOCK_SIZE_QN,
        BLOCK_SIZE_D,
        head_dim_v,
        GQA_RATIO,
        sm_scale,
    )

    return dq, dk.sum(dim=-2), dv.sum(dim=-2)


class flash_attn_triton(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,  # [1, total_seq_len, num_heads, head_dim]
        k: torch.Tensor,  # [1, total_seq_len, num_kv_heads, head_dim]
        v: torch.Tensor,  # [1, total_seq_len, num_kv_heads, head_dim]
        cu_seq_lens: torch.Tensor,  # [batch_size + 1]
    ) -> torch.Tensor:  # [1, total_seq_len, num_heads, head_dim]
        """
        Forward pass for Flash Attention using Triton implementation.

        Args:
            ctx: Context object to save tensors for backward pass.
            q: Query tensor of shape [1, total_total_seq_len, num_heads, head_dim].
            k: Key tensor of shape [1, total_seq_len, num_kv_heads, head_dim].
            v: Value tensor of shape [1, total_seq_len, num_kv_heads, head_dim].
            cu_seq_lens: Cumulative sequence lengths, shape [batch_size + 1].

        Returns:
            Output tensor of shape [1, total_seq_len, num_heads, head_dim].
        """
        sm_scale = (1 / q.shape[-1]) ** (0.5) * 1.4426950408889634
        # Compute forward pass using Triton implementation
        o, lse = flash_attn_triton_fwd(q, k, v, cu_seq_lens, sm_scale)

        # Save tensors needed for backward pass
        ctx.save_for_backward(q, k, v, o, lse, cu_seq_lens)
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(
        ctx,
        do: torch.Tensor,  # [1, total_seq_len, num_heads, head_dim]
    ) -> tuple:  # (dq, dk, dv, None)
        """
        Backward pass for Flash Attention using Triton implementation.

        Args:
            ctx: Context object containing saved tensors from forward pass.
            do: Gradient of the output, shape [1, total_seq_len, num_heads, head_dim].

        Returns:
            Tuple containing:
                - dq: Gradient of query tensor, same shape as q.
                - dk: Gradient of key tensor, same shape as k.
                - dv: Gradient of value tensor, same shape as v.
                - None: Placeholder for cu_seq_lens gradient (not computed).
        """
        # Retrieve saved tensors
        q, k, v, o, lse, cu_seq_lens = ctx.saved_tensors

        dq, dk, dv = flash_attn_triton_bwd(
            q, k, v, o, do, lse, cu_seq_lens, ctx.sm_scale
        )

        return dq, dk, dv, None


def flash_attn_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seq_lens: torch.Tensor,
):
    _, T, H, Dq, Dv = *q.shape, v.shape[-1]
    kH = k.shape[2]
    k = k.repeat_interleave(H // kH, dim=2)
    v = v.repeat_interleave(H // kH, dim=2)
    o = torch.zeros(1, T, H, Dv, device=q.device, dtype=q.dtype)
    B = len(cu_seq_lens) - 1
    sm_scale = (1 / Dq) ** (0.5)
    for i in range(B):
        seq_start = cu_seq_lens[i]
        seq_end = cu_seq_lens[i + 1]
        seq_len = seq_end - seq_start
        b_q = q[0, seq_start:seq_end, ...]
        b_k = k[0, seq_start:seq_end, ...]
        b_v = v[0, seq_start:seq_end, ...]

        qk = torch.einsum("QHD, KHD->HQK", b_q, b_k) * sm_scale
        causal_mask = (
            torch.arange(0, seq_len)[:, None] >= torch.arange(0, seq_len)[None, :]
        ).to(q.device)

        qk = torch.masked_fill(qk, ~causal_mask[None, ...], float("-inf")).to(
            torch.float32
        )
        # qk_max = qk.max(dim=-1, keepdim=True).values

        # softmax_qk = (qk - qk_max).exp() / (qk - qk_max).exp().sum(dim=-1, keepdim=True)
        softmax_qk = torch.softmax(qk, dim=-1, dtype=torch.float32)

        o_i = torch.einsum("HQK, KHD-> QHD", softmax_qk.to(q.dtype), b_v)

        o[0, seq_start:seq_end, ...] = o_i

    return o


def generate_test_data(batch_size, max_seq_len, num_heads, num_kv_heads, head_dim):
    # Generate random sequence lengths for each batch
    seq_lengths = torch.randint(low=1, high=max_seq_len + 1, size=(batch_size,)).cuda()
    # seq_lengths = torch.tensor(64)

    # Compute cumulative sequence lengths
    cu_seq_lens = torch.zeros(batch_size + 1, dtype=torch.int32).cuda()
    cu_seq_lens[1:] = torch.cumsum(seq_lengths, dim=0)
    total_seq_len = cu_seq_lens[-1].item()

    # Generate tensors with total sequence length
    # Generate tensors with values between 0 and 1
    q = torch.rand(1, total_seq_len, num_heads, head_dim, dtype=torch.bfloat16).cuda()
    k = torch.rand(
        1, total_seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16
    ).cuda()
    v = torch.rand(
        1, total_seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16
    ).cuda()

    return q, k, v, cu_seq_lens


def test_flash_attention_fwdonly():

    # Test configurations
    test_configs = [
        (1, 128, 8, 96),  # Small sequence
        (2, 256, 8, 128),  # Medium sequence
        (4, 512, 4, 64),  # Larger sequence
    ]

    torch.manual_seed(42)

    for batch_size, max_seq_len, num_heads, head_dim in test_configs:
        print(
            f"\nTesting: batch_size={batch_size}, max_seq_len={max_seq_len}, "
            f"num_heads={num_heads}, head_dim={head_dim}"
        )

        # Generate test data
        q, k, v, cu_seq_lens = generate_test_data(
            batch_size, max_seq_len, num_heads, head_dim
        )

        # Run both implementations
        with torch.no_grad():
            flash_output = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seq_lens,
                cu_seq_lens,
                max_seq_len,
                max_seq_len,
                causal=True,
            )
            flash_output = flash_output.unsqueeze(0)
            pytorch_output = flash_attn_torch(q, k, v, cu_seq_lens)
            triton_output = flash_attn_triton(q, k, v, cu_seq_lens)

        # Compare results
        absolute_diff = torch.abs(pytorch_output - triton_output)
        relative_diff = absolute_diff / (torch.abs(pytorch_output) + 1e-6)

        max_abs_diff = absolute_diff.max().item()
        mean_abs_diff = absolute_diff.mean().item()
        max_rel_diff = relative_diff.max().item()

        print(f"Max absolute difference: {max_abs_diff:.6f}")
        print(f"Mean absolute difference: {mean_abs_diff:.6f}")
        print(f"Max relative difference: {max_rel_diff:.6f}")
        print(f"cu_seq_lens: {cu_seq_lens}")

        # Assert that results are close
        # assert torch.allclose( pytorch_output, triton_output, rtol=1e-2, atol=1e-2), f"Test failed for config: {batch_size}, {max_seq_len}, {num_heads}, {head_dim}"
        # assert torch.allclose( pytorch_output, flash_output, rtol=1e-2, atol=1e-2)
        # assert torch.allclose( pytorch_output, triton_output, rtol=1e-2, atol=1e-2)
        assert torch.allclose(flash_output, triton_output, rtol=1e-2, atol=1e-2)

        print("Test passed!")


def test_flash_attention():
    import torch

    # Test configurations
    test_configs = [
        (3, 128, 2, 2, 96),  # Small sequence
        (2, 256, 8, 4, 128),  # Medium sequence
        (4, 512, 4, 1, 64),  # Larger sequence
    ]

    torch.manual_seed(42)

    for batch_size, max_seq_len, num_heads, num_kv_heads, head_dim in test_configs:
        print(
            f"\nTesting: batch_size={batch_size}, max_seq_len={max_seq_len}, "
            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}"
        )

        # Generate test data with requires_grad=True for gradient checking
        q, k, v, cu_seq_lens = generate_test_data(
            batch_size, max_seq_len, num_heads, num_kv_heads, head_dim
        )
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        # Clone inputs for both implementations
        q_torch, k_torch, v_torch = (
            q.clone().detach().requires_grad_(True),
            k.clone().detach().requires_grad_(True),
            v.clone().detach().requires_grad_(True),
        )
        q_triton, k_triton, v_triton = (
            q.clone().detach().requires_grad_(True),
            k.clone().detach().requires_grad_(True),
            v.clone().detach().requires_grad_(True),
        )

        # Forward pass
        pytorch_output = flash_attn_torch(q_torch, k_torch, v_torch, cu_seq_lens)
        triton_output = flash_attn_triton.apply(
            q_triton, k_triton, v_triton, cu_seq_lens
        )
        # Check forward pass
        # abs_diff_forward = torch.abs(pytorch_output - triton_output)
        # rel_diff_forward = abs_diff_forward / (torch.abs(pytorch_output) + 1e-6)

        # print("\nForward Pass Comparison:")
        # print(f"Max absolute difference: {abs_diff_forward.max().item():.6f}")
        # print(f"Mean absolute difference: {abs_diff_forward.mean().item():.6f}")
        # print(f"Max relative difference: {rel_diff_forward.max().item():.6f}")

        # Backward pass test
        randn = torch.randn_like(pytorch_output)
        loss = (pytorch_output * randn).sum()
        loss.backward()

        # PyTorch backward
        dq_torch = q_torch.grad.clone()
        dk_torch = k_torch.grad.clone()
        dv_torch = v_torch.grad.clone()

        # Triton backward
        randn2 = randn.clone().detach()
        loss2 = (triton_output * randn2).sum()
        loss2.backward()
        dq_triton = q_triton.grad.clone()
        dk_triton = k_triton.grad.clone()
        dv_triton = v_triton.grad.clone()

        # # Check backward pass
        # for name, torch_grad, triton_grad in [
        #     ("dq", dq_torch, dq_triton),
        #     ("dk", dk_torch, dk_triton),
        #     ("dv", dv_torch, dv_triton),
        # ]:
        #     abs_diff = torch.abs(torch_grad - triton_grad)
        #     rel_diff = abs_diff / (torch.abs(torch_grad) + 1e-6)

        #     print(f"\nGradient Comparison for {name}:")
        #     print(f"Max absolute difference: {abs_diff.max().item():.6f}")
        #     print(f"Mean absolute difference: {abs_diff.mean().item():.6f}")
        #     print(f"Max relative difference: {rel_diff.max().item():.6f}")

        # Assertions
        torch.testing.assert_close(pytorch_output, triton_output, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(dq_torch, dq_triton, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(dk_torch, dk_triton, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(dv_torch, dv_triton, rtol=1e-1, atol=1e-1)

        print("Test passed successfully!")


if __name__ == "__main__":
    # Make sure we're running on GPU
    assert torch.cuda.is_available(), "CUDA is required for this test"
    torch.cuda.set_device(0)

    # Run the tests
    test_flash_attention()
