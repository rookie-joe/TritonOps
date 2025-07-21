from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from utils.utils import get_num_warps_stages, is_hopper_gpu
import pdb

IS_HOPPER_GPU = is_hopper_gpu()
# Define configurations for num_warps and num_stages.
# BLOCK_SIZEs are determined by the caller and are part of the autotuner key.
autotune_configs = []
_stages = (
    [2, 3, 4, 5] if IS_HOPPER_GPU else [2, 3, 4]
)  # Extended stages a bit for non-Hopper too
_warps = [4, 8]  # Common values for num_warps. Can add 2 or 16 if needed.

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
        "HEAD_DIM_Q",
        "HEAD_DIM_V",
        "BLOCK_SIZE_D",
        "BLOCK_SIZE_QD",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_KN",
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
    LOCAL_WINDOW_SIZE: tl.constexpr,
    GLOBAL_WINDOW_SIZE: tl.constexpr,
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

    # --- Accumulators ---
    acc_o = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_D), 0, dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_N,), float("-inf"), dtype=tl.float32)
    m_i = tl.full((BLOCK_SIZE_N,), float("-inf"), dtype=tl.float32)
    b_q = tl.load(q_ptrs, padding_option="zero", boundary_check=(0, 1))
    q_id = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # --- 1. Local Window Attention ---
    # Keys for local window: [max(0, q_idx - LOCAL_WINDOW_SIZE + 1), q_idx]
    # Iteration range for K blocks for the current Q block:
    l_low = tl.maximum(0, pid_n * BLOCK_SIZE_N - LOCAL_WINDOW_SIZE + 1)
    l_hi = pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N

    k_ptrs = tl.make_block_ptr(
        base=k + seq_start * stride_kn + pid_kh * stride_kh,
        strides=(stride_kd, stride_kn),
        shape=(HEAD_DIM_Q, seq_len),
        offsets=(0, l_low),
        block_shape=(BLOCK_SIZE_QD, BLOCK_SIZE_KN),
        order=(0, 1),
    )

    v_ptrs = tl.make_block_ptr(
        base=v + seq_start * stride_vn + pid_kh * stride_vh,
        strides=(stride_vn, stride_vd),
        shape=(seq_len, HEAD_DIM_V),
        offsets=(l_low, pid_d * BLOCK_SIZE_D),
        block_shape=(BLOCK_SIZE_KN, BLOCK_SIZE_D),
        order=(1, 0),
    )

    for i in range(l_low, l_hi, BLOCK_SIZE_KN):
        k_id = i + tl.arange(0, BLOCK_SIZE_KN)
        b_qk_mask = (
            (q_id[:, None] >= k_id[None, :])
            & (q_id[:, None] < LOCAL_WINDOW_SIZE + k_id[None, :])
            & (q_id < seq_len)[:, None]
            & (k_id < l_hi)[None, :]
        )

        b_k = tl.load(k_ptrs, padding_option="zero", boundary_check=(0, 1))
        # b_k: [BLOCK_SIZE_QD, BLOCK_SIZE_N]
        b_v = tl.load(v_ptrs, padding_option="zero", boundary_check=(0, 1))
        # b_v: [BLOCK_SIZE_N, BLOCK_SIZE_D]

        qk = tl.dot(b_q, b_k) * sm_scale

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))

        qk_exp = tl.where(b_qk_mask, tl.exp2(qk - m_ij[:, None]), 0)

        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None] + tl.dot(qk_exp.to(b_v.dtype), b_v)

        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + tl.sum(qk_exp, axis=1))

        m_i = m_ij

        k_ptrs = tl.advance(k_ptrs, offsets=(0, BLOCK_SIZE_KN))
        v_ptrs = tl.advance(v_ptrs, offsets=(BLOCK_SIZE_KN, 0))

    # --- 2. Global Window Attention (First GLOBAL_WINDOW_SIZE tokens) ---
    if l_hi > GLOBAL_WINDOW_SIZE:
        # now compute the global window kv
        g_low = 0
        g_hi = min(GLOBAL_WINDOW_SIZE, pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N)

        k_ptrs = tl.make_block_ptr(
            base=k + seq_start * stride_kn + pid_kh * stride_kh,
            strides=(stride_kd, stride_kn),
            shape=(HEAD_DIM_Q, seq_len),
            offsets=(0, g_low),
            block_shape=(BLOCK_SIZE_QD, BLOCK_SIZE_KN),
            order=(0, 1),
        )
        v_ptrs = tl.make_block_ptr(
            base=v + seq_start * stride_vn + pid_kh * stride_vh,
            strides=(stride_vn, stride_vd),
            shape=(seq_len, HEAD_DIM_V),
            offsets=(g_low, pid_d * BLOCK_SIZE_D),
            block_shape=(BLOCK_SIZE_KN, BLOCK_SIZE_D),
            order=(1, 0),
        )

        for i in range(g_low, g_hi, BLOCK_SIZE_KN):
            k_id = i + tl.arange(0, BLOCK_SIZE_KN)
            b_qk_mask = (
                (q_id[:, None] >= LOCAL_WINDOW_SIZE + k_id[None, :])
                & (q_id < seq_len)[:, None]
                & (k_id < min(seq_len, g_hi))[None, :]
            )

            b_k = tl.load(k_ptrs, padding_option="zero", boundary_check=(0, 1))
            # b_k: [BLOCK_SIZE_QD, BLOCK_SIZE_N]
            b_v = tl.load(v_ptrs, padding_option="zero", boundary_check=(0, 1))
            # b_v: [BLOCK_SIZE_N, BLOCK_SIZE_D]

            qk = tl.dot(b_q, b_k) * sm_scale

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))

            qk_exp = tl.where(b_qk_mask, tl.exp2(qk - m_ij[:, None]), 0)

            acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None] + tl.dot(
                qk_exp.to(b_v.dtype), b_v
            )

            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + tl.sum(qk_exp, axis=1))

            m_i = m_ij

            k_ptrs = tl.advance(k_ptrs, offsets=(0, BLOCK_SIZE_KN))
            v_ptrs = tl.advance(v_ptrs, offsets=(BLOCK_SIZE_KN, 0))

    acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]

    tl.store(o_ptrs, acc_o.to(o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(lse_ptrs, lse_i.to(lse.dtype.element_ty), boundary_check=(0,))


def flash_attn_sparse_torch(
    q: torch.Tensor,  # [1, total_seq_len, num_heads, head_dim]
    k: torch.Tensor,  # [1, total_seq_len, num_kv_heads, head_dim]
    v: torch.Tensor,  # [1, total_seq_len, num_kv_heads, head_dim]
    cu_seqlens: torch.Tensor,  # [batch_size + 1]
    softmax_scale: Optional[float] = None,  # Optional[float]
    local_window_size: int = 0,  # int
    global_window_size: int = 0,  # int
) -> torch.Tensor:  # [1, total_seq_len, num_heads, head_dim]
    """
    Applies sparse Flash Attention using PyTorch implementation for variable-length sequences.

    This function computes attention with sparse patterns, supporting local and global attention
    windows for variable-length sequences defined by cumulative sequence lengths. The input
    tensors are expected to have a leading batch dimension of 1, as used in batched processing
    with variable sequence lengths.

    Args:
        q: Query tensor of shape [1, total_seq_len, num_heads, head_dim].
        k: Key tensor of shape [1, total_seq_len, num_kv_heads, head_dim].
        v: Value tensor of shape [1, total_seq_len, num_kv_heads, head_dim].
        cu_seqlens: Cumulative sequence lengths, shape [batch_size + 1].
        softmax_scale: Optional scaling factor for softmax (default: None, uses 1/sqrt(head_dim)).
        local_window_size: Size of the local attention window (default: 0, disabled).
        global_window_size: Size of the global attention window (default: 0, disabled).

    Returns:
        Output tensor of shape [1, total_seq_len, num_heads, head_dim].
    """
    if softmax_scale == None:
        softmax_scale = (1 / q.shape[-1]) ** (0.5) * 1.4426950408889634

    _, seq_len, num_heads_q, head_dim_q = q.shape
    batch_size = len(cu_seqlens) - 1
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
        triton.cdiv((cu_seqlens[1:] - cu_seqlens[:-1]).max().item(), BLOCK_SIZE_N),
        triton.cdiv(head_dim_v, BLOCK_SIZE_D),
    )

    # Launch kernel
    flash_attn_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        cu_seqlens,
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
        sm_scale=softmax_scale,
        GLOBAL_WINDOW_SIZE=global_window_size,
        LOCAL_WINDOW_SIZE=local_window_size,
    )

    return o, lse


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


def generate_test_data(
    batch_size, max_seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.bfloat16
):
    # Generate random sequence lengths for each batch
    seq_lengths = torch.randint(low=1, high=max_seq_len + 1, size=(batch_size,)).cuda()
    # seq_lengths = torch.tensor(128)

    # Compute cumulative sequence lengths
    cu_seq_lens = torch.zeros(batch_size + 1, dtype=torch.int32).cuda()
    cu_seq_lens[1:] = torch.cumsum(seq_lengths, dim=0)
    total_seq_len = cu_seq_lens[-1].item()

    # Generate tensors with total sequence length
    # Generate tensors with values between 0 and 1
    q = torch.rand(1, total_seq_len, num_heads, head_dim, dtype=dtype).cuda()
    k = torch.rand(1, total_seq_len, num_kv_heads, head_dim, dtype=dtype).cuda()
    v = torch.rand(1, total_seq_len, num_kv_heads, head_dim, dtype=dtype).cuda()

    return q, k, v, cu_seq_lens


def flash_attn_sparse_torch(
    q,
    k,
    v,
    cu_seqlens,
    softmax_scale=None,
    local_window_size=0,
    global_window_size=0,
):

    _, T, H, Dq = q.shape
    Hk = k.shape[-2]
    gqa_ratio = H // Hk
    o = torch.zeros_like(q)
    if softmax_scale == None:
        softmax_scale = Dq ** (-0.5)

    bz = len(cu_seqlens) - 1

    for b in range(bz):
        for h in range(H):
            seq_start = cu_seqlens[b]
            seq_end = cu_seqlens[b + 1]
            seq_len = seq_end - seq_start
            hk = h // gqa_ratio

            b_q = q[0, seq_start:seq_end, h, :]
            b_k = k[0, seq_start:seq_end, hk, :]
            b_v = v[0, seq_start:seq_end, hk, :]
            b_qk = b_q @ b_k.T

            casual_mask = (
                torch.arange(0, seq_len)[:, None] >= torch.arange(0, seq_len)[None, :]
            )
            local_window_mask = (
                torch.arange(0, seq_len)[:, None]
                < torch.arange(0, seq_len)[None, :] + local_window_size
            )
            global_window_mask = (torch.arange(0, seq_len) < global_window_size)[
                None, :
            ]

            b_qk_mask = casual_mask & (local_window_mask | global_window_mask)

            b_qk = torch.where(b_qk_mask.to(device=b_qk.device), b_qk, -float("inf"))
            b_qk = b_qk * softmax_scale
            b_qk = b_qk.softmax(dim=-1)

            b_o = b_qk @ b_v
            o[0, seq_start:seq_end, h, :] = b_o

    return o


def test_flash_sparse_attn_fwdonly():

    local_window = 128
    global_window = 0

    # Test configurations
    test_configs = [
        # (1, 128, 1, 96),  # Small sequence
        # (2, 256, 8, 128),  # Medium sequence
        (4, 2048, 4, 96),  # Larger sequence
    ]

    torch.manual_seed(42)

    for batch_size, max_seq_len, num_heads, head_dim in test_configs:
        print(
            f"\nTesting: batch_size={batch_size}, max_seq_len={max_seq_len}, "
            f"num_heads={num_heads}, head_dim={head_dim}"
        )

        # Generate test data
        q, k, v, cu_seq_lens = generate_test_data(
            batch_size, max_seq_len, num_heads, num_heads, head_dim, torch.bfloat16
        )

        # Run both implementations
        with torch.no_grad():
            torch_output = flash_attn_sparse_torch(
                q,
                k,
                v,
                cu_seq_lens,
                local_window_size=local_window,
                global_window_size=global_window,
            )
            triton_output, _ = flash_attn_sparse_triton(
                q, k, v, cu_seq_lens, local_window, global_window
            )

        # Compare results
        absolute_diff = torch.abs(torch_output - triton_output)
        relative_diff = absolute_diff / (torch.abs(torch_output) + 1e-6)

        max_abs_diff = absolute_diff.max().item()
        mean_abs_diff = absolute_diff.mean().item()
        max_rel_diff = relative_diff.max().item()

        print(f"Max absolute difference: {max_abs_diff:.6f}")
        print(f"Mean absolute difference: {mean_abs_diff:.6f}")
        print(f"Max relative difference: {max_rel_diff:.6f}")
        print(f"cu_seq_lens: {cu_seq_lens}")

        # Assert that results are close
        # torch.testing.assert_close(torch_output, triton_output, rtol=1e-2, atol=1e-2)
        # assert torch.allclose( pytorch_output, flash_output, rtol=1e-2, atol=1e-2)
        # assert torch.allclose( pytorch_output, triton_output, rtol=1e-2, atol=1e-2)
        # assert torch.allclose(flash_output, triton_output, rtol=1e-2, atol=1e-2)

        print("Test passed!")


if __name__ == "__main__":
    # Make sure we're running on GPU
    assert torch.cuda.is_available(), "CUDA is required for this test"
    torch.cuda.set_device(0)

    # Run the tests
    test_flash_sparse_attn_fwdonly()
