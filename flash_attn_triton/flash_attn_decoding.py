from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_decoding_kernel(
    q,
    k,
    v,
    o,
    lse,
    cu_seq_lens,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_on,
    stride_o_cn,
    stride_oh,
    stride_od,
    stride_lse_n,
    stride_lse_cn,
    stride_lse_h,
    HEAD_DIM_Q: tl.constexpr,
    BLOCK_SIZE_QD: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
    BLOCK_SIZE_KN: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    qk_scale: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_cn = tl.program_id(2)
    pid_kh = pid_h // GQA_RATIO
    seq_start = tl.load(cu_seq_lens + pid_b)
    seq_end = tl.load(cu_seq_lens + pid_b + 1)
    seq_len = seq_end - seq_start
    if pid_cn * BLOCK_SIZE_KN >= seq_len:
        return

    # only single q
    q_ptrs = tl.make_block_ptr(
        base=q + pid_b * stride_qn + pid_h * stride_qh,
        strides=(stride_qd,),
        shape=(HEAD_DIM_Q,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_QD,),
        order=(0,),
    )

    k_ptrs = tl.make_block_ptr(
        base=k + pid_b * stride_kb + pid_kh * stride_kh,
        strides=(stride_kn, stride_kd),
        shape=(seq_len, HEAD_DIM_Q),
        offsets=(pid_cn * BLOCK_SIZE_KN, 0),
        block_shape=(BLOCK_SIZE_KN, BLOCK_SIZE_QD),
        order=(1, 0),
    )
    # k_ptrs_base = (
    #     k
    #     + seq_start * stride_kn
    #     + pid_kh * stride_kh
    #     + pid_cn * BLOCK_SIZE_KN * stride_kn
    # )
    # k_ptrs_offsets = (
    #     tl.arange(0, BLOCK_SIZE_KN)[:, None] * stride_kn
    #     + tl.arange(0, BLOCK_SIZE_QD)[None, :] * stride_kd
    # )
    k_ptrs_mask = (pid_cn * BLOCK_SIZE_KN + tl.arange(0, BLOCK_SIZE_KN)) < seq_len
    # k_ptrs = k_ptrs_base + k_ptrs_offsets

    v_ptrs = tl.make_block_ptr(
        base=v + pid_b * stride_vb + pid_kh * stride_vh,
        strides=(stride_vn, stride_vd),
        shape=(seq_len, HEAD_DIM_V),
        offsets=(pid_cn * BLOCK_SIZE_KN, 0),
        block_shape=(BLOCK_SIZE_KN, BLOCK_SIZE_VD),
        order=(1, 0),
    )

    o_ptrs = tl.make_block_ptr(
        base=o + pid_b * stride_on + pid_cn * stride_o_cn + pid_h * stride_oh,
        strides=(stride_od,),
        shape=(HEAD_DIM_V,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_VD,),
        order=(0,),
    )

    lse_ptrs = (
        lse + pid_b * stride_lse_n + pid_cn * stride_lse_cn + pid_h * stride_lse_h
    )

    b_q = tl.load(q_ptrs, padding_option="zero", boundary_check=(0,))
    b_k = tl.load(k_ptrs, padding_option="zero", boundary_check=(0, 1))
    # b_k : [BLOCK_SIZE_KN, BLOCK_SIZE_QD]
    b_v = tl.load(v_ptrs, padding_option="zero", boundary_check=(0, 1))
    # b_v : [BLOCK_SIZE_KN, BLOCK_SIZE_VD]
    qk = tl.sum(b_q[None, :] * b_k, axis=1).to(tl.float32) * qk_scale
    qk = tl.where(k_ptrs_mask, qk, float("-inf"))
    # qk : [BLOCK_SIZE_KN, ]
    mi = tl.max(qk, axis=-1)
    lse_i = mi + tl.math.log2(tl.sum(tl.exp2(qk - mi)))
    # qk_exp = tl.exp2(qk - mi)
    b_o = tl.sum((tl.exp2(qk - lse_i))[:, None] * b_v, axis=0)

    # b_o *= tl.exp2(mi)
    # b_o : [BLOCK_SIZE_VD, ]

    tl.store(o_ptrs, b_o.to(o_ptrs.dtype.element_ty), boundary_check=(0,))
    tl.store(lse_ptrs, lse_i.to(lse_ptrs.dtype.element_ty))


@triton.jit
def flash_attn_decoding_correction_kernel(
    o,
    lse,
    final_o,
    cu_seq_lens,
    stride_on,
    stride_o_cn,
    stride_oh,
    stride_od,
    stride_lse_n,
    stride_lse_cn,
    stride_lse_h,
    stride_final_o_n,
    stride_final_o_h,
    stride_final_o_d,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_KN: tl.constexpr,
    BLOCK_CHUNK_NUM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    seq_start = tl.load(cu_seq_lens + pid_b)
    seq_end = tl.load(cu_seq_lens + pid_b + 1)
    seq_len = seq_end - seq_start
    CHUNK_NUM = (seq_len + BLOCK_SIZE_KN - 1) // BLOCK_SIZE_KN
    o_ptrs = tl.make_block_ptr(
        base=o + pid_b * stride_on + pid_h * stride_oh,
        strides=(stride_o_cn, stride_od),
        shape=(CHUNK_NUM, HEAD_DIM),
        offsets=(0, 0),
        block_shape=(BLOCK_CHUNK_NUM, BLOCK_SIZE_D),
        order=(1, 0),
    )

    final_o_ptrs = tl.make_block_ptr(
        base=final_o + pid_b * stride_final_o_n + pid_h * stride_final_o_h,
        strides=(stride_final_o_d,),
        shape=(HEAD_DIM,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_D,),
        order=(0,),
    )
    lse_base = lse + pid_b * stride_lse_n + pid_h * stride_lse_h
    lse_offsets = tl.arange(0, BLOCK_CHUNK_NUM)
    g_se = tl.zeros((1,), dtype=tl.float32)
    m_i = tl.full((1,), value=-float("inf"), dtype=tl.float32)
    for i in range(0, CHUNK_NUM, BLOCK_CHUNK_NUM):
        lse_ptrs_tmp = lse_base + (i + lse_offsets) * stride_lse_cn
        lse_mask = (i + lse_offsets) < CHUNK_NUM

        b_lse = tl.load(lse_ptrs_tmp, mask=lse_mask, other=-float("inf"))
        # b_lse: [BLOCK_CHUNK_NUM, ]
        b_lse_max = tl.maximum(tl.max(b_lse, axis=0), m_i)

        g_se = g_se * tl.exp2(m_i - b_lse_max) + tl.sum(tl.exp2(b_lse - b_lse_max))

        m_i = b_lse_max

    g_lse = tl.log2(g_se) + m_i

    # re_initialize the lse ptrs
    lse_ptrs = tl.make_block_ptr(
        base=lse + pid_b * stride_lse_n + pid_h * stride_lse_h,
        strides=(stride_lse_cn,),
        shape=(CHUNK_NUM,),
        offsets=(0,),
        block_shape=(BLOCK_CHUNK_NUM,),
        order=(0,),
    )

    acc_o = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    for i in range(0, CHUNK_NUM, BLOCK_CHUNK_NUM):
        b_o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
        # b_o: [BLOCK_CHUNK_NUM, BLOCK_SIZE_D]

        b_lse = tl.load(lse_ptrs, boundary_check=(0,), padding_option="zero")
        # b_lse: [BLOCK_CHUNK_NUM, ]

        acc_o += tl.sum((b_o * tl.exp2(b_lse - g_lse[:])[:, None]), axis=0)

        o_ptrs = tl.advance(o_ptrs, (BLOCK_CHUNK_NUM, 0))
        lse_ptrs = tl.advance(lse_ptrs, (BLOCK_CHUNK_NUM,))

    tl.store(final_o_ptrs, acc_o.to(final_o_ptrs.dtype.element_ty), boundary_check=(0,))


def flash_attention_decoding(
    q,  # [batch_size, num_heads, head_dim_q]
    k,  # [batch_size, max_seq_len, num_kv_heads, head_dim_q]
    v,  # [batch_size, max_seq_len, num_kv_heads, head_dim_v]
    cu_seq_lens,  # [batch_size + 1]
    qk_scale=None,
):
    """
    Function to call both flash attention decoding kernels sequentially.

    Args:
        q: Query tensor [batch_size, num_heads, head_dim_q]
        k: Key tensor [batch_size, max_seq_len, num_kv_heads, head_dim_q]
        v: Value tensor [batch_size, max_seq_len, num_kv_heads, head_dim_v]
        cu_seq_lens: Cumulative sequence lengths [batch_size + 1]

    Returns:
        output: Output tensor after attention [batch_size, num_heads, head_dim_v]
    """

    batch_size, head_dim_q = q.shape[0], q.shape[-1]
    num_heads = q.shape[1]
    num_kv_heads, head_dim_v = v.shape[2], v.shape[-1]
    if qk_scale == None:
        qk_scale = head_dim_q ** (-0.5)

    gqa_ratio = num_heads // num_kv_heads

    # Compute maximum sequence length for grid configuration
    max_seq_len = (cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item()

    # Prepare intermediate tensors
    block_size_kn = min(128, triton.next_power_of_2(max_seq_len))
    chunk_nums = (max_seq_len + block_size_kn - 1) // block_size_kn

    # Intermediate outputs
    intermediate_o = torch.zeros(
        (batch_size, chunk_nums, num_heads, head_dim_v), device=q.device, dtype=q.dtype
    )

    # Log-sum-exp for numerical stability
    lse = torch.zeros(
        (batch_size, chunk_nums, num_heads), device=q.device, dtype=torch.float32
    )

    # Final output
    final_o = torch.zeros(
        (batch_size, num_heads, head_dim_v), device=q.device, dtype=q.dtype
    )

    # Grid configuration for first kernel
    grid_1 = (
        batch_size,  # pid_b
        num_heads,  # pid_h
        chunk_nums,  # pid_cn
    )
    BLOCK_SIZE_QD = triton.next_power_of_2(head_dim_q)
    BLOCK_SIZE_VD = triton.next_power_of_2(head_dim_v)

    # Launch first kernel
    flash_attn_decoding_kernel[grid_1](
        q,
        k,
        v,
        intermediate_o,
        lse,
        cu_seq_lens,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        intermediate_o.stride(0),
        intermediate_o.stride(1),
        intermediate_o.stride(2),
        intermediate_o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        head_dim_q,
        BLOCK_SIZE_QD,
        head_dim_v,
        BLOCK_SIZE_VD,
        block_size_kn,
        gqa_ratio,
        qk_scale * 1.442695,
    )

    # Grid configuration for second kernel
    grid_2 = (
        batch_size,  # pid_b
        num_heads,  # pid_h
    )
    BLOCK_CHUNK_NUM = min(triton.next_power_of_2(chunk_nums), 128)

    # Launch second kernel
    flash_attn_decoding_correction_kernel[grid_2](
        intermediate_o,
        lse,
        final_o,
        cu_seq_lens,
        intermediate_o.stride(0),
        intermediate_o.stride(1),
        intermediate_o.stride(2),
        intermediate_o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        final_o.stride(0),
        final_o.stride(1),
        final_o.stride(2),
        head_dim_v,
        BLOCK_SIZE_VD,
        block_size_kn,
        BLOCK_CHUNK_NUM,
    )

    return final_o


def flash_attn_decoding_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    qk_scale=None,
):
    # during decoding, we note that q in a shape of [B, H, Dq]
    # whiel kv in a shape [1, T, H, DK/DV] and cu_seq_lens only indicate the kv lens
    _, T, H, Dv = v.shape
    Dq = q.shape[-1]
    B = len(cu_seq_lens) - 1
    o = torch.zeros(B, H, Dv, device=q.device, dtype=q.dtype)
    if qk_scale == None:
        qk_scale = Dq ** (-0.5)
    for i in range(B):
        seq_start = cu_seq_lens[i]
        seq_end = cu_seq_lens[i + 1]
        b_q = q[i, ...]  # [H, Dq]
        b_k = k[0, seq_start:seq_end, ...]  # [T, H, DK]
        b_v = v[0, seq_start:seq_end, ...]  # [T, H, DV]
        qk = torch.einsum("HD, KHD->HK", b_q, b_k).to(b_q.dtype) * qk_scale

        softmax_qk = qk.softmax(dim=-1)

        o_i = torch.einsum("HK, KHD-> HD", softmax_qk.to(q.dtype), b_v)

        o[i, ...] = o_i

    return o


def test_flash_attention_decoding():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define test dimensions
    num_heads = 2
    num_kv_heads = 2  # GQA ratio = 2
    head_dim_q = 96
    head_dim_v = 96

    # Create sequence lengths
    # For simplicity, we'll use fixed sequence lengths
    seq_lens = torch.tensor([512, 2049, 8192], device="cuda", dtype=torch.int32)
    cu_seq_lens = torch.cat(
        [
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.cumsum(seq_lens, dim=0),
        ],
        dim=0,
    )

    seq_len = cu_seq_lens[-1].item()
    # Create random tensors
    batch_size = len(cu_seq_lens) - 1
    q = torch.randn(
        batch_size, num_heads, head_dim_q, device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        1, seq_len, num_kv_heads, head_dim_q, device="cuda", dtype=torch.float16
    )
    v = torch.randn(
        1, seq_len, num_kv_heads, head_dim_v, device="cuda", dtype=torch.float16
    )
    k_triton = torch.zeros(
        batch_size,
        seq_len,
        num_kv_heads,
        head_dim_q,
        device="cuda",
        dtype=torch.float16,
    )
    v_triton = torch.zeros(
        batch_size,
        seq_len,
        num_kv_heads,
        head_dim_v,
        device="cuda",
        dtype=torch.float16,
    )
    for i in range(batch_size):
        seq_start = cu_seq_lens[i]
        seq_end = cu_seq_lens[i + 1]
        seq_len = seq_end - seq_start
        k_triton[i, :seq_len, ...] = k[0, seq_start:seq_end, ...]
        v_triton[i, :seq_len, ...] = v[0, seq_start:seq_end, ...]

    # Run the Triton implementation
    triton_output = flash_attention_decoding(q, k_triton, v_triton, cu_seq_lens)

    # Run the PyTorch implementation
    torch_output = flash_attn_decoding_torch(q, k, v, cu_seq_lens)

    # Check if results are close
    torch.testing.assert_close(triton_output, torch_output, rtol=1e-2, atol=1e-2)
    # torch.testing.assert_close(triton_output[1, 1, :], torch_output[1, 1, :], rtol=1e-2, atol=1e-2)

    print("Test passed! Triton and PyTorch implementations give similar results.")


if __name__ == "__main__":
    test_flash_attention_decoding()
