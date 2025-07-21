from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
import pdb


@triton.jit
def flash_attn_decoding_kernel(
    q,
    k,
    v,
    o,
    lse,
    cu_seq_lens,
    window_offsets,
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
    GLOBAL_WINDOW_SIZE: tl.constexpr,
    LOCAL_WINDOW_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_cn = tl.program_id(2)
    pid_kh = pid_h // GQA_RATIO
    seq_start = tl.load(cu_seq_lens + pid_b)
    seq_end = tl.load(cu_seq_lens + pid_b + 1)
    window_offset = max(tl.load(window_offsets + pid_b), 0)
    seq_len = seq_end - seq_start

    if pid_cn > 0:
        kv_offsets = window_offset + pid_cn * BLOCK_SIZE_KN
    else:
        kv_offsets = pid_cn * BLOCK_SIZE_KN

    if kv_offsets >= seq_len:
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
        offsets=(kv_offsets, 0),
        block_shape=(BLOCK_SIZE_KN, BLOCK_SIZE_QD),
        order=(1, 0),
    )

    k_ids = kv_offsets + tl.arange(0, BLOCK_SIZE_KN)
    k_ptrs_mask = (k_ids < seq_len) & (
        (k_ids < GLOBAL_WINDOW_SIZE) | (seq_len <= k_ids + LOCAL_WINDOW_SIZE)
    )

    v_ptrs = tl.make_block_ptr(
        base=v + pid_b * stride_vb + pid_kh * stride_vh,
        strides=(stride_vn, stride_vd),
        shape=(seq_len, HEAD_DIM_V),
        offsets=(kv_offsets, 0),
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
    mi = tl.max(qk, axis=-1)

    # qk : [BLOCK_SIZE_KN, ]
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
    window_offsets,
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
    window_offset = max(tl.load(window_offsets + pid_b), 0)
    seq_len = seq_end - seq_start - window_offset

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


def flash_attention_sparse_decoding(
    q,  # [batch_size, num_heads, head_dim_q]
    k,  # [1, seq_len, num_kv_heads, head_dim_q]
    v,  # [1, seq_len, num_kv_heads, head_dim_v]
    cu_seq_lens,  # [batch_size + 1]
    global_window_size=64,
    local_window_size=64,
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
    qk_scale = head_dim_q ** (-0.5)
    gqa_ratio = num_heads // num_kv_heads

    assert global_window_size in {
        64,
        128,
        256,
    }, f"global_window_size must be one of 64, 128, or 256, got {global_window_size}"

    # Compute maximum sequence length for grid configuration
    max_seq_len = min(
        (cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item(),
        local_window_size + global_window_size,
    )
    window_offsets = (
        cu_seq_lens[1:] - cu_seq_lens[:-1] - (local_window_size + global_window_size)
    ).to(torch.int32)

    # Prepare intermediate tensors
    block_size_kn = global_window_size
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
        window_offsets,
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
        global_window_size,
        local_window_size,
    )
    has_nan = torch.isnan(intermediate_o).any()
    has_inf = torch.isinf(intermediate_o).any()
    if has_nan:
        import ipdb

        ipdb.set_trace()
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
        window_offsets,
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
    global_window_size=64,
    local_window_size=64,
):
    # during decoding, we note that q in a shape of [B, H, Dq]
    # whiel kv in a shape [1, T, H, DK/DV] and cu_seq_lens only indicate the kv lens
    _, T, H, Dv = v.shape
    Dq = q.shape[-1]
    B = len(cu_seq_lens) - 1
    o = torch.zeros(B, H, Dv, device=q.device, dtype=q.dtype)
    qk_scale = Dq ** (-0.5)
    for i in range(B):
        seq_start = cu_seq_lens[i]
        seq_end = cu_seq_lens[i + 1]
        seq_len = seq_end - seq_start
        b_q = q[i, ...]  # [H, Dq]
        b_k = k[0, seq_start:seq_end, ...]  # [T, H, DK]
        b_v = v[0, seq_start:seq_end, ...]  # [T, H, DV]

        local_window_mask = (
            seq_len.item() <= torch.arange(0, seq_len) + local_window_size
        )
        global_window_mask = torch.arange(0, seq_len) < global_window_size

        b_qk_mask = local_window_mask | global_window_mask
        qk = torch.einsum("HD, KHD->HK", b_q, b_k).to(b_q.dtype) * qk_scale
        qk = torch.where(b_qk_mask.to(device=qk.device)[None, :], qk, -float("inf"))

        softmax_qk = qk.softmax(dim=-1)

        o_i = torch.einsum("HK, KHD-> HD", softmax_qk.to(q.dtype), b_v)

        o[i, ...] = o_i

    return o


def test_flash_attention_decoding_original():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define test dimensions
    num_heads = 24
    num_kv_heads = 4  # GQA ratio = 2
    head_dim_q = 128
    head_dim_v = 128
    local_window_size = 4096
    global_window_size = 256

    # Create sequence lengths
    # For simplicity, we'll use fixed sequence lengths
    seq_lens = torch.tensor([512, 2561, 8192], device="cuda", dtype=torch.int32)
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
    triton_output = flash_attention_sparse_decoding(
        q,
        k_triton,
        v_triton,
        cu_seq_lens,
        global_window_size=global_window_size,
        local_window_size=local_window_size,
    )

    # Run the PyTorch implementation
    if num_heads > num_kv_heads:
        k = k[:, :, :, None, :].expand(
            1, k.shape[1], num_kv_heads, num_heads // num_kv_heads, head_dim_q
        )
        k = k.reshape(1, k.shape[1], num_heads, head_dim_q)

        v = v[:, :, :, None, :].expand(
            1, v.shape[1], num_kv_heads, num_heads // num_kv_heads, head_dim_v
        )
        v = v.reshape(1, k.shape[1], num_heads, head_dim_v)

    torch_output = flash_attn_decoding_torch(
        q,
        k,
        v,
        cu_seq_lens,
        global_window_size=global_window_size,
        local_window_size=local_window_size,
    )

    # Check if results are close
    torch.testing.assert_close(triton_output, torch_output, rtol=1e-2, atol=1e-2)
    # torch.testing.assert_close(triton_output[1, 1, :], torch_output[1, 1, :], rtol=1e-2, atol=1e-2)

    print("Test passed! Triton and PyTorch implementations give similar results.")


def test_flash_attention_decoding():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define parameter ranges for testing
    batch_sizes = [3]  # Small, medium, large batch sizes
    seq_lens_options = [
        [0, 512, 2049],  # Includes empty sequence
        [1, 2, 4],  # Very short sequences
        [512, 2049, 8192],  # Mixed lengths
        [1024, 1024, 1024],  # Uniform medium length
    ]
    num_heads_options = [4, 8, 16]  # Varying number of heads
    num_kv_heads_options = [1, 2, 4]  # Ensure num_heads % num_kv_heads == 0
    head_dim_options = [32, 64, 96]  # Varying head dimensions
    qkv_scales = [1.0, 10.0, 50.0, 100.0]  # Moderate norm scaling
    window_sizes = [(64, 1024), (128, 128), (512, 512)]  # (global, local)

    # Generate test cases
    case_id = 0
    for batch_size in batch_sizes:
        for seq_lens in (
            seq_lens_options[:batch_size]
            if batch_size <= len(seq_lens_options)
            else seq_lens_options[-1:]
        ):
            for num_heads in num_heads_options:
                for num_kv_heads in num_kv_heads_options:
                    if num_heads % num_kv_heads != 0:  # Skip invalid GQA ratios
                        continue
                    for head_dim_q, head_dim_v in zip(
                        head_dim_options, head_dim_options
                    ):
                        for qkv_scale in qkv_scales:
                            for global_window_size, local_window_size in window_sizes:
                                case_id += 1
                                case_name = f"Case_{case_id}_bs{batch_size}_seq{seq_lens}_h{num_heads}_kvh{num_kv_heads}_hd{head_dim_q}_scale{qkv_scale}_win{global_window_size}_{local_window_size}"
                                print(f"\nRunning test case: {case_name}")
                                try:
                                    # Create cumsum sequence lengths
                                    seq_lens_tensor = torch.tensor(
                                        seq_lens, device="cuda", dtype=torch.int32
                                    )
                                    cu_seq_lens = torch.cat(
                                        [
                                            torch.zeros(
                                                1, device="cuda", dtype=torch.int32
                                            ),
                                            torch.cumsum(seq_lens_tensor, dim=0),
                                        ],
                                        dim=0,
                                    )
                                    seq_len = cu_seq_lens[-1].item()

                                    # Create q, k, v tensors
                                    q = (
                                        torch.randn(
                                            batch_size,
                                            num_heads,
                                            head_dim_q,
                                            device="cuda",
                                            dtype=torch.bfloat16,
                                        )
                                        * qkv_scale
                                    )
                                    k = (
                                        torch.randn(
                                            batch_size,
                                            seq_len,
                                            num_kv_heads,
                                            head_dim_q,
                                            device="cuda",
                                            dtype=torch.bfloat16,
                                        )
                                        * qkv_scale
                                    )
                                    v = (
                                        torch.randn(
                                            batch_size,
                                            seq_len,
                                            num_kv_heads,
                                            head_dim_v,
                                            device="cuda",
                                            dtype=torch.bfloat16,
                                        )
                                        * qkv_scale
                                    )

                                    # Run the Triton implementation
                                    triton_output = flash_attention_sparse_decoding(
                                        q,
                                        k,
                                        v,
                                        cu_seq_lens,
                                        global_window_size=global_window_size,
                                        local_window_size=local_window_size,
                                    )

                                    # Check for NaN, inf, or zero outputs
                                    has_nan = torch.isnan(triton_output).any()
                                    has_inf = torch.isinf(triton_output).any()
                                    is_zero = torch.all(triton_output == 0)
                                    print(f"Output shape: {triton_output.shape}")
                                    print(f"Has NaN: {has_nan}")
                                    if has_nan:
                                        import pdb

                                        pdb.set_trace()
                                    print(f"Has inf: {has_inf}")
                                    print(f"Is all zero: {is_zero}")
                                    if has_nan or has_inf or is_zero:
                                        print(f"Problem detected in case: {case_name}")
                                        print(f"Input seq_lens: {seq_lens}")
                                        print(
                                            f"q stats: min={q.min().item()}, max={q.max().item()}, mean={q.mean().item()}"
                                        )
                                        print(
                                            f"k stats: min={k.min().item()}, max={k.max().item()}, mean={k.mean().item()}"
                                        )
                                        print(
                                            f"v stats: min={v.min().item()}, max={v.max().item()}, mean={v.mean().item()}"
                                        )

                                except Exception as e:
                                    print(f"Error in case {case_name}: {str(e)}")


if __name__ == "__main__":
    test_flash_attention_decoding_original()
    # test_flash_attention_decoding()
