import math

import torch
import triton
import triton.language as tl
from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_func
from termcolor import colored


@torch.no_grad
def flash_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    qkv = torch.stack([q, k, v], dim=2).contiguous()
    out = flash_attn_func(qkv, causal=True)
    out = out.transpose(1, 2).contiguous()
    return out


@torch.no_grad
def torch_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    b, h, n, d = q.shape
    assert k.shape == q.shape
    assert v.shape == k.shape

    attn_weight = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(d)
    mask = torch.tril(torch.ones(1, 1, n, n, dtype=torch.bool, device=q.device))
    attn_weight.masked_fill_(~mask, float("-inf"))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    o = torch.einsum("bhij,bhjd->bhid", attn_weight, v)
    return o


# autotune config copy from https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64]
    for s in [3, 4, 7]
    for w in [4, 8]
]


def keep(conf):
    BLOCK_SIZE_M = conf.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = conf.kwargs["BLOCK_SIZE_N"]
    if BLOCK_SIZE_M * BLOCK_SIZE_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def attention_kernel(
    q_ptr,  # Q: n x d
    k_ptr,  # K: n x d
    v_ptr,  # V: n x d
    o_ptr,
    # shape
    BATCH_SIZE,
    NUM_HEADS,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    # softmax_scale
    softmax_scale,
    # stride
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    # META parameters
    BLOCK_SIZE_M: tl.constexpr,  # q block size
    BLOCK_SIZE_N: tl.constexpr,  # k block size
):
    # 二维 grid，第一维是 seq 上的划分，第二维是 batch_size x num_head 上的划分
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # 确定是第几个 batch，第几个 head
    pid_b = pid_bh // NUM_HEADS
    pid_h = pid_bh % NUM_HEADS

    # 初始化 qkv 指针
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qb + pid_h * stride_qh,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, HEAD_DIM),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + pid_b * stride_kb + pid_h * stride_kh,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_kn, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_N, HEAD_DIM),
        order=(1, 0),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + pid_b * stride_vb + pid_h * stride_vh,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_N, HEAD_DIM),
        order=(1, 0),
    )
    #  载入 q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    # 中间变量
    off_m = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    off_n = tl.arange(0, BLOCK_SIZE_N)
    max_qk = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=tl.float32)
    sum_exp = tl.full((BLOCK_SIZE_M,), 1, dtype=tl.float32)
    o = tl.full((BLOCK_SIZE_M, HEAD_DIM), 0, dtype=tl.float32)
    # 划分 full 和 causal 两个部分
    full_block_num = pid_m * BLOCK_SIZE_M // BLOCK_SIZE_N
    casual_block_num = (
        tl.cdiv((pid_m + 1) * BLOCK_SIZE_M, BLOCK_SIZE_N) - full_block_num
    )
    # 处理 full 的部分。此时，不需要做 mask。
    for i in range(0, full_block_num):
        # 载入 k
        k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
        # 计算 qk，以及加 mask
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        # 计算max_qk
        last_max_qk = max_qk
        max_qk = tl.maximum(last_max_qk, tl.max(qk, axis=1))
        # 计算 exp(qk-max_qk) 和 exp(last_max_qk - max_qk)
        exp_qk = tl.math.exp2(qk - max_qk[:, None])
        alpha = tl.math.exp2(last_max_qk - max_qk)
        # 计算 sum(exp(qk-max_qk))
        last_sum_exp = sum_exp
        sum_exp = last_sum_exp * alpha + tl.sum(exp_qk, axis=1)
        # 更新 output，这里不去对 sum_exp 进行处理，最后再除就好
        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
        o = o * alpha[:, None] + tl.dot(exp_qk.to(v.dtype), v)
        # 指针移动一个 block
        k_ptrs = tl.advance(k_ptrs, (BLOCK_SIZE_N, 0))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_N, 0))
    # 处理 causal 的部分
    for i in range(full_block_num, full_block_num + casual_block_num):
        # 载入 k
        k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
        # 计算 qk，以及加 mask
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        qk_mask = off_m[:, None] >= (i * BLOCK_SIZE_N + off_n)[None, :]
        qk = tl.where(qk_mask, qk, float("-inf"))
        # 计算max_qk
        last_max_qk = max_qk
        max_qk = tl.maximum(last_max_qk, tl.max(qk, axis=1))
        # 计算 exp(qk-max_qk) 和 exp(last_max_qk - max_qk)
        exp_qk = tl.math.exp2(qk - max_qk[:, None])
        alpha = tl.math.exp2(last_max_qk - max_qk)
        # 计算 sum(exp(qk-max_qk))
        last_sum_exp = sum_exp
        sum_exp = last_sum_exp * alpha + tl.sum(exp_qk, axis=1)
        # 更新 output，这里不去对 sum_exp 进行处理，最后再除就好
        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
        o = o * alpha[:, None] + tl.dot(exp_qk.to(v.dtype), v)
        # 指针移动一个 block
        k_ptrs = tl.advance(k_ptrs, (BLOCK_SIZE_N, 0))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_N, 0))
    o = o / sum_exp[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_on, stride_od),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(o_ptrs, o.to(tl.bfloat16), boundary_check=(0, 1))


def triton_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    b, h, n, d = q.shape
    assert k.shape == q.shape
    assert v.shape == k.shape
    assert q.dtype == torch.bfloat16
    assert d in {16, 32, 64, 128}
    softmax_scale = 1 / math.sqrt(d) * 1.44269504

    o = torch.empty_like(v)

    grid = lambda META: (
        triton.cdiv(n, META["BLOCK_SIZE_M"]),
        b * h,
    )
    attention_kernel[grid](
        q,
        k,
        v,
        o,
        b,
        h,
        n,
        d,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
    )
    return o


torch.manual_seed(42)
B, H, N, D = 2, 24, 1000, 128
Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
K = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
V = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
output_torch = torch_causal_attention(Q, K, V)
output_flash = flash_causal_attention(Q, K, V)
output_triton = triton_causal_attention(Q, K, V)

if torch.allclose(output_torch, output_triton, rtol=1e-2, atol=1e-2):
    print(colored("triton output == torch output:", "green"))
else:
    print(colored("triton output != torch output:", "red"))
print(
    f"The maximum difference between torch and triton is "
    f"{torch.max(torch.abs(output_torch - output_triton))}"
)

if torch.allclose(output_flash, output_triton, rtol=1e-2, atol=1e-2):
    print(colored("triton output == flash output:", "green"))
else:
    print(colored("triton output != flash output:", "red"))

print(
    f"The maximum difference between flash and triton is "
    f"{torch.max(torch.abs(output_flash - output_triton))}"
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # 用作图表x轴的参数名
        x_vals=[1024 * i for i in range(1, 16)],  # `x_name`的不同可能值
        line_arg="provider",  # 其值对应于图表中不同线条的参数名
        # `line_arg`的可能值
        line_vals=["torch", "flash", "triton"],
        # 线条的标签名称
        line_names=[
            "Torch",
            "Flash",
            "Triton",
        ],
        # 线条样式
        styles=[("green", "-"), ("blue", "-"), ("blue", "--")],
        ylabel="TFLOPS",  # y轴的标签名称
        plot_name="attention-performance",  # 图表的名称，也用作保存图表的文件名。
        args={"B": 2, "H": 24, "D": 128},  # 其他参数
    )
)
def benchmark(B, H, N, D, provider):
    # 初始化张量
    q = torch.randn((B, H, N, D), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((B, H, N, D), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((B, H, N, D), device="cuda", dtype=torch.bfloat16)
    quantiles = [0.5, 0.2, 0.8]  # 分位数
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_causal_attention(q, k, v), quantiles=quantiles
        )
    if provider == "flash":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_causal_attention(q, k, v), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_causal_attention(q, k, v), quantiles=quantiles
        )
    perf = lambda ms: 2.0 * 2.0 * B * H * N * N * D / ms * 1e-9
    return perf(ms), perf(max_ms), perf(min_ms)


# 运行基准测试，展示图表和打印数据
benchmark.run(show_plots=True, print_data=True)
