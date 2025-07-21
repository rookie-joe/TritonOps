import math

import torch
import triton
import triton.language as tl
from termcolor import colored

# 相比 v1，有一些小的优化，去掉一些没用的计算


def torch_full_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    n, d = q.shape
    assert k.shape == q.shape
    assert v.shape == k.shape

    attn_weight = torch.einsum("id,jd->ij", q, k) / math.sqrt(d)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    o = torch.einsum("ij,jd->id", attn_weight, v)
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
    N_CTX,
    HEAD_DIM: tl.constexpr,
    # softmax_scale
    softmax_scale,
    # stride
    stride_qn,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    stride_on,
    stride_od,
    # META parameters
    BLOCK_SIZE_M: tl.constexpr,  # q block size
    BLOCK_SIZE_N: tl.constexpr,  # k block size
):
    pid = tl.program_id(0)

    off_m = tl.arange(0, BLOCK_SIZE_M)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_d = tl.arange(0, HEAD_DIM)
    # 初始化 qkv 指针
    q_ptrs = (
        q_ptr
        + pid * BLOCK_SIZE_N * stride_qn
        + off_m[:, None] * stride_qn
        + off_d[None, :] * stride_qd
    )
    k_ptrs = k_ptr + off_n[:, None] * stride_kn + off_d[None, :] * stride_kd
    v_ptrs = v_ptr + off_n[:, None] * stride_vn + off_d[None, :] * stride_vd
    #  载入 q
    q_mask = (pid * BLOCK_SIZE_M + off_m < N_CTX)[:, None]
    q = tl.load(q_ptrs, q_mask, 0)
    # 中间变量
    max_qk = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=tl.float32)
    sum_exp = tl.full((BLOCK_SIZE_M,), 1, dtype=tl.float32)
    o = tl.full((BLOCK_SIZE_M, HEAD_DIM), 0, dtype=tl.float32)
    # 对 kv 块做循环
    # 易错点：这里如果用 range(0, N, BLOCK_SIZE_N)，那么 i 就不需要再乘一个 BLOCK_SIZE_N 了
    for i in range(0, tl.cdiv(N_CTX, BLOCK_SIZE_N)):
        # 载入 k
        kv_mask = (i * BLOCK_SIZE_N + off_n < N_CTX)[:, None]
        k = tl.load(k_ptrs, kv_mask, 0)
        # 计算 qk，以及加 mask
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        qk_mask = (i * BLOCK_SIZE_N + off_n < N_CTX)[None, :]
        qk = tl.where(qk_mask, qk, float("-inf"))
        # 计算max_qk
        last_max_qk = max_qk
        max_qk = tl.maximum(last_max_qk, tl.max(qk, axis=1))
        # 计算 exp(qk-max_qk) 和 exp(last_max_qk - max_qk)，这里换用 exp2，而不是 exp
        exp_qk = tl.math.exp2(qk - max_qk[:, None])
        alpha = tl.math.exp2(last_max_qk - max_qk)
        # 计算 sum(exp(qk-max_qk))
        last_sum_exp = sum_exp
        sum_exp = last_sum_exp * alpha + tl.sum(exp_qk, axis=1)
        # 更新 output，这里不去对 sum_exp 进行处理，最后再除就好
        v = tl.load(v_ptrs, kv_mask, 0)
        o = o * alpha[:, None] + tl.dot(exp_qk.to(v.dtype), v)
        # 指针移动一个 block
        k_ptrs += BLOCK_SIZE_N * stride_kn
        v_ptrs += BLOCK_SIZE_N * stride_vn

    o = o / sum_exp[:, None]
    off_on = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    o_ptrs = o_ptr + off_on[:, None] * stride_on + off_d[None, :] * stride_od
    tl.store(o_ptrs, o, (off_on < N_CTX)[:, None])


def triton_full_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    n, d = q.shape
    assert k.shape == q.shape
    assert v.shape == k.shape
    assert q.dtype == torch.bfloat16
    assert d in {16, 32, 64, 128}
    o = torch.empty_like(v)
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE_M"]),)
    attention_kernel[grid](
        q,
        k,
        v,
        o,
        n,
        d,
        1
        / math.sqrt(d)
        * 1.44269504,  # 这里的 scale 乘以1/log(2)，kernel 中就可以用 exp2
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
    )
    return o


torch.manual_seed(42)
N, D = 1000, 128
Q = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
K = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
V = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
output_torch = torch_full_attention(Q, K, V)
output_triton = triton_full_attention(Q, K, V)

print("=== torch output ===")
print(output_torch)
print("=== triton output ===")
print(output_triton)

if torch.allclose(output_torch, output_triton, rtol=1e-2, atol=1e-2):
    print(colored("triton output == torch output:", "green"))
else:
    print(colored("triton output != torch output:", "red"))

print(
    f"The maximum difference between torch and triton is "
    f"{torch.max(torch.abs(output_torch - output_triton))}"
)


# @triton.testing.perf_report(
#     triton.testing.Benchmark(
#         x_names=["N"],  # 用作图表x轴的参数名
#         x_vals=[1024 * i for i in range(1, 16)],  # `x_name`的不同可能值
#         line_arg="provider",  # 其值对应于图表中不同线条的参数名
#         # `line_arg`的可能值
#         line_vals=["torch", "triton"],
#         # 线条的标签名称
#         line_names=["Torch", "Triton"],
#         # 线条样式
#         styles=[("green", "-"), ("blue", "-")],
#         ylabel="TFLOPS",  # y轴的标签名称
#         plot_name="attention-performance",  # 图表的名称，也用作保存图表的文件名。
#         args={"D": 128},  # 其他参数
#     )
# )
# def benchmark(N, D, provider):
#     # 初始化张量
#     q = torch.randn((N, D), device="cuda", dtype=torch.bfloat16)
#     k = torch.randn((N, D), device="cuda", dtype=torch.bfloat16)
#     v = torch.randn((N, D), device="cuda", dtype=torch.bfloat16)
#     quantiles = [0.5, 0.2, 0.8]  # 分位数
#     if provider == "torch":
#         ms, min_ms, max_ms = triton.testing.do_bench(
#             lambda: torch_full_attention(q, k, v), quantiles=quantiles
#         )
#     if provider == "triton":
#         ms, min_ms, max_ms = triton.testing.do_bench(
#             lambda: triton_full_attention(q, k, v), quantiles=quantiles
#         )
#     perf = lambda ms: 2.0 * 2.0 * N * N * D / ms * 1e-9
#     return perf(ms), perf(max_ms), perf(min_ms)


# # 运行基准测试，展示图表和打印数据
# benchmark.run(show_plots=True, print_data=True)
