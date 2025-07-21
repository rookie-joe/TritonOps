import torch
import triton
import triton.language as tl


def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight[None, :]


@triton.jit
def rmsnorm_kernel(
    x_ptr,  # 输入
    y_ptr,  # 输出
    weight_ptr,  # weight，一维向量
    stride_x,  # x.stride(0)
    stride_y,  # y.stride(0)
    n_cols,  # 需要做 norm 的维度有多少列
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # 每个kernel处理一行数据
    pid = tl.program_id(0)
    x_row_start_ptr = x_ptr + pid * stride_x

    # 求 mean(pow(x,2))
    block_pow_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for col in range(0, n_cols, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE) + col
        mask = offsets < n_cols
        x = tl.load(x_row_start_ptr + offsets, mask, 0).to(tl.float32)
        block_pow_sum += x * x
    mean_square = tl.sum(block_pow_sum) / n_cols
    rrms = 1 / tl.sqrt(mean_square + eps)

    # 用mean和std进行normalize，同时加上weight和bias
    y_row_start_ptr = y_ptr + pid * stride_y
    for col in range(0, n_cols, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE) + col
        mask = offsets < n_cols
        x = tl.load(x_row_start_ptr + offsets, mask, 0).to(tl.float32)
        w = tl.load(weight_ptr + offsets, mask, 0)
        # layernorm
        y = x * rrms * w
        # 写入结果
        tl.store(y_row_start_ptr + offsets, y, mask=mask)


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    x_size = x.shape
    x = x.reshape(-1, x.shape[-1])
    n_rows, n_cols = x.shape
    assert weight.shape[0] == n_cols

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    y = torch.empty_like(x)
    rmsnorm_kernel[(n_rows,)](
        x,
        y,
        weight,
        x.stride(0),
        y.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE,
        num_warps=num_warps,
    )
    y = y.reshape(*x_size)
    return y


torch.manual_seed(0)
B, N, D = 2, 1000, 512
x = torch.randn(B, N, D, device="cuda")
weight = torch.rand(D, device="cuda")

output_torch = torch_rmsnorm(x, weight)
output_triton = triton_rmsnorm(x, weight)

print("=== torch output ===")
print(output_torch)
print("=== triton output ===")
print(output_triton)
print(
    f"triton output {'==' if torch.allclose(output_torch, output_triton,rtol=1e-6,atol=1e-6) else '!='} torch output:",
)
print(
    f"The maximum difference between torch and triton is "
    f"{torch.max(torch.abs(output_torch - output_triton))}"
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["D"],  # 用作绘图x轴的参数名
        x_vals=[128 * i for i in range(1, 100)],  # 对列取不同值进行测试
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="layernorm-performance",
        args={"B": 2, "N": 1024},
    )
)
def benchmark(B, N, D, provider):
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    weight = torch.rand(D, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_rmsnorm(x, weight), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_rmsnorm(x, weight), quantiles=quantiles
        )
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, print_data=True)
