import torch

import triton
import triton.language as tl


@torch.jit.script
def naive_softmax(x):
    x_max = x.max(dim=1)[0]

    z = x - x_max[:, None]

    numerator = torch.exp(z)

    denominator = numerator.sum(dim=1)

    ret = numerator / denominator[:, None]

    return ret


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_ptrs = row_start_ptr + col_offsets

    mask = col_offsets < n_cols
    row = tl.load(row_ptrs, mask, other=float("-inf"))
    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    row_exp = tl.exp(row_minus_max)
    row_exp_sum = tl.sum(row_exp)
    softmax_output = row_exp / row_exp_sum

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_row_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_row_ptrs, softmax_output, mask)


def triton_softmax(x: torch.Tensor):
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    y = torch.empty_like(x)

    # 我们可以使用的另一个技巧是要求编译器通过增加每行分布的warp数（`num_warps`）来使用更多的线程。
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    softmax_kernel[(n_rows,)](
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # 用作绘图x轴的参数名
        x_vals=[128 * i for i in range(1, 100)],  # 对列取不同值进行测试
        line_arg="provider",
        line_vals=["triton", "torch", "torch-naive"],
        line_names=["Triton", "Torch", "Torch-naive"],
        styles=[("blue", "-"), ("green", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 2048},  # 固定行数为 2048
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.softmax(x, dim=-1), quantiles=quantiles
        )
    if provider == "torch-naive":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: naive_softmax(x), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_softmax(x), quantiles=quantiles
        )
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (4096, 2048)
    x = torch.rand(size, device="cuda")
    output_torch = torch.softmax(x, dim=-1)
    output_triton = triton_softmax(x)
    print("triton output == torch output:", torch.allclose(output_torch, output_triton))

    benchmark.run(show_plots=True, print_data=True)
