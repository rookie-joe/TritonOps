import torch

import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows: int,
    n_cols: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # 注意此时每个 kernel 处理的是多行（BLOCK_SIZE_M），但 grid 依然是一维的
    pid_0 = tl.program_id(0)
    # row 的 offset，即当前处理的是哪些行
    # input_ptr + row_offsets * input_row_stride 就得到了每一行的开始指针
    row_offsets = tl.arange(0, BLOCK_SIZE_M) + pid_0 * BLOCK_SIZE_M

    max_x = tl.full((BLOCK_SIZE_M,), -float("inf"), dtype=tl.float32)
    sum_exp_x = tl.full((BLOCK_SIZE_M,), 0, dtype=tl.float32)

    # online softmax pass 1
    for i in range(0, n_cols, BLOCK_SIZE_N):
        # col 的 offset，即当前处理的是哪些列
        col_offsets = tl.arange(0, BLOCK_SIZE_N) + i
        # 这个 block 的所有元素的 offset，此处的 offsets 是一个二维的张量，offsets[i,j] 就代表ij位置的元素的offset
        offsets = row_offsets[:, None] * input_row_stride + col_offsets[None, :]
        # mask 掉超出 n_rows 或 n_cols 的行，此处的 mask 也是一个二维的张量
        mask = (row_offsets < n_rows)[:, None] & (col_offsets < n_cols)[None, :]
        # 载入一个 block 的元素
        x = tl.load(input_ptr + offsets, mask, 0)
        # 获取这一个 block 每行的最大值
        new_max_x = tl.max(x, axis=1)
        sum_exp_x = sum_exp_x * tl.exp(max_x - new_max_x) + tl.sum(
            tl.exp(x - new_max_x[:, None]), axis=1
        )
        max_x = new_max_x

    # online softmax pass 2
    for i in range(0, n_cols, BLOCK_SIZE_N):
        col_offsets = tl.arange(0, BLOCK_SIZE_N) + i
        offsets = row_offsets[:, None] * input_row_stride + col_offsets[None, :]
        mask = (row_offsets < n_rows)[:, None] & (col_offsets < n_cols)[None, :]
        x = tl.load(input_ptr + offsets, mask, -float("inf"))
        # 计算这个 block 的 softmax 值
        softmax_x = tl.exp(x - max_x[:, None]) / sum_exp_x[:, None]
        # 将结果存到 output 对应的位置中
        offsets = row_offsets[:, None] * output_row_stride + col_offsets[None, :]
        tl.store(output_ptr + offsets, softmax_x, mask)


def triton_softmax(x: torch.Tensor, BLOCK_SIZE_M: int = 1, BLOCK_SIZE_N: int = 1024):
    n_rows, n_cols = x.shape
    # block size 必须为 2 的幂
    BLOCK_SIZE_M = triton.next_power_of_2(BLOCK_SIZE_M)
    BLOCK_SIZE_N = triton.next_power_of_2(BLOCK_SIZE_N)

    # 行的分块数目
    n_row_blocks = triton.cdiv(n_rows, BLOCK_SIZE_M)
    # 结果的存储张量
    y = torch.empty_like(x)
    # 调用 kernel，这里依然是一维的 grid

    num_warps = 4
    if n_cols >= 2048:
        num_warps = 8
    if n_cols >= 4096:
        num_warps = 16
    softmax_kernel[(n_row_blocks,)](
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        num_warps=num_warps,
    )
    return y


torch.manual_seed(0)
size = (4096, 2048)
x = torch.rand(size, device="cuda")
output_torch = torch.softmax(x, dim=-1)
output_triton = triton_softmax(x)
print("triton output == torch output:", torch.allclose(output_torch, output_triton))
print(
    f"The maximum difference between torch and triton is "
    f"{torch.max(torch.abs(output_torch - output_triton))}"
)


# 固定行数，变动列数，速度测试
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # 用作绘图x轴的参数名
        x_vals=[128 * i for i in range(1, 100)],  # 对列取不同值进行测试
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
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
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_softmax(x, 1, 1024), quantiles=quantiles
        )
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, print_data=True)


# 变动行数，固定列数，速度测试
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],  # 用作绘图x轴的参数名
        x_vals=[128 * i for i in range(1, 100)],  # 对列取不同值进行测试
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"N": 256},  # 固定列为 256
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.softmax(x, dim=-1), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_softmax(x, 4, 256), quantiles=quantiles
        )
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, print_data=True)
