import torch

import triton
import triton.language as tl


# 注：这里的自动配置调优可以先不管，先看主函数的逻辑
# 使用`triton.jit`装饰的函数可以通过`triton.autotune`装饰器进行自动调优，该装饰器包括：
#   - 一系列定义不同配置的`triton.Config`对象，
#       这些配置涉及元参数（例如`BLOCK_SIZE_M`）和编译选项（例如`num_warps`）的不同设置
#   - 一个自动调优*关键字*，其值的变化将触发对所有
#       提供的配置的评估
@triton.autotune(
    configs=[
        # 每个Config定义了一组特定的配置参数和编译选项
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
    ],
    key=["M", "N", "K"],  # 自动调优关键字
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: int,
    K: int,
    N: int,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # META-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """用于计算矩阵乘法C = A x B的内核。
        A的形状为(M, K)，B的形状为(K, N)，C的形状为(M, N)。
        每个kernel计算A的(BLOCK_SIZE_M x BLOCK_SIZE_K)大小的块和B的(BLOCK_SIZE_K x BLOCK_SIZE_N)大小的块，获得一个
    (BLOCK_SIZE_M x BLOCK_SIZE_N) 大小的结果。即共有 ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N) 个 kernel 要运行。
        在对kernel执行进行的“循环”中，这个“循环”是按照program_id递增来运行的。也即，极端情况下，GPU一次只处理一个block，
    那么就是对program_id从0开始递增执行。
        因此，我们可以通过更改从 program_id 获得当前操作的块的位置的方式，修改期望的运行顺序，
    例如从从先遍历行，再遍历列，更改为先遍历列，再遍历行。
        在这里，我们将块在 M 维度上进行分组，每个组内的块按照列主序运行。极端情况下，GROUP_SIZE_M=1，那么就相当于整体的行主序。
    以16个block举例说明，数字表示pid，也是运行顺序，如果不分组，那么为：
        0  1  2  3
        4  5  6  7
        8  9  10 11
        12 13 14 15
    如果分组为2，那么为：（group内列主序，group自身为行主序）
        0  2  4  6
        1  3  5  7
        8  10 12 14
        9  11 13 15
    从例子中可以看出，只需要对行做分组即可，因为不管是否对列做分组，block的运行顺序都不会变
    """
    # 程序ID
    pid = tl.program_id(axis=0)
    # 沿M轴的程序ID数量
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # 沿N轴的程序ID数量
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # 组中的程序数量，这里只对行做了分组，没有对列做分组
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    #####################
    # 下面的代码就是通过pid，确定这个block具体处于什么位置
    #####################

    # 该程序所在组的ID
    group_id = pid // num_pid_in_group
    # 组中第一个程序的行ID
    first_pid_m = group_id * GROUP_SIZE_M
    # 当前所在group的大小。如果`num_pid_m`不能被`GROUP_SIZE_M`整除，最后一个组更小
    cur_group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # 程序在*启动网格*中的行ID，因为在组内，block为列主序，所以根据group_size_m来确定在组内的第几行
    # pid % num_pid_in_group 为当前block在当前组内是第几个block
    # (pid % num_pid_in_group) % cur_group_size_m 为当前block在当前组内是第几行
    # 注意不能直接用 pid % cur_group_size_m 作为当前block在组内是第几行，因为 cur_group_size 不一定等于 GROUP_SIZE_M
    pid_m = first_pid_m + ((pid % num_pid_in_group) % cur_group_size_m)
    # 程序在*启动网格*中的列ID，注意ID都是0开头的，所以用 // 而不是cdiv
    pid_n = (pid % num_pid_in_group) // cur_group_size_m
    # 此时我们就获得了当前block到底位于网格中的什么位置，也就是我们计算的是这一块的结果:
    # C[pid_m*BLOCK_SIZE_M:(pid_m+1)*BLOCK_SIZE_M, pid_n*BLOCK_SIZE_N:(pid_n+1)*BLOCK_SIZE_N]

    #####################
    # 获取这个块在A，B中的起始指针，我们需要在 K 维度循环，不断推进这个指针
    #####################

    # `a_ptrs`是[BLOCK_SIZE_M, BLOCK_SIZE_K]块的指针
    # `b_ptrs`是[BLOCK_SIZE_K, BLOCK_SIZE_N]块的指针
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    #####################
    # 在 K 维度循环，取得AB对应的全部行列值的矩阵运算结果
    #####################

    # c 用于储存这个块的结果
    c = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 0, dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # 生成mask。此时，offs_am 和 offs_bn 都不会改变，唯一会改变的就是 offs_k，因为在 K 维度循环
        # k + offs_k 就是当前block在k维度的位置，这个位置显然不应该超过 K
        a_mask = (offs_am < M)[:, None] & (k + offs_k < K)[None, :]
        b_mask = (k + offs_k < K)[:, None] & (offs_bn < N)[None, :]
        # 载入数据
        a = tl.load(a_ptrs, a_mask, 0)
        b = tl.load(b_ptrs, b_mask, 0)
        # 累加 block 矩阵乘法的结果
        c += tl.dot(a, b)
        # 指针往 K 方向推进一个 BLOCK
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    #####################
    # 把结果写回 C 矩阵的对应位置
    #####################

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    K, N = B.shape
    # 存储结果
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)
    # 由于这里的 BLOCK_SIZE_M 和 BLOCK_SIZE_N 是自动调优得到的，所以这里的grid只能用函数，不能用预设的元组
    # 也是一维 grid，一共 ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N) 个 kernel 要运行
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    # 启动kernel
    matmul_kernel[grid](
        A,
        B,
        C,  # ABC 为输入输出矩阵
        M,
        K,
        N,  # MNK 为矩阵的size
        A.stride(0),
        A.stride(1),  # A 的stride
        B.stride(0),
        B.stride(1),  # B 的stride
        C.stride(0),
        C.stride(1),  # C 的stride
        # 这里的BLOCK_SIZE，GROUP_SIZE等，是自动调的，所以可以不用写这些参数
    )

    return C


torch.manual_seed(0)
M, K, N = 200, 400, 800
A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)
output_torch = torch.mm(A, B)
output_triton = triton_matmul(A, B)

print("=== torch output ===")
print(output_torch)
print("=== triton output ===")
print(output_triton)
print(
    f"triton output {'==' if torch.allclose(output_torch, output_triton) else '!='} torch output:",
)
print(
    f"The maximum difference between torch and triton is "
    f"{torch.max(torch.abs(output_torch - output_triton))}"
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # 用作图表x轴的参数名
        x_vals=[128 * i for i in range(2, 33)],  # `x_name`的不同可能值
        line_arg="provider",  # 其值对应于图表中不同线条的参数名
        # `line_arg`的可能值
        line_vals=["cublas", "triton"],
        # 线条的标签名称
        line_names=["cuBLAS", "Triton"],
        # 线条样式
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # y轴的标签名称
        plot_name="matmul-performance",  # 图表的名称，也用作保存图表的文件名。
        args={},  # 其他参数
    )
)
def benchmark(M, N, K, provider):
    # 初始化张量
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]  # 分位数
    # 如果提供者是cublas
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.mm(a, b), quantiles=quantiles
        )
    # 如果提供者是triton
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_matmul(a, b), quantiles=quantiles
        )
    # 性能计算函数
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


# 运行基准测试，展示图表和打印数据
benchmark.run(show_plots=True, print_data=True)
