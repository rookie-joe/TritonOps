import torch

import triton
import triton.language as tl

# 整体和 matmul 基本完全一致，不过将激活函数relu融合到kernel中


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
def matmul_relu_kernel(
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
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    cur_group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % cur_group_size_m)
    pid_n = (pid % num_pid_in_group) // cur_group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    c = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 0, dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = (offs_am < M)[:, None] & (k + offs_k < K)[None, :]
        b_mask = (k + offs_k < K)[:, None] & (offs_bn < N)[None, :]
        a = tl.load(a_ptrs, a_mask, 0)
        b = tl.load(b_ptrs, b_mask, 0)
        c += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 和 matmul 唯一的区别：在这里加入一个relu操作
    c = tl.where(c >= 0, c, 0)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, c_mask)


def triton_matmul_relu(A: torch.Tensor, B: torch.Tensor):
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
    matmul_relu_kernel[grid](
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
output_torch = torch.relu(torch.mm(A, B))
output_triton = triton_matmul_relu(A, B)

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
