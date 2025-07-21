import torch
import torch.nn as nn
import triton
import triton.language as tl
import pdb
import math

from utils.utils import get_num_warps_stages, is_hopper_gpu

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
    key=["T", "D", "N", "BLOCK_SIZE_T", "BLOCK_SIZE_D", "BLOCK_SIZE_N"],
)
@triton.jit
def cross_entropy_fused_fwd_lse_kernel(
    x,
    w,
    y,
    o,
    lse,
    stride_xt,
    stride_xd,
    stride_wN,
    stride_wd,
    stride_yt,
    stride_ot,
    stride_lse_t,
    stride_lse_n,
    T,
    D,
    N,
    LSE_N,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_n: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_N = tl.program_id(1)

    x_ptrs_base = x + (pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)) * stride_xt
    x_mask_t = (pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)) < T

    lse_ptrs = tl.make_block_ptr(
        base=lse,
        shape=(T, LSE_N),
        strides=(stride_lse_t, stride_lse_n),
        offsets=(pid_t * BLOCK_SIZE_T, pid_N),
        block_shape=(BLOCK_SIZE_T, 1),
        order=(1, 0),
    )
    y_ptrs = tl.make_block_ptr(
        base=y,
        shape=(T,),
        strides=(stride_yt,),
        offsets=(pid_t * BLOCK_SIZE_T,),
        block_shape=(BLOCK_SIZE_T,),
        order=(0,),
    )
    o_ptrs = tl.make_block_ptr(
        base=o,
        shape=(T,),
        strides=(stride_ot,),
        offsets=(pid_t * BLOCK_SIZE_T,),
        block_shape=(BLOCK_SIZE_T,),
        order=(0,),
    )

    b_lse = tl.full((BLOCK_SIZE_T, 1), 0.0, tl.float32)

    m_i = tl.full((BLOCK_SIZE_T,), float("-inf"), dtype=tl.float32)
    i_start = pid_N * BLOCK_SIZE_N
    i_end = tl.minimum((pid_N + 1) * BLOCK_SIZE_N, N)
    for i in range(i_start, i_end, BLOCK_SIZE_n):

        b_xw = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_n), tl.float32)

        for j in range(0, D, BLOCK_SIZE_D):

            x_ptrs = (
                x_ptrs_base[:, None]
                + ((j + tl.arange(0, BLOCK_SIZE_D)) * stride_xd)[None, :]
            )
            x_mask = x_mask_t[:, None] & ((j + tl.arange(0, BLOCK_SIZE_D)) < D)[None, :]

            w_ptrs = (
                w
                + ((j + tl.arange(0, BLOCK_SIZE_D)) * stride_wd)[:, None]
                + ((i + tl.arange(0, BLOCK_SIZE_n)) * stride_wN)[None, :]
            )
            w_mask = ((j + tl.arange(0, BLOCK_SIZE_D)) < D)[:, None] & (
                (i + tl.arange(0, BLOCK_SIZE_n)) < i_end
            )[None, :]

            b_x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            # [BLOCK_SIZE_T, BLOCK_SIZE_D]

            b_w = tl.load(w_ptrs, mask=w_mask, other=0.0)
            # [BLOCK_SIZE_D, BLOCK_SIZE_n]

            b_xw += tl.dot(b_x, b_w)
            # [BLOCK_SIZE_T, BLOCK_SIZE_n]

        m_i = tl.maximum(m_i, tl.max(b_xw, axis=1))

        b_xw_mask = ((pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)) < T)[
            :, None
        ] & (i + tl.arange(0, BLOCK_SIZE_n) < i_end)[None, :]

        b_xw_exp = tl.where(b_xw_mask, tl.exp(b_xw - m_i[:, None]), 0.0)

        b_lse = m_i[:, None] + tl.log(
            tl.exp(b_lse - m_i[:, None]) + tl.sum(b_xw_exp, axis=1, keep_dims=True)
        )

    tl.store(lse_ptrs, b_lse.to(lse_ptrs.dtype.element_ty), boundary_check=(0, 1))

    # cal the numinator
    b_y = tl.load(y_ptrs, boundary_check=(0,), padding_option="zero")
    # [BLOCK_SIZE_T, ]

    b_o = tl.zeros((BLOCK_SIZE_T,), tl.float32)

    for j in range(0, D, BLOCK_SIZE_D):

        x_ptrs = (
            x_ptrs_base[:, None]
            + ((j + tl.arange(0, BLOCK_SIZE_D)) * stride_xd)[None, :]
        )
        x_mask = x_mask_t[:, None] & ((j + tl.arange(0, BLOCK_SIZE_D)) < D)[None, :]

        w_ptrs = (
            w
            + (b_y * stride_wN)[:, None]
            + ((j + tl.arange(0, BLOCK_SIZE_D)) * stride_wd)[None, :]
        )

        w_mask = ((j + tl.arange(0, BLOCK_SIZE_D)) < D)[None, :]

        b_x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        # [BLOCK_SIZE_T, BLOCK_SIZE_D]

        b_w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        # [BLOCK_SIZE_T, BLOCK_SIZE_D]

        b_o += tl.sum(b_x * b_w, axis=-1, keep_dims=False)
        # [BLOCK_SIZE_T, ]

    tl.store(o_ptrs, b_o.to(o_ptrs.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=autotune_configs,
    key=["T", "N", "BLOCK_SIZE_T", "BLOCK_SIZE_N"],
)
@triton.jit
def cross_entropy_fused_bwd_kernel(
    y,
    lse,
    xw,
    do,
    d_xw,
    stride_yt,
    stride_lse_t,
    stride_xw_t,
    stride_xw_N,
    stride_dot,
    stride_dxw_t,
    stride_dxw_N,
    x_start,
    T,
    N,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)

    xw_ptrs = tl.make_block_ptr(
        base=xw,
        shape=(T, N),
        strides=(stride_xw_t, stride_xw_N),
        offsets=(pid_t * BLOCK_SIZE_T, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_N),
        order=(1, 0),
    )
    d_xw_ptrs = tl.make_block_ptr(
        base=d_xw,
        shape=(T, N),
        strides=(stride_dxw_t, stride_dxw_N),
        offsets=(pid_t * BLOCK_SIZE_T, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_N),
        order=(1, 0),
    )

    y_ptrs = tl.make_block_ptr(
        base=y + x_start * stride_yt,
        shape=(T,),
        strides=(stride_yt,),
        offsets=(pid_t * BLOCK_SIZE_T,),
        block_shape=(BLOCK_SIZE_T,),
        order=(0,),
    )

    lse_ptrs = tl.make_block_ptr(
        base=lse + x_start * stride_lse_t,
        shape=(T,),
        strides=(stride_lse_t,),
        offsets=(pid_t * BLOCK_SIZE_T,),
        block_shape=(BLOCK_SIZE_T,),
        order=(0,),
    )

    do_ptrs = tl.make_block_ptr(
        base=do + x_start * stride_dot,
        shape=(T,),
        strides=(stride_dot,),
        offsets=(pid_t * BLOCK_SIZE_T,),
        block_shape=(BLOCK_SIZE_T,),
        order=(0,),
    )

    b_lse = tl.load(lse_ptrs, boundary_check=(0,), padding_option="zero")
    b_do = tl.load(do_ptrs, boundary_check=(0,), padding_option="zero")
    b_y = tl.load(y_ptrs, boundary_check=(0,), padding_option="zero")

    # recompute the softmax for each xw
    b_xw = tl.load(xw_ptrs, boundary_check=(0, 1), padding_option="zero")

    b_xw_sm = tl.exp(b_xw.to(tl.float32) - b_lse[:, None])
    # [BLOCK_SIZE_T, BLOCK_SIZE_N]

    n_ids = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_y_id_mask = b_y[:, None] == n_ids[None, :]
    # [BLOCK_SIZE_T, BLOCK_SIZE_N]

    b_do_xw = b_do[:, None] * (b_xw_sm - n_y_id_mask.to(b_xw_sm.dtype))
    tl.store(d_xw_ptrs, b_do_xw.to(d_xw_ptrs.dtype.element_ty), boundary_check=(0, 1))


# --- PyTorch autograd Function ---
class CrossEntropyFusedFunction(torch.autograd.Function):
    """
    Integrates the custom fused kernel with PyTorch's autograd.
    """

    @staticmethod
    def forward(ctx, x, w, y):
        """
        x: Input tensor (e.g., activations), shape (T, D)
        w: Weight tensor, shape (N, D)
        y: Target labels, shape (T, 1)
        """
        if not x.is_cuda or not w.is_cuda or not y.is_cuda:
            raise NotImplementedError(
                "This kernel currently only supports CUDA tensors."
            )

        # Input validation
        if x.ndim != 2 or w.ndim != 2:
            raise ValueError("x and w must be 2D tensors.")
        if y.ndim != 1:
            if y.shape[-1] != 1:
                raise ValueError("y must be a 1D tensor.")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Dimension mismatch: x.shape[0] ({x.shape[0]}) != y.shape[0] ({y.shape[0]})"
            )

        T, D = x.shape
        N, D = w.shape

        BLOCK_SIZE_T = min(128, triton.next_power_of_2(T))
        BLOCK_SIZE_D = min(128, triton.next_power_of_2(D))
        BLOCK_SIZE_N = min(2048, triton.next_power_of_2(N))
        BLOCK_SIZE_n = min(128, triton.next_power_of_2(N))
        LSE_N = math.ceil(N / BLOCK_SIZE_N)

        o = torch.empty(T, device=x.device, dtype=x.dtype)  # Output loss
        lse_n = torch.empty(
            T, LSE_N, device=x.device, dtype=torch.float32
        )  # lse for bwd

        grid = (triton.cdiv(T, BLOCK_SIZE_T), LSE_N)
        # Call the forward kernel (lse first)
        cross_entropy_fused_fwd_lse_kernel[grid](
            x,
            w,
            y,
            o,
            lse_n,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            y.stride(0),
            o.stride(0),
            lse_n.stride(0),
            lse_n.stride(1),
            T,
            D,
            N,
            LSE_N,
            BLOCK_SIZE_T,
            BLOCK_SIZE_D,
            BLOCK_SIZE_n,
            BLOCK_SIZE_N,
        )
        lse_max = lse_n.max()
        lse = (lse_n - lse_max).exp().sum(dim=-1).log() + lse_max
        o = lse - o

        # Save tensors for backward pass
        ctx.save_for_backward(x, w, y, lse)

        ctx.BLOCK_SIZE_T = BLOCK_SIZE_T
        ctx.BLOCK_SIZE_D = BLOCK_SIZE_D
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.BLOCK_SIZE_n = BLOCK_SIZE_n
        ctx.T = T
        ctx.D = D
        ctx.N = N

        return o  # Return the loss

    @staticmethod
    def backward(ctx, grad_o):
        """
        grad_o: Gradient of the output of the forward pass (loss), shape (T)
        """
        x, w, y, lse = ctx.saved_tensors
        BLOCK_SIZE_T = ctx.BLOCK_SIZE_T
        BLOCK_SIZE_D = ctx.BLOCK_SIZE_D
        BLOCK_SIZE_N = ctx.BLOCK_SIZE_n
        T, D, N = ctx.T, ctx.D, ctx.N

        grad_w = torch.zeros_like(w, dtype=torch.bfloat16)
        grad_x = torch.zeros_like(x, dtype=torch.float32)
        T_partial = 2048

        # Initialize gradient tensors for inputs
        for i in range(0, T, T_partial):
            x_start = i
            x_end = min(i + T_partial, T)
            x_partial = x[x_start:x_end, :]
            xw_partial = torch.mm(x_partial, w.transpose(0, 1))
            grad_xw_partial = torch.zeros_like(xw_partial, dtype=torch.float32)

            if not grad_o.is_cuda:
                raise NotImplementedError(
                    "This kernel currently only supports CUDA tensors for gradients."
                )

            # Get strides for all tensors involved in backward
            # Call the backward kernel
            grid = (
                triton.cdiv(T_partial, BLOCK_SIZE_T),
                triton.cdiv(N, BLOCK_SIZE_N),
            )
            cross_entropy_fused_bwd_kernel[grid](
                y,
                lse,
                xw_partial,
                grad_o,
                grad_xw_partial,
                y.stride(0),
                lse.stride(0),
                xw_partial.stride(0),
                xw_partial.stride(1),
                grad_o.stride(0),
                grad_xw_partial.stride(0),
                grad_xw_partial.stride(1),
                x_start,
                x_end - x_start,
                N,
                BLOCK_SIZE_T,
                BLOCK_SIZE_N,
            )
            grad_x[x_start:x_end, ...] = torch.mm(grad_xw_partial.to(w.dtype), w)
            grad_w = torch.addmm(
                grad_w,
                grad_xw_partial.transpose(0, 1).to(x_partial.dtype),
                x_partial,
                alpha=1,
                beta=1,
            )

        return (
            grad_x,
            grad_w,
            None,
        )


def cross_entropy_fused_triton(x, w, y):

    return CrossEntropyFusedFunction.apply(x, w, y)


def cross_entropy(hidden_states, lm_head_weight, labels):
    flatten_hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    flattn_logits = flatten_hidden_states @ lm_head_weight.T
    flattn_logits_exp = flattn_logits.exp()
    flattn_logits_softmax = flattn_logits.softmax(dim=-1)
    loss = flattn_logits_softmax.gather(dim=-1, index=labels.view(-1, 1)).squeeze(-1)
    return -loss.log()
    # return flattn_logits_exp.sum(dim=-1).log()


def test_cross_entropy():
    torch.manual_seed(42)

    # Test parameters
    batch_size = 1
    seq_length = 139
    vocab_size = 279
    hidden_size = 256
    device = "cuda"
    dtype = torch.bfloat16
    # dtype = torch.float32

    # Generate test data
    hidden_states = torch.rand(
        batch_size,
        seq_length,
        hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    lm_head_weight = torch.rand(
        vocab_size, hidden_size, device=device, dtype=dtype, requires_grad=True
    )

    labels = torch.randint(
        0, vocab_size, (batch_size, seq_length), device=device, dtype=torch.int64
    )

    # Clone inputs for forward and backward testing
    hidden_states_clone = hidden_states.clone().detach().requires_grad_(True)
    lm_head_weight_clone = lm_head_weight.clone().detach().requires_grad_(True)
    labels_clone = labels.clone().detach()

    # Standard cross entropy loss (forward)
    standard_ce = nn.CrossEntropyLoss(reduction="none")
    logits = hidden_states @ lm_head_weight.T
    standard_loss = standard_ce(logits.view(-1, vocab_size), labels.view(-1))

    # Triton cross entropy loss (forward)
    triton_loss = cross_entropy_fused_triton(
        hidden_states_clone.view(-1, hidden_states_clone.size(-1)),
        lm_head_weight_clone,
        labels_clone.view(-1, 1),
    )

    # Compare forward losses

    # torch.testing.assert_close(triton_loss, standard_loss, atol=1e-1, rtol=1e-1)

    # Backward pass for standard loss
    (standard_loss).mean().backward()
    x_torch_grad = hidden_states.grad.clone()
    w_torch_grad = lm_head_weight.grad.clone()

    # Reset gradients
    hidden_states.grad = None
    lm_head_weight.grad = None

    # Backward pass for triton loss
    (triton_loss).mean().backward()
    x_triton_grad = hidden_states_clone.grad.clone()
    w_triton_grad = lm_head_weight_clone.grad.clone()

    # Compare gradients
    torch.testing.assert_close(
        x_torch_grad,
        x_triton_grad,
        atol=1e-1,
        rtol=1e-1,
    )
    torch.testing.assert_close(
        w_torch_grad,
        w_triton_grad,
        atol=1e-1,
        rtol=1e-1,
    )


if __name__ == "__main__":
    test_cross_entropy()
