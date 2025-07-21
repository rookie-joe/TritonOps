import torch
import torch.nn as nn
from torch.autograd import Function
import triton.language as tl
import triton


@triton.jit
def rmsnorm_fwd_kernel(
    x,
    w,
    y,
    rsqrt,
    stride_x_t,
    stride_x_d,
    stride_w_d,
    stride_y_t,
    stride_y_d,
    stride_rsqrt_t,
    eps: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    T: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    x_ptrs = tl.make_block_ptr(
        base=x,
        shape=(T, D),
        offsets=(pid_t * BLOCK_SIZE_T, 0),
        block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_D),
        strides=(stride_x_t, stride_x_d),
        order=(1, 0),
    )
    w_ptrs = tl.make_block_ptr(
        base=w,
        shape=(D,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_D,),
        strides=(stride_w_d,),
        order=(0,),
    )
    y_ptrs = tl.make_block_ptr(
        base=y,
        shape=(T, D),
        offsets=(pid_t * BLOCK_SIZE_T, 0),
        block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_D),
        strides=(stride_y_t, stride_y_d),
        order=(1, 0),
    )
    rsqrt_ptrs = tl.make_block_ptr(
        base=rsqrt,
        shape=(T,),
        offsets=(pid_t * BLOCK_SIZE_T,),
        block_shape=(BLOCK_SIZE_T,),
        strides=(stride_rsqrt_t,),
        order=(0,),
    )

    sum_xx = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_D), dtype=tl.float32)
    for i in range(0, D, BLOCK_SIZE_D):
        b_x = tl.load(x_ptrs, boundary_check=(0, 1), padding_option="zero")
        sum_xx += b_x * b_x
        x_ptrs = tl.advance(x_ptrs, (0, BLOCK_SIZE_D))

    sum_xx = tl.sum(sum_xx, axis=1)

    rsqrt = 1.0 / tl.sqrt(eps + sum_xx / D)
    # reset x_ptrs
    x_ptrs = tl.make_block_ptr(
        base=x,
        shape=(T, D),
        offsets=(pid_t * BLOCK_SIZE_T, 0),
        block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_D),
        strides=(stride_x_t, stride_x_d),
        order=(1, 0),
    )
    for i in range(0, D, BLOCK_SIZE_D):
        b_x = tl.load(x_ptrs, boundary_check=(0, 1), padding_option="zero")
        b_w = tl.load(w_ptrs, boundary_check=(0,), padding_option="zero")
        b_y = b_x * rsqrt[:, None] * b_w[None, :]
        tl.store(y_ptrs, b_y, boundary_check=(0, 1))

        x_ptrs = tl.advance(x_ptrs, (0, BLOCK_SIZE_D))
        w_ptrs = tl.advance(w_ptrs, (BLOCK_SIZE_D,))
        y_ptrs = tl.advance(y_ptrs, (0, BLOCK_SIZE_D))
    tl.store(rsqrt_ptrs, rsqrt.to(rsqrt_ptrs.dtype.element_ty), boundary_check=(0,))


@triton.jit
def rmsnorm_bwd_kernel(
    dy,
    x,
    w,
    rsqrt,
    dx,
    dw,
    stride_dy_t,
    stride_dy_d,
    stride_x_t,
    stride_x_d,
    stride_w_d,
    stride_rsqrt_t,
    stride_dx_t,
    stride_dx_d,
    stride_dw_d,
    BLOCK_SIZE_T: tl.constexpr,
    T: tl.constexpr,
    BLOCK_SIZE_YD: tl.constexpr,
    BLOCK_SIZE_XD: tl.constexpr,
    D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_xd = tl.program_id(1)
    dy_ptrs = tl.make_block_ptr(
        base=dy,
        shape=(T, D),
        offsets=(pid_t * BLOCK_SIZE_T, 0),
        block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_YD),
        strides=(stride_dy_t, stride_dy_d),
        order=(1, 0),
    )
    x_ptrs = tl.make_block_ptr(
        base=x,
        shape=(T, D),
        offsets=(pid_t * BLOCK_SIZE_T, pid_xd * BLOCK_SIZE_XD),
        block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_XD),
        strides=(stride_x_t, stride_x_d),
        order=(1, 0),
    )
    x_y_ptrs = tl.make_block_ptr(
        base=x,
        shape=(T, D),
        offsets=(pid_t * BLOCK_SIZE_T, 0),
        block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_YD),
        strides=(stride_x_t, stride_x_d),
        order=(1, 0),
    )

    w_ptrs = tl.make_block_ptr(
        base=w,
        shape=(D,),
        offsets=(pid_xd * BLOCK_SIZE_XD,),
        block_shape=(BLOCK_SIZE_XD,),
        strides=(stride_w_d,),
        order=(0,),
    )

    rsqrt_ptrs = tl.make_block_ptr(
        base=rsqrt,
        shape=(T,),
        offsets=(pid_t * BLOCK_SIZE_T,),
        block_shape=(BLOCK_SIZE_T,),
        strides=(stride_rsqrt_t,),
        order=(0,),
    )

    dx_ptrs = tl.make_block_ptr(
        base=dx,
        shape=(T, D),
        offsets=(pid_t * BLOCK_SIZE_T, pid_xd * BLOCK_SIZE_XD),
        block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_XD),
        strides=(stride_dx_t, stride_dx_d),
        order=(1, 0),
    )
    dy_dx_ptrs = tl.make_block_ptr(
        base=dy,
        shape=(T, D),
        offsets=(pid_t * BLOCK_SIZE_T, pid_xd * BLOCK_SIZE_XD),
        block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_XD),
        strides=(stride_dy_t, stride_dy_d),
        order=(1, 0),
    )

    # dw_ptrs = tl.make_block_ptr(
    #     base=dw,
    #     shape=(D,),
    #     offsets=(pid_xd * BLOCK_SIZE_XD,),
    #     block_shape=(BLOCK_SIZE_XD,),
    #     strides=(stride_dw_d,),
    #     order=(0,),
    # )
    dw_ptrs = (
        dw
        + pid_xd * BLOCK_SIZE_XD * stride_dw_d
        + tl.arange(0, BLOCK_SIZE_XD) * stride_dw_d
    )
    dw_mask = (pid_xd * BLOCK_SIZE_XD + tl.arange(0, BLOCK_SIZE_XD)) < D

    b_rsqrt = tl.load(rsqrt_ptrs, boundary_check=(0,), padding_option="zero")

    # for loop over dy
    b_dx = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_XD), dtype=tl.float32)
    b_dw = tl.zeros((BLOCK_SIZE_XD,), dtype=tl.float32)
    b_w = tl.load(w_ptrs, boundary_check=(0,), padding_option="zero")
    b_x = tl.load(x_ptrs, boundary_check=(0, 1), padding_option="zero")
    for i in range(0, D, BLOCK_SIZE_YD):
        b_dy = tl.load(dy_ptrs, boundary_check=(0, 1), padding_option="zero")

        b_dy_w = b_dy * b_w[None, :]
        # [BLOCK_SIZE_YD, BLOCK_SIZE_YD]

        b_x_y = tl.load(x_y_ptrs, boundary_check=(0, 1), padding_option="zero")
        b_dx -= (
            (tl.sum(b_dy_w * b_x_y, axis=1)[:, None] * b_x)
            * (b_rsqrt * b_rsqrt * b_rsqrt)[:, None]
            / D
        )

        dy_ptrs = tl.advance(dy_ptrs, (0, BLOCK_SIZE_YD))
        x_y_ptrs = tl.advance(x_y_ptrs, (0, BLOCK_SIZE_YD))

    b_dy_dx = tl.load(dy_dx_ptrs, boundary_check=(0, 1), padding_option="zero")
    b_dx += b_dy_dx * b_w[None, :] * b_rsqrt[:, None]
    b_dw = tl.sum(b_dy_dx * b_x * b_rsqrt[:, None], axis=0)

    tl.atomic_add(dw_ptrs, b_dw.to(dw_ptrs.dtype.element_ty), mask=dw_mask)
    tl.store(dx_ptrs, b_dx.to(dx_ptrs.dtype.element_ty), boundary_check=(0, 1))


class RMSNormTritonFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Forward pass for RMSNorm using Triton.

        Args:
            ctx: Context object to save tensors for backward pass.
            x (torch.Tensor): Input tensor of shape (..., dim).
            weight (torch.Tensor): Learnable scale parameter of shape (dim,).
            eps (float): Small value for numerical stability.

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Store original shape for output
        orig_shape = x.shape
        dim = x.shape[-1]

        # Reshape x to (-1, dim) if more than 2 dimensions
        if x.dim() > 2:
            x = x.view(-1, dim)
        elif x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure at least 2D

        # Check input validity
        assert (
            x.shape[-1] == weight.shape[0]
        ), f"Expected weight dim {x.shape[-1]}, got {weight.shape[0]}"

        # Allocate output tensor
        T, D = x.shape
        y = torch.empty_like(x)
        rsqrt = torch.zeros((T,), dtype=torch.float32, device=x.device)

        # Triton kernel configuration
        BLOCK_SIZE_D = min(128, triton.next_power_of_2(D))
        BLOCK_SIZE_T = min(128, triton.next_power_of_2(T))
        grid = (triton.cdiv(T, BLOCK_SIZE_T),)

        # Launch kernel
        rmsnorm_fwd_kernel[grid](
            x=x,
            w=weight,
            y=y,
            rsqrt=rsqrt,
            stride_x_t=x.stride(0),
            stride_x_d=x.stride(1),
            stride_w_d=weight.stride(0),
            stride_y_t=y.stride(0),
            stride_y_d=y.stride(1),
            stride_rsqrt_t=rsqrt.stride(0),
            eps=eps,
            BLOCK_SIZE_T=BLOCK_SIZE_T,
            T=T,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            D=D,
        )

        # Reshape output back to original shape
        y = y.view(orig_shape) if orig_shape != y.shape else y

        # Save for backward (if needed)
        ctx.eps = eps
        ctx.save_for_backward(x, weight, y, rsqrt)

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass for RMSNorm using Triton.

        Args:
            ctx: Context object containing saved tensors and eps.
            grad_output (torch.Tensor): Gradient of the loss w.r.t. output.

        Returns:
            tuple: Gradients w.r.t. input (x), weight, and eps (None for eps).
        """
        # Retrieve saved tensors
        x, weight, y, rsqrt = ctx.saved_tensors
        eps = ctx.eps
        orig_shape = grad_output.shape

        # Reshape grad_output to (-1, dim) if more than 2 dimensions
        if grad_output.dim() > 2:
            grad_output = grad_output.view(-1, grad_output.shape[-1])
        elif grad_output.dim() == 1:
            grad_output = grad_output.unsqueeze(0)  # Ensure at least 2D

        # Allocate gradient tensors
        T, D = x.shape
        grad_input = torch.empty_like(x)
        grad_weight = torch.zeros_like(weight)

        # Triton kernel configuration
        BLOCK_SIZE_XD = min(128, triton.next_power_of_2(D))
        BLOCK_SIZE_YD = min(128, triton.next_power_of_2(D))
        BLOCK_SIZE_T = min(16, triton.next_power_of_2(T))
        grid = (triton.cdiv(T, BLOCK_SIZE_T), triton.cdiv(D, BLOCK_SIZE_XD))

        # Launch backward kernel
        rmsnorm_bwd_kernel[grid](
            dy=grad_output,
            x=x,
            w=weight,
            rsqrt=rsqrt,
            dx=grad_input,
            dw=grad_weight,
            stride_dy_t=grad_output.stride(0),
            stride_dy_d=grad_output.stride(1),
            stride_x_t=x.stride(0),
            stride_x_d=x.stride(1),
            stride_w_d=weight.stride(0),
            stride_rsqrt_t=rsqrt.stride(0),
            stride_dx_t=grad_input.stride(0),
            stride_dx_d=grad_input.stride(1),
            stride_dw_d=grad_weight.stride(0),
            BLOCK_SIZE_T=BLOCK_SIZE_T,
            T=T,
            BLOCK_SIZE_YD=BLOCK_SIZE_YD,
            BLOCK_SIZE_XD=BLOCK_SIZE_XD,
            D=D,
        )

        # Reshape grad_input back to original shape
        grad_input = (
            grad_input.view(orig_shape)
            if orig_shape != grad_input.shape
            else grad_input
        )

        return grad_input, grad_weight, None


# RMSNorm module using Triton
class RMSNormTriton(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize RMSNorm layer using Triton kernel.

        Args:
            dim (int): The dimension of the input tensor to normalize (e.g., hidden size).
            eps (float): Small value added for numerical stability. Default: 1e-6.
        """
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (gamma), initialized to ones
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Call the Triton-based RMSNorm function
        return RMSNormTritonFunction.apply(x, self.weight, self.eps)


class RMSNormTorch(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize RMSNorm layer.

        Args:
            dim (int): The dimension of the input tensor to normalize (e.g., hidden size).
            eps (float): Small value added for numerical stability. Default: 1e-6.
        """
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (gamma), initialized to ones
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Compute the mean square: E[x^2]
        mean_square = torch.mean(x**2, dim=-1, keepdim=True)
        # Compute RMS: sqrt(E[x^2] + eps)
        rms = torch.sqrt(mean_square + self.eps)
        # Normalize: x / RMS
        x_norm = x / rms
        # Scale with learnable parameter
        return self.weight * x_norm


def test_rms_norm():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define dimensions and tolerances
    dim = 255
    eps = 1e-1
    rtol = 1e-2
    atol = 1e-2

    # Initialize both RMSNorm modules
    rms_norm_triton = RMSNormTriton(dim=dim, eps=eps).to("cuda")
    rms_norm_ref = RMSNormTorch(dim=dim, eps=eps).to("cuda")

    # Ensure weights are identical
    with torch.no_grad():
        rms_norm_ref.weight.copy_(rms_norm_triton.weight)

    # Test cases with different input shapes
    test_cases = [
        (1, dim),  # 2D: (batch_size, dim)
        (16, 128, dim),  # 3D: (batch_size, seq_len, dim)
        (8, 3, 28, dim),  # 4D: (batch_size, channels, height, dim)
    ]

    for shape in test_cases:
        print(f"\nTesting shape: {shape}")
        # Create input tensor with requires_grad=True
        x = torch.randn(*shape, device="cuda", requires_grad=True)
        x_ref = x.clone().detach().requires_grad_(True)  # Clone for reference

        # Forward pass
        output_triton = rms_norm_triton(x)
        output_ref = rms_norm_ref(x_ref)

        # Compare forward outputs
        torch.testing.assert_close(output_triton, output_ref, rtol=rtol, atol=atol)

        # Create random gradient for backward pass
        grad_output = torch.randn_like(output_triton)
        grad_output_ref = grad_output.clone()

        # Reset gradients for both models
        x.grad = None
        x_ref.grad = None
        rms_norm_triton.zero_grad()
        rms_norm_ref.zero_grad()

        # Backward pass: Triton
        output_triton.backward(grad_output)
        grad_input_triton = x.grad.clone()
        grad_weight_triton = rms_norm_triton.weight.grad.clone()

        # Backward pass: Reference
        output_ref.backward(grad_output_ref)
        grad_input_ref = x_ref.grad.clone()
        grad_weight_ref = rms_norm_ref.weight.grad.clone()

        # Compare gradients
        torch.testing.assert_close(
            grad_input_triton, grad_input_ref, rtol=rtol, atol=atol
        )
        torch.testing.assert_close(
            grad_weight_triton, grad_weight_ref, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    test_rms_norm()
