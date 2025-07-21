import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_dim, moe_inner_dim, dtype):
        super().__init__()
        self.fc11 = nn.Linear(hidden_dim, moe_inner_dim, bias=False, dtype=dtype)
        self.fc12 = nn.Linear(hidden_dim, moe_inner_dim, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(moe_inner_dim, hidden_dim, bias=False, dtype=dtype)

    def forward(self, x):
        # Split into gate and value
        gate = self.fc11(x)
        value = self.fc12(x)

        # Swish activation: x * sigmoid(beta * x)
        gate = gate * torch.sigmoid(gate)

        fc1_output = gate * value
        # fc1_output = value

        fc2_output = self.fc2(fc1_output)

        return fc2_output


class SwiGLUMoETorch(nn.Module):
    def __init__(
        self, hidden_dim, moe_inner_dim, num_experts, top_k=2, dtype=torch.bfloat16
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Gating network
        self.moe_gate = nn.Linear(hidden_dim, num_experts, bias=False, dtype=dtype)

        # Experts
        self.experts = nn.ModuleList(
            [SwiGLUExpert(hidden_dim, moe_inner_dim, dtype) for _ in range(num_experts)]
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Compute gating scores
        gate_scores = F.softmax(self.moe_gate(x), dim=-1)  # [batch, seq, num_experts]

        # Select top-k experts
        top_k_scores, top_k_indices = gate_scores.topk(
            self.top_k, dim=-1
        )  # [batch, seq, top_k]

        # Normalize top-k scores
        top_k_scores = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # Initialize output
        output = torch.zeros_like(x)

        # Process each expert
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]  # [batch, seq]
            expert_score = top_k_scores[:, :, i].unsqueeze(-1)  # [batch, seq, 1]

            # Gather expert outputs
            expert_output = torch.zeros_like(x)
            for j in range(self.num_experts):
                mask = (expert_idx == j).float().unsqueeze(-1)  # [batch, seq, 1]
                if mask.sum() > 0:
                    # Apply expert only to relevant tokens
                    expert_output += mask * self.experts[j](x)

            output += expert_score * expert_output
            # output += expert_output

        return output


# Example usage
if __name__ == "__main__":
    # Parameters
    hidden_dim = 1024
    moe_inner_dim = 512
    num_experts = 8
    batch_size = 1
    seq_len = 128

    # Create model
    model = SwiGLUMoETorch(hidden_dim, moe_inner_dim, num_experts)

    # Sample input
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
