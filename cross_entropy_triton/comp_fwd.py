import torch
import torch.nn as nn
import triton
import triton.testing

from cross_entropy_triton.cross_entropy_old import cross_entropy_fused_triton_old
from cross_entropy_triton.cross_entropy import cross_entropy_fused_triton


def test_cross_entropy(N, hidden_size, vocab_size, provider):
    torch.manual_seed(42)

    # Test parameters
    batch_size = 1
    seq_length = N
    device = "cuda"
    dtype = torch.bfloat16

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

    # Clone inputs
    hidden_states_clone = hidden_states.clone().detach().requires_grad_(True)
    lm_head_weight_clone = lm_head_weight.clone().detach().requires_grad_(True)
    labels_clone = labels.clone().detach()

    quantiles = [0.5, 0.2, 0.8]

    if provider == "pytorch":
        standard_ce = nn.CrossEntropyLoss(reduction="none")

        def run_standard():
            logits = hidden_states @ lm_head_weight.T
            return standard_ce(logits.view(-1, vocab_size), labels.view(-1))

        ms, min_ms, max_ms = triton.testing.do_bench(run_standard, quantiles=quantiles)

    elif provider == "triton":

        def run_triton():
            return cross_entropy_fused_triton_old(
                hidden_states_clone.view(-1, hidden_states_clone.size(-1)),
                lm_head_weight_clone,
                labels_clone.view(-1, 1),
            )

        ms, min_ms, max_ms = triton.testing.do_bench(run_triton, quantiles=quantiles)

    elif provider == "triton_acc":

        def run_triton_acc():
            return cross_entropy_fused_triton(
                hidden_states_clone.view(-1, hidden_states_clone.size(-1)),
                lm_head_weight_clone,
                labels_clone.view(-1, 1),
            )

        ms, min_ms, max_ms = triton.testing.do_bench(
            run_triton_acc, quantiles=quantiles
        )

    return ms, min_ms, max_ms


if __name__ == "__main__":
    # Benchmark for hidden_size=2048
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(5, 14)],
            line_arg="provider",
            line_vals=["pytorch", "triton", "triton_acc"],
            line_names=[
                "PyTorch",
                "Triton (fused_triton)",
                "Triton (fused_triton_acc)",
            ],
            styles=[("blue", "-"), ("green", "--"), ("red", "--")],
            ylabel="ms",
            plot_name="Cross Entropy Forward (hidden_size=2048)",
            args={"hidden_size": 2048, "vocab_size": 155136},
        )
    )
    def benchmark_2048(N, hidden_size, vocab_size, provider):
        return test_cross_entropy(N, hidden_size, vocab_size, provider)

    # Benchmark for hidden_size=1024
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(5, 14)],
            line_arg="provider",
            line_vals=["pytorch", "triton", "triton_acc"],
            line_names=[
                "PyTorch",
                "Triton (fused_triton)",
                "Triton (fused_triton_acc)",
            ],
            styles=[("blue", "-"), ("green", "--"), ("red", "--")],
            ylabel="ms",
            plot_name="Cross Entropy Forward (hidden_size=1024)",
            args={"hidden_size": 1024, "vocab_size": 155136},
        )
    )
    def benchmark_1024(N, hidden_size, vocab_size, provider):
        return test_cross_entropy(N, hidden_size, vocab_size, provider)

    # Run benchmarks
    print("Running benchmark for hidden_size=2048...")
    benchmark_2048.run(show_plots=True, print_data=True)

    print("\nRunning benchmark for hidden_size=1024...")
    benchmark_1024.run(show_plots=True, print_data=True)
