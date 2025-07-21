# TritonOps

A collection of optimized operations implemented using Triton for PyTorch, focusing on performance-critical kernels like attention mechanisms, normalizations, and more.

## Overview

This repository provides Triton-based implementations for various deep learning operations, along with tutorials to get started with Triton kernel development.

Key components:
- **common**: Shared utilities (e.g., RMS norm).
- **cross_entropy_triton**: Triton-accelerated cross-entropy computations.
- **flash_attn_triton**: Efficient attention mechanisms, including flash attention and sparse variants.
- **moe**: Mixture of Experts (MoE) implementations (Torch and Triton versions). *Note: This module is still under development and not 100% finished.*
- **tutorial**: Step-by-step examples for writing Triton kernels, covering basics like softmax, matmul, layer norms, and attention.

All other modules are ready for use.

## Requirements

Install dependencies via:
```
pip install -r requirements.txt
```

Requires PyTorch with Triton support (typically installed via PyTorch nightly or compatible versions).

## Usage

Import and use the modules directly in your PyTorch code. For example:
```python
from flash_attn_triton.flash_attn_triton import flash_attention
# Use as needed
```

Explore the `tutorial` directory for hands-on examples to learn Triton kernel authoring.

Contributions welcome! If you encounter issues, open a GitHub issue.