"""pytest test suite for CUDA operator correctness.

Flow:
  1. conftest.py builds the CUDA project and runs the binary (once per session)
  2. The binary writes .bin files containing GPU kernel outputs
  3. Each test_* function reads .bin files, computes PyTorch reference, asserts closeness

Run:  pytest tests/ -v
"""

import os

import numpy as np
import torch

N, D = 4096, 8192  # must match main.cu


def read_float_bin(filepath, count):
    """Read a binary file of float32 values into a numpy array."""
    with open(filepath, "rb") as f:
        raw = f.read(count * 4)
    return np.frombuffer(raw, dtype=np.float32)


def read_inputs(bin_dir):
    """Read input .bin files and reshape to (N, D) tensors."""
    base = os.path.join(bin_dir, "input_data.bin")
    inp = torch.from_numpy(read_float_bin(base, N * D).copy()).view(N, D)
    gamma = torch.from_numpy(read_float_bin(os.path.join(bin_dir, "gamma.bin"), D).copy())
    beta = torch.from_numpy(read_float_bin(os.path.join(bin_dir, "beta.bin"), D).copy())
    residual = torch.from_numpy(read_float_bin(os.path.join(bin_dir, "residual.bin"), N * D).copy()).view(N, D)
    return inp, gamma, beta, residual


def read_cuda_out(bin_dir, name):
    """Read a CUDA kernel output .bin file as (N, D) tensor."""
    path = os.path.join(bin_dir, name)
    return torch.from_numpy(read_float_bin(path, N * D).copy()).view(N, D)


# ============================================================
# Tests
# ============================================================

class TestSoftmax:
    """Softmax: CUDA online kernel vs torch.softmax."""

    def test_online_vs_pytorch(self, bin_dir):
        inp, *_ = read_inputs(bin_dir)
        cuda_out = read_cuda_out(bin_dir, "softmax_cuda_out.bin")
        ref = torch.softmax(inp, dim=1)
        diff = (cuda_out - ref).abs().max().item()
        assert diff < 0.001, \
            f"Softmax CUDA vs PyTorch max diff = {diff:.2e} (threshold 1e-3)"

    def test_online_vs_naive(self, bin_dir):
        """Online softmax and naive softmax should produce nearly identical results."""
        online = read_cuda_out(bin_dir, "softmax_cuda_out.bin")
        naive = read_cuda_out(bin_dir, "softmax_naive_out.bin")
        diff = (online - naive).abs().max().item()
        assert diff < 1e-4, \
            f"Online vs Naive Softmax max diff = {diff:.2e} (threshold 1e-4)"

    def test_sum_close_to_one(self, bin_dir):
        """For each row, softmax output should sum to 1.0."""
        cuda_out = read_cuda_out(bin_dir, "softmax_cuda_out.bin")
        row_sums = cuda_out.sum(dim=1)
        max_err = (row_sums - 1.0).abs().max().item()
        assert max_err < 1e-5, \
            f"Softmax row sum max error = {max_err:.2e} (expected ~1.0)"


class TestLayerNorm:
    """LayerNorm: CUDA sum_sq kernel vs torch.layer_norm."""

    def test_layernorm(self, bin_dir):
        inp, gamma, beta, *_ = read_inputs(bin_dir)
        cuda_out = read_cuda_out(bin_dir, "layernorm_cuda_out.bin")
        ref = torch.layer_norm(inp, (D,), weight=gamma, bias=beta, eps=1e-5)
        diff = (cuda_out - ref).abs().max().item()

        # float32 precision: 8192-element reduction accumulates ~5% error
        assert diff < 0.05, \
            f"LayerNorm CUDA vs PyTorch max diff = {diff:.2e} (threshold 0.05)"

    def test_float4_vs_scalar(self, bin_dir):
        """Float4 LayerNorm should closely match scalar LayerNorm.

        Different memory access patterns produce slightly different
        accumulation order, so tolerance accounts for float32 rounding.
        """
        float4_out = read_cuda_out(bin_dir, "layernorm_float4_out.bin")
        scalar_out = read_cuda_out(bin_dir, "layernorm_cuda_out.bin")
        diff = (float4_out - scalar_out).abs().max().item()
        assert diff < 0.05, \
            f"Float4 vs scalar LayerNorm max diff = {diff:.2e} (threshold 0.05)"

    def test_welford_vs_sum_sq(self, bin_dir):
        """Welford LayerNorm should closely match sum_sq LayerNorm."""
        welford = read_cuda_out(bin_dir, "layernorm_welford_out.bin")
        sum_sq = read_cuda_out(bin_dir, "layernorm_cuda_out.bin")
        diff = (welford - sum_sq).abs().max().item()
        assert not torch.isnan(welford).any(), "Welford output contains NaN!"
        assert not torch.isnan(sum_sq).any(), "SumSq output contains NaN!"
        assert diff < 0.05, \
            f"Welford vs SumSq LayerNorm max diff = {diff:.2e} (threshold 0.05)"


class TestRMSNorm:
    """RMSNorm: CUDA kernel vs manual reference."""

    def test_rmsnorm(self, bin_dir):
        inp, gamma, *_ = read_inputs(bin_dir)
        cuda_out = read_cuda_out(bin_dir, "rmsnorm_cuda_out.bin")
        rms = torch.sqrt((inp ** 2).mean(dim=1, keepdim=True) + 1e-5)
        ref = inp / rms * gamma
        diff = (cuda_out - ref).abs().max().item()
        assert diff < 1e-4, \
            f"RMSNorm CUDA vs PyTorch max diff = {diff:.2e} (threshold 1e-4)"

    def test_float4_vs_scalar(self, bin_dir):
        """Float4 RMSNorm should match scalar RMSNorm."""
        float4_out = read_cuda_out(bin_dir, "rmsnorm_float4_out.bin")
        scalar_out = read_cuda_out(bin_dir, "rmsnorm_cuda_out.bin")
        diff = (float4_out - scalar_out).abs().max().item()
        assert diff < 1e-5, \
            f"Float4 vs scalar RMSNorm max diff = {diff:.2e} (threshold 1e-5)"


class TestFusedLayerNorm:
    """Fused Residual + LayerNorm: CUDA vs sequential residual add + torch.layer_norm."""

    def test_fused_residual_layernorm(self, bin_dir):
        inp, gamma, beta, residual = read_inputs(bin_dir)
        cuda_out = read_cuda_out(bin_dir, "fused_layernorm_cuda_out.bin")
        ref = torch.layer_norm(inp + residual, (D,), weight=gamma, bias=beta, eps=1e-5)
        diff = (cuda_out - ref).abs().max().item()
        assert diff < 0.05, \
            f"Fused LayerNorm CUDA vs PyTorch max diff = {diff:.2e} (threshold 0.05)"

    def test_float4_vs_scalar(self, bin_dir):
        """Float4 Fused LayerNorm should closely match scalar."""
        float4_out = read_cuda_out(bin_dir, "fused_ln_float4_out.bin")
        scalar_out = read_cuda_out(bin_dir, "fused_layernorm_cuda_out.bin")
        diff = (float4_out - scalar_out).abs().max().item()
        assert diff < 0.25, \
            f"Float4 vs scalar Fused LN max diff = {diff:.2e} (threshold 0.25)"


class TestSoftmaxFloat4:
    """Softmax Float4 variant vs naive baseline."""

    def test_vs_naive(self, bin_dir):
        online_f4 = read_cuda_out(bin_dir, "softmax_cuda_out.bin")
        naive = read_cuda_out(bin_dir, "softmax_naive_out.bin")
        diff = (online_f4 - naive).abs().max().item()
        assert diff < 1e-4, \
            f"Online+Float4 vs Naive Softmax max diff = {diff:.2e} (threshold 1e-4)"
