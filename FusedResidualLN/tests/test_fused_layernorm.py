"""Fused Residual + LayerNorm correctness verification using PyTorch."""
import pytest
import torch
import os
import numpy as np

def load_bin(path, dtype=np.float32):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=dtype)

@pytest.fixture(scope="module")
def fused_bin_dir(bin_dir):
    for f in ["fused_layernorm_cuda_out.bin", "fused_ln_float4_out.bin",
              "input_data.bin", "residual.bin", "gamma.bin", "beta.bin"]:
        assert os.path.exists(os.path.join(bin_dir, f)), f"Missing {f}"
    return bin_dir

def test_fused_layernorm_vs_torch(fused_bin_dir, data_dir):
    """Verify fused kernel against PyTorch: LN(x + residual)."""
    N, D = 4096, 8192
    bin_dir = fused_bin_dir

    x = torch.from_numpy(load_bin(os.path.join(bin_dir, "input_data.bin")).reshape(N, D))
    res = torch.from_numpy(load_bin(os.path.join(bin_dir, "residual.bin")).reshape(N, D))
    gamma = torch.from_numpy(load_bin(os.path.join(bin_dir, "gamma.bin")))
    beta = torch.from_numpy(load_bin(os.path.join(bin_dir, "beta.bin")))

    fused_out = load_bin(os.path.join(bin_dir, "fused_layernorm_cuda_out.bin")).reshape(N, D)

    # Ref: layer_norm(x + residual)
    y = x + res
    ref = torch.layer_norm(y, [D], weight=gamma, bias=beta, eps=1e-5)

    cuda_t = torch.from_numpy(fused_out)
    diff = torch.abs(cuda_t - ref)

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"\nfused_residual_layernorm vs PyTorch: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

    assert max_diff < 1e-3, f"max_diff {max_diff:.2e} exceeds threshold"

def test_fused_variants_agree(fused_bin_dir, data_dir):
    """Cross-validate scalar and float4 fused outputs."""
    bin_dir = fused_bin_dir

    scalar = load_bin(os.path.join(bin_dir, "fused_layernorm_cuda_out.bin"))
    float4 = load_bin(os.path.join(bin_dir, "fused_ln_float4_out.bin"))

    diff = np.abs(scalar - float4).max()
    print(f"\nScalar vs Float4 fused LN max diff: {diff:.2e}")
    assert diff < 1e-4, f"Fused LN variants disagree: max_diff={diff:.2e}"
