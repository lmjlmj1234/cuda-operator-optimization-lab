"""LayerNorm correctness verification using PyTorch reference."""
import pytest
import torch
import os
import numpy as np

def load_bin(path, dtype=np.float32):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=dtype)

@pytest.fixture(scope="module")
def ln_bin_dir(bin_dir):
    """Check that layernorm .bin files exist."""
    for f in ["layernorm_cuda_out.bin", "layernorm_welford_out.bin",
              "layernorm_float4_out.bin", "input_data.bin",
              "gamma.bin", "beta.bin"]:
        assert os.path.exists(os.path.join(bin_dir, f)), f"Missing {f}"
    return bin_dir

def test_layernorm_vs_torch(ln_bin_dir, data_dir):
    """Verify layernorm_sum_sq output against PyTorch F.layer_norm."""
    N, D = 4096, 8192
    bin_dir = ln_bin_dir

    x = torch.from_numpy(load_bin(os.path.join(bin_dir, "input_data.bin")).reshape(N, D))
    gamma = torch.from_numpy(load_bin(os.path.join(bin_dir, "gamma.bin")))
    beta = torch.from_numpy(load_bin(os.path.join(bin_dir, "beta.bin")))
    cuda_out = load_bin(os.path.join(bin_dir, "layernorm_cuda_out.bin")).reshape(N, D)

    ref = torch.layer_norm(x, [D], weight=gamma, bias=beta, eps=1e-5)
    cuda_t = torch.from_numpy(cuda_out)
    diff = torch.abs(cuda_t - ref)

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"\nlayernorm_sum_sq vs PyTorch: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

    assert max_diff < 0.05, f"max_diff {max_diff:.2e} exceeds threshold"

def test_layernorm_variants_agree(ln_bin_dir, data_dir):
    """Cross-validate sum_sq, float4, and welford outputs."""
    bin_dir = ln_bin_dir

    sum_sq  = load_bin(os.path.join(bin_dir, "layernorm_cuda_out.bin"))
    welford = load_bin(os.path.join(bin_dir, "layernorm_welford_out.bin"))
    float4  = load_bin(os.path.join(bin_dir, "layernorm_float4_out.bin"))

    for name, out in [("welford", welford), ("float4", float4)]:
        diff = np.abs(sum_sq - out).max()
        assert diff < 0.05, f"layernorm {name} disagrees with sum_sq: max_diff={diff:.2e}"
