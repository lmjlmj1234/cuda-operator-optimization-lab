"""RMSNorm correctness verification using PyTorch reference."""
import pytest
import torch
import os
import numpy as np

def load_bin(path, dtype=np.float32):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=dtype)

@pytest.fixture(scope="module")
def rmsnorm_bin_dir(bin_dir):
    for f in ["rmsnorm_cuda_out.bin", "rmsnorm_float4_out.bin",
              "input_data.bin", "gamma.bin"]:
        assert os.path.exists(os.path.join(bin_dir, f)), f"Missing {f}"
    return bin_dir

def torch_rmsnorm(x, gamma, eps=1e-5):
    """PyTorch reference for RMSNorm."""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * gamma

def test_rmsnorm_vs_torch(rmsnorm_bin_dir, data_dir):
    """Verify rmsnorm_kernel output against PyTorch reference."""
    N, D = 4096, 8192
    bin_dir = rmsnorm_bin_dir

    x = torch.from_numpy(load_bin(os.path.join(bin_dir, "input_data.bin")).reshape(N, D))
    gamma = torch.from_numpy(load_bin(os.path.join(bin_dir, "gamma.bin")))
    cuda_out = load_bin(os.path.join(bin_dir, "rmsnorm_cuda_out.bin")).reshape(N, D)

    ref = torch_rmsnorm(x, gamma)
    cuda_t = torch.from_numpy(cuda_out)
    diff = torch.abs(cuda_t - ref)

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"\nrmsnorm_kernel vs PyTorch: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

    assert max_diff < 1e-3, f"max_diff {max_diff:.2e} exceeds threshold"
    assert mean_diff < 1e-6, f"mean_diff {mean_diff:.2e} exceeds threshold"

def test_rmsnorm_variants_agree(rmsnorm_bin_dir, data_dir):
    """Cross-validate scalar and float4 rmsnorm outputs."""
    bin_dir = rmsnorm_bin_dir

    scalar = load_bin(os.path.join(bin_dir, "rmsnorm_cuda_out.bin"))
    float4 = load_bin(os.path.join(bin_dir, "rmsnorm_float4_out.bin"))

    diff = np.abs(scalar - float4).max()
    print(f"\nScalar vs Float4 RMSNorm max diff: {diff:.2e}")
    assert diff < 1e-4, f"RMSNorm variants disagree: max_diff={diff:.2e}"
