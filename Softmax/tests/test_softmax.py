"""Softmax correctness verification using PyTorch reference."""
import pytest
import torch
import os
import numpy as np

@pytest.fixture(scope="module")
def softmax_bin_dir(bin_dir):
    """Check that softmax .bin files exist."""
    expected_files = [
        "softmax_cuda_out.bin",
        "softmax_naive_out.bin",
        "input_data.bin",
    ]
    for f in expected_files:
        path = os.path.join(bin_dir, f)
        assert os.path.exists(path), f"Missing {path}"
    return bin_dir

def test_softmax_online_vs_torch(softmax_bin_dir, data_dir):
    """Verify online softmax output against PyTorch reference."""
    bin_dir = softmax_bin_dir

    # Load input and output from .bin files
    input_path = os.path.join(bin_dir, "input_data.bin")
    output_path = os.path.join(bin_dir, "softmax_cuda_out.bin")

    with open(input_path, "rb") as f:
        input_data = np.frombuffer(f.read(), dtype=np.float32)
    with open(output_path, "rb") as f:
        cuda_output = np.frombuffer(f.read(), dtype=np.float32)

    N, D = 4096, 8192
    assert len(input_data) == N * D
    assert len(cuda_output) == N * D

    x = torch.from_numpy(input_data.reshape(N, D))
    ref = torch.softmax(x, dim=1)

    cuda_t = torch.from_numpy(cuda_output.reshape(N, D))
    diff = torch.abs(cuda_t - ref)

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nsoftmax_online vs PyTorch: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

    assert max_diff < 1e-3, f"Softmax online max diff {max_diff:.2e} exceeds threshold"
    assert mean_diff < 1e-6, f"Softmax online mean diff {mean_diff:.2e} exceeds threshold"

def test_softmax_naive_vs_torch(softmax_bin_dir, data_dir):
    """Verify naive softmax output against PyTorch reference."""
    bin_dir = softmax_bin_dir

    output_path = os.path.join(bin_dir, "softmax_naive_out.bin")
    input_path = os.path.join(bin_dir, "input_data.bin")

    with open(input_path, "rb") as f:
        input_data = np.frombuffer(f.read(), dtype=np.float32)
    with open(output_path, "rb") as f:
        cuda_output = np.frombuffer(f.read(), dtype=np.float32)

    N, D = 4096, 8192
    x = torch.from_numpy(input_data.reshape(N, D))
    ref = torch.softmax(x, dim=1)

    cuda_t = torch.from_numpy(cuda_output.reshape(N, D))
    diff = torch.abs(cuda_t - ref)

    max_diff = diff.max().item()
    assert max_diff < 1e-3, f"Softmax naive max diff {max_diff:.2e} exceeds threshold"

def test_softmax_variants_agree(softmax_bin_dir, data_dir):
    """Cross-validate naive and online softmax outputs."""
    bin_dir = softmax_bin_dir

    naive_path = os.path.join(bin_dir, "softmax_naive_out.bin")
    online_path = os.path.join(bin_dir, "softmax_cuda_out.bin")

    with open(naive_path, "rb") as f:
        naive = np.frombuffer(f.read(), dtype=np.float32)
    with open(online_path, "rb") as f:
        online = np.frombuffer(f.read(), dtype=np.float32)

    diff = np.abs(naive - online).max()
    print(f"\nNaive vs Online max diff: {diff:.2e}")
    assert diff < 1e-4, f"Softmax variants disagree: max_diff={diff:.2e}"
