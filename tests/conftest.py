"""pytest configuration: build CUDA binary and run once per session."""

import subprocess
import os
import shutil
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
BINARY_PATH = os.path.join(BUILD_DIR, "cuda_operators")


def build_cuda_project():
    """Build the CUDA project with CMake."""
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR, exist_ok=True)

    result = subprocess.run(
        ["cmake", PROJECT_ROOT],
        cwd=BUILD_DIR,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.fail(f"CMake configuration failed:\n{result.stderr}")

    result = subprocess.run(
        ["cmake", "--build", "."],
        cwd=BUILD_DIR,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.fail(f"CMake build failed:\n{result.stderr}")

    if not os.path.exists(BINARY_PATH):
        pytest.fail(f"Binary not found after build: {BINARY_PATH}")

    return binary_exists()


def binary_exists():
    """Check if the CUDA binary already exists."""
    return os.path.isfile(BINARY_PATH) and os.access(BINARY_PATH, os.X_OK)


def run_binary():
    """Run the CUDA operators binary to produce .bin files."""
    result = subprocess.run(
        [BINARY_PATH],
        capture_output=True, text=True, timeout=120,
        cwd=BUILD_DIR  # dump_gpu_buffer writes .bin files relative to CWD
    )
    if result.returncode != 0:
        pytest.fail(f"CUDA binary failed (rc={result.returncode}):\n{result.stderr}")
    return result.stdout


def pytest_configure(config):
    """Session-level setup: build and run CUDA binary once."""
    # Prevent this from running during collection/--help
    if config.option.help:
        return
    if config.option.collectonly:
        return

    build_cuda_project()
    output = run_binary()

    # Store output for potential use in tests
    config.cuda_binary_output = output


@pytest.fixture(scope="session")
def bin_dir():
    """Return the build directory path where .bin files are written.

    The binary's working directory is the build dir, so .bin files
    land there.
    """
    return BUILD_DIR


@pytest.fixture(scope="session")
def data_dir():
    """Return the project root (for reference data if needed)."""
    return PROJECT_ROOT
