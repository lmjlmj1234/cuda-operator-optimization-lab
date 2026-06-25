"""Root conftest: build CUDA binary and run once per session.

Provides shared fixtures (bin_dir, data_dir) for all tests.
"""

import subprocess
import os
import shutil
import pytest

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
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

    return True


def run_binary():
    """Run the CUDA operators binary to produce .bin files."""
    result = subprocess.run(
        [BINARY_PATH],
        capture_output=True, text=True, timeout=120,
        cwd=BUILD_DIR
    )
    if result.returncode != 0:
        pytest.fail(f"CUDA binary failed (rc={result.returncode}):\n{result.stderr}")
    return result.stdout


def pytest_configure(config):
    """Session-level setup: build and run CUDA binary once."""
    if config.option.help:
        return
    if config.option.collectonly:
        return

    build_cuda_project()
    output = run_binary()
    config.cuda_binary_output = output


@pytest.fixture(scope="session")
def bin_dir():
    """Return the build directory path where .bin files are written."""
    return BUILD_DIR


@pytest.fixture(scope="session")
def data_dir():
    """Return the project root (for reference data if needed)."""
    return PROJECT_ROOT
