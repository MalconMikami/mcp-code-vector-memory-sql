import os

import pytest


def test_gpu_offload_support_is_present_when_required(tmp_path, monkeypatch):
    """
    This is a validation test, not a benchmark.

    It only runs when explicitly enabled, because many environments do not have a GPU-enabled
    llama-cpp-python build.
    """

    if os.getenv("REQUIRE_GPU_OFFLOAD", "0").lower() not in ("1", "true", "yes"):
        pytest.skip("Set REQUIRE_GPU_OFFLOAD=1 to require GPU-offload support.")

    monkeypatch.setenv("CODE_MEMORY_AUTO_INSTALL", "0")
    monkeypatch.setenv("CODE_MEMORY_SUMMARY_AUTO_DOWNLOAD", "0")
    monkeypatch.setenv("CODE_MEMORY_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("CODE_MEMORY_CONFIG_PATH", str(tmp_path / "no-config.jsonc"))

    from code_memory import summary

    assert summary.ensure_llama_cpp() is True
    assert summary.gpu_offload_supported() is True, f"GPU offload not supported.\n\n{summary.llama_system_info()}"

