#!/usr/bin/env python3
"""
test_vllm_ocr_service.py
------------------------
Tests for src/vllm/deepseek_ocr_vllm_service.py

Run:
    make test-vllm-ocr-service
    # or
    python test/test_vllm_ocr_service.py
    python test/test_vllm_ocr_service.py --image /path/to/image.png
"""

import os
import sys
import argparse

# ── Project root on path ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Env defaults (override via real .env or shell exports on the GPU server) ─
os.environ.setdefault("VLLM_OCR_MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", "0.85")
os.environ.setdefault("VLLM_OCR_MAX_TOKENS", "8192")
os.environ.setdefault("VLLM_OCR_NGRAM_SIZE", "30")
os.environ.setdefault("VLLM_OCR_WINDOW_SIZE", "90")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def _ok(msg):  print(f"  ✓ {msg}")
def _fail(msg): print(f"  ✗ {msg}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_singleton():
    """get_vllm_ocr_service() must return the same object every time."""
    _sep("Test 1: Singleton pattern")
    from src.vllm.deepseek_ocr_vllm_service import get_vllm_ocr_service

    s1 = get_vllm_ocr_service()
    s2 = get_vllm_ocr_service()
    assert s1 is s2, "Expected the same singleton instance"
    _ok("Singleton pattern works — same instance returned")
    return True


def test_env_config():
    """Service reads config from environment variables."""
    _sep("Test 2: Environment config")
    from src.vllm.deepseek_ocr_vllm_service import DeepSeekOCRVllmService

    # Reset singleton so we can test fresh config pickup
    DeepSeekOCRVllmService._instance = None
    DeepSeekOCRVllmService._model_loaded = False

    os.environ["VLLM_OCR_MAX_TOKENS"] = "4096"
    os.environ["VLLM_OCR_NGRAM_SIZE"] = "15"

    from src.vllm.deepseek_ocr_vllm_service import DeepSeekOCRVllmService as Svc

    # Patch _load_model so we don't actually load the GPU model in unit tests
    original_load = Svc._load_model
    Svc._load_model = lambda self: None

    svc = Svc()
    assert svc.max_tokens == 4096,  f"max_tokens expected 4096, got {svc.max_tokens}"
    assert svc.ngram_size == 15,    f"ngram_size expected 15, got {svc.ngram_size}"
    _ok(f"max_tokens={svc.max_tokens}, ngram_size={svc.ngram_size}")

    # Restore
    Svc._load_model = original_load
    DeepSeekOCRVllmService._instance = None
    DeepSeekOCRVllmService._model_loaded = False
    os.environ["VLLM_OCR_MAX_TOKENS"] = "8192"
    os.environ["VLLM_OCR_NGRAM_SIZE"] = "30"
    return True


def test_perform_ocr(image_path: str, prompt: str = "Free OCR."):
    """Full end-to-end OCR on a real image (requires GPU + model)."""
    _sep(f"Test 3: perform_ocr  [{image_path}]")

    if not os.path.exists(image_path):
        print(f"  ⚠  Image not found: {image_path} — skipping GPU test")
        return True

    from src.vllm.deepseek_ocr_vllm_service import get_vllm_ocr_service

    service = get_vllm_ocr_service()
    _ok("Service loaded")

    result = service.perform_ocr(image_path=image_path, prompt=prompt)

    assert "text" in result,     "Result missing 'text' key"
    assert "metadata" in result, "Result missing 'metadata' key"
    assert "usage" in result,    "Result missing 'usage' key"
    assert result["metadata"]["image_path"] == image_path
    assert result["metadata"]["prompt"] == prompt

    _ok(f"OCR completed — {result['usage'].get('completion_tokens', '?')} tokens generated")
    print(f"\n  ── OCR OUTPUT ──\n{result['text'][:500]}")
    if len(result["text"]) > 500:
        print("  ... (truncated)")
    return True


def test_perform_ocr_batch(image_path: str):
    """Batch OCR with a duplicated request (requires GPU + model)."""
    _sep(f"Test 4: perform_ocr_batch  [{image_path}]")

    if not os.path.exists(image_path):
        print(f"  ⚠  Image not found: {image_path} — skipping GPU test")
        return True

    from src.vllm.deepseek_ocr_vllm_service import get_vllm_ocr_service

    service = get_vllm_ocr_service()

    requests = [
        {"image_path": image_path, "prompt": "Free OCR."},
        {"image_path": image_path, "prompt": "Extract all text."},
    ]
    results = service.perform_ocr_batch(requests)

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    for i, res in enumerate(results):
        assert "text" in res,     f"Result {i} missing 'text'"
        assert "metadata" in res, f"Result {i} missing 'metadata'"
        _ok(f"Result {i+1}: {res['usage'].get('completion_tokens', '?')} tokens")
        print(f"  {res['text'][:200]}")
    return True


def test_result_structure_mock():
    """
    Verify the returned dict structure without loading the GPU model.
    Patches the internal llm.generate() call.
    """
    _sep("Test 5: Result structure (mock inference)")

    from src.vllm.deepseek_ocr_vllm_service import DeepSeekOCRVllmService

    # Reset singleton
    DeepSeekOCRVllmService._instance = None
    DeepSeekOCRVllmService._model_loaded = False

    svc = DeepSeekOCRVllmService.__new__(DeepSeekOCRVllmService)
    svc._initialised = True
    svc.model_name = "deepseek-ai/DeepSeek-OCR"
    svc.max_tokens = 8192
    svc.ngram_size = 30
    svc.window_size = 90
    svc.gpu_memory_utilization = 0.85

    # Mock output object
    class _FakeTokenOut:
        text = "Hello World"
        token_ids = list(range(10))

    class _FakeOutput:
        outputs = [_FakeTokenOut()]

    # Patch llm property
    DeepSeekOCRVllmService._llm = type("FakeLLM", (), {
        "generate": lambda self, inputs, params: [_FakeOutput()] * len(inputs)
    })()
    DeepSeekOCRVllmService._model_loaded = True

    # Also mock _build_sampling_params
    svc._build_sampling_params = lambda: None

    # Temporarily monkey-patch PIL.Image.open
    import unittest.mock as mock
    fake_img = mock.MagicMock()
    fake_img.convert.return_value = fake_img

    with mock.patch("PIL.Image.open", return_value=fake_img):
        result = svc.perform_ocr("/fake/path/image.png", "Free OCR.")

    assert result["text"] == "Hello World",          f"Unexpected text: {result['text']}"
    assert result["metadata"]["image_path"] == "/fake/path/image.png"
    assert result["metadata"]["prompt"] == "Free OCR."
    assert result["usage"]["completion_tokens"] == 10

    _ok("Result structure is correct")

    # Reset
    DeepSeekOCRVllmService._instance = None
    DeepSeekOCRVllmService._model_loaded = False
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test DeepSeekOCRVllmService")
    parser.add_argument(
        "--image",
        default="/home/admin/Pre-trained/src/vllm/1752049668_7801_region_det_res.png",
        help="Path to a real image for GPU inference tests",
    )
    parser.add_argument(
        "--prompt", default="Free OCR.",
        help="OCR prompt for the single-image test",
    )
    parser.add_argument(
        "--skip-gpu", action="store_true",
        help="Skip tests that require a GPU / loaded model",
    )
    args = parser.parse_args()

    results = {}

    print("\n" + "=" * 60)
    print("  DeepSeekOCRVllmService — Test Suite")
    print("=" * 60)

    # Always run (no GPU needed)
    results["singleton"]       = test_singleton()
    results["env_config"]      = test_env_config()
    results["result_structure"] = test_result_structure_mock()

    # GPU tests
    if not args.skip_gpu:
        results["perform_ocr"]       = test_perform_ocr(args.image, args.prompt)
        results["perform_ocr_batch"] = test_perform_ocr_batch(args.image)
    else:
        print("\n  ⚠  GPU tests skipped (--skip-gpu)")

    # Summary
    _sep("Summary")
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
    print(f"\n  {passed}/{total} tests passed")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

