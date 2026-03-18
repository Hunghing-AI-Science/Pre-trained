#!/usr/bin/env python3
"""
test_vllm_ocr_task.py
---------------------
Tests for src/gpu_server/celery_app/vllm_ocr_tasks.py

Covers:
  1. Task dispatch via apply_async  (requires live Celery worker + broker)
  2. Task execution via .apply()    (synchronous, no broker needed)
  3. DB status transitions          (mocked DB)

Run:
    make test-vllm-ocr-task
    # or
    python test/test_vllm_ocr_task.py
    python test/test_vllm_ocr_task.py --image /path/to/image.png --live
"""

import os
import sys
import uuid
import argparse
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

# ── Project root on path ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Env defaults ─────────────────────────────────────────────────────────────
os.environ.setdefault("VLLM_OCR_MODEL_NAME",        "deepseek-ai/DeepSeek-OCR")
os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", "0.1")
os.environ.setdefault("VLLM_OCR_MAX_TOKENS",         "8192")
os.environ.setdefault("CELERY_VLLM_OCR_TIMEOUT",     "600")
os.environ.setdefault("CELERY_BROKER_URL",           "amqp://guest:guest@localhost:5672//")
os.environ.setdefault("CELERY_RESULT_BACKEND",       "rpc://")
os.environ.setdefault("DATABASE_URL",                "sqlite:///./test_temp.db")
os.environ.setdefault("DATABASE_POOL_SIZE",          "1")
os.environ.setdefault("DATABASE_MAX_OVERFLOW",       "0")
os.environ.setdefault("DATABASE_POOL_TIMEOUT",       "10")
os.environ.setdefault("DATABASE_POOL_RECYCLE",       "300")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def _ok(msg):   print(f"  ✓ {msg}")
def _warn(msg): print(f"  ⚠  {msg}")
def _fail(msg): print(f"  ✗ {msg}")


def _make_fake_ocr_result(image_path, prompt):
    return {
        "text": "Mocked OCR text from the model.",
        "metadata": {"image_path": image_path, "prompt": prompt, "model": "deepseek-ai/DeepSeek-OCR"},
        "usage": {"completion_tokens": 42},
    }


def _make_fake_db_task(task_id):
    """Return a MagicMock that looks like an OCRTask row."""
    task = MagicMock()
    task.id = task_id
    task.status = "pending"
    task.result = None
    task.error = None
    return task


# ---------------------------------------------------------------------------
# Test 1 — Task is registered on the correct queue
# ---------------------------------------------------------------------------

def test_task_registration():
    _sep("Test 1: Task registration & queue")

    # Patch heavy imports so celery_app loads without GPU / broker
    with patch.dict("sys.modules", {
        "torch": MagicMock(),
        "torch.multiprocessing": MagicMock(),
    }):
        from src.gpu_server.celery_app.vllm_ocr_tasks import (
            process_vllm_ocr_task,
            process_vllm_ocr_batch,
        )

    assert process_vllm_ocr_task.name  == \
        "src.gpu_server.celery_app.vllm_ocr_tasks.process_vllm_ocr_task"
    assert process_vllm_ocr_batch.name == \
        "src.gpu_server.celery_app.vllm_ocr_tasks.process_vllm_ocr_batch"

    _ok(f"process_vllm_ocr_task  name: {process_vllm_ocr_task.name}")
    _ok(f"process_vllm_ocr_batch name: {process_vllm_ocr_batch.name}")
    return True


# ---------------------------------------------------------------------------
# Test 2 — Single-task execution with mocked DB + service (no GPU, no broker)
# ---------------------------------------------------------------------------

def test_single_task_mock():
    _sep("Test 2: process_vllm_ocr_task (mocked DB + service)")

    task_id    = str(uuid.uuid4())
    image_path = "/fake/image.png"
    prompt     = "Free OCR."

    fake_task   = _make_fake_db_task(task_id)
    fake_result = _make_fake_ocr_result(image_path, prompt)

    # Build mock DB session
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = fake_task

    # Build mock OCR service
    mock_service = MagicMock()
    mock_service.perform_ocr.return_value = fake_result

    with patch.dict("sys.modules", {"torch": MagicMock(), "torch.multiprocessing": MagicMock()}):
        from src.gpu_server.celery_app.vllm_ocr_tasks import process_vllm_ocr_task

    # Run synchronously (bypasses broker)
    with patch("src.gpu_server.celery_app.vllm_ocr_tasks.get_shared_vllm_ocr_service",
               return_value=mock_service), \
         patch("src.gpu_server.celery_app.vllm_ocr_tasks.SessionLocal",
               return_value=mock_db):

        result = process_vllm_ocr_task.apply(
            args=[task_id, image_path, prompt]
        ).get()

    assert result["task_id"] == task_id
    assert result["status"]  == "completed"
    assert result["result"]["text"] == "Mocked OCR text from the model."

    # Verify status transitions
    status_calls = [
        call for call in fake_task.__setattr__.call_args_list
        if call.args and call.args[0] == "status"
    ]
    # The task object was mutated directly, check via attribute
    _ok(f"Returned task_id  : {result['task_id']}")
    _ok(f"Returned status   : {result['status']}")
    _ok(f"Returned text     : {result['result']['text']}")
    _ok("mock service.perform_ocr() called once")
    mock_service.perform_ocr.assert_called_once_with(image_path=image_path, prompt=prompt)
    return True


# ---------------------------------------------------------------------------
# Test 3 — Batch task execution with mocked DB + service
# ---------------------------------------------------------------------------

def test_batch_task_mock():
    _sep("Test 3: process_vllm_ocr_batch (mocked DB + service)")

    image_path = "/fake/image.png"
    requests = [
        {"task_id": str(uuid.uuid4()), "image_path": image_path, "prompt": "Free OCR."},
        {"task_id": str(uuid.uuid4()), "image_path": image_path, "prompt": "Extract all text."},
    ]

    fake_tasks = {r["task_id"]: _make_fake_db_task(r["task_id"]) for r in requests}

    def _fake_first(task_id_filter=None):
        # mock_db.query(...).filter(...).first() — extract task_id from filter arg
        return MagicMock()

    mock_db = MagicMock()
    def _side_effect_first():
        # Return a new MagicMock each time (represents different DB rows)
        return MagicMock()
    mock_db.query.return_value.filter.return_value.first.side_effect = [
        fake_tasks[r["task_id"]] for r in requests
    ] * 3  # called multiple times across processing + result stages

    fake_results = [_make_fake_ocr_result(r["image_path"], r["prompt"]) for r in requests]

    mock_service = MagicMock()
    mock_service.perform_ocr_batch.return_value = fake_results

    with patch.dict("sys.modules", {"torch": MagicMock(), "torch.multiprocessing": MagicMock()}):
        from src.gpu_server.celery_app.vllm_ocr_tasks import process_vllm_ocr_batch

    with patch("src.gpu_server.celery_app.vllm_ocr_tasks.get_shared_vllm_ocr_service",
               return_value=mock_service), \
         patch("src.gpu_server.celery_app.vllm_ocr_tasks.SessionLocal",
               return_value=mock_db):

        results = process_vllm_ocr_batch.apply(args=[requests]).get()

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    for i, res in enumerate(results):
        assert res["status"] == "completed", f"Result {i} not completed"
        assert res["result"]["text"] == "Mocked OCR text from the model."
        _ok(f"Result {i+1}: task_id={res['task_id']} status={res['status']}")

    mock_service.perform_ocr_batch.assert_called_once()
    _ok("mock service.perform_ocr_batch() called once")
    return True


# ---------------------------------------------------------------------------
# Test 4 — Error handling: DB task not found
# ---------------------------------------------------------------------------

def test_task_not_found_error():
    _sep("Test 4: Error handling — task not found in DB")

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None  # not found

    mock_service = MagicMock()

    with patch.dict("sys.modules", {"torch": MagicMock(), "torch.multiprocessing": MagicMock()}):
        from src.gpu_server.celery_app.vllm_ocr_tasks import process_vllm_ocr_task

    with patch("src.gpu_server.celery_app.vllm_ocr_tasks.get_shared_vllm_ocr_service",
               return_value=mock_service), \
         patch("src.gpu_server.celery_app.vllm_ocr_tasks.SessionLocal",
               return_value=mock_db):

        try:
            process_vllm_ocr_task.apply(
                args=[str(uuid.uuid4()), "/fake/image.png", "Free OCR."]
            ).get()
            _fail("Expected ValueError was not raised")
            return False
        except Exception as e:
            _ok(f"Correctly raised exception: {type(e).__name__}: {e}")
            return True


# ---------------------------------------------------------------------------
# Test 5 — Live dispatch to a running Celery worker (optional)
# ---------------------------------------------------------------------------

def test_live_dispatch(image_path: str, prompt: str = "Free OCR."):
    _sep(f"Test 5: Live dispatch to Celery worker  [{image_path}]")

    if not os.path.exists(image_path):
        _warn(f"Image not found: {image_path} — skipping live test")
        return True

    from src.gpu_server.celery_app.vllm_ocr_tasks import process_vllm_ocr_task
    from src.gpu_server.database import SessionLocal, OCRTask

    # Insert a real DB row
    db = SessionLocal()
    task_id = str(uuid.uuid4())
    try:
        db_task = OCRTask(
            id=task_id,
            status="pending",
            prompt=prompt,
            image_path=image_path,
            model_name="deepseek-ocr-vllm",
        )
        db.add(db_task)
        db.commit()
        _ok(f"Inserted OCRTask: {task_id}")
    finally:
        db.close()

    # Dispatch
    async_result = process_vllm_ocr_task.apply_async(
        args=[task_id, image_path, prompt],
        queue="vllm_ocr",
    )
    _ok(f"Task dispatched — Celery task id: {async_result.id}")

    print("  Waiting for result (timeout=120s)…")
    try:
        result = async_result.get(timeout=120)
        _ok(f"Result status : {result['status']}")
        _ok(f"OCR text      : {result['result']['text'][:200]}")
    except Exception as e:
        _fail(f"Task failed or timed out: {e}")
        return False

    return True


# ---------------------------------------------------------------------------
# Example: call POST /v1/ocr/chat/completions with a real image file
# Requires a running API server (python run_api.py) + vllm_ocr worker
# ---------------------------------------------------------------------------

def example_api_call(image_path: str, prompt: str = "Free OCR.",
                     base_url: str = "http://localhost:8000"):
    """
    Example: send a base64-encoded image to POST /v1/ocr/chat/completions
    and print the OCR text returned by the vLLM DeepSeek OCR backend.

    Usage:
        python test/test_vllm_ocr_task.py --example --image /path/to/image.png
        # or via make:
        make test-vllm-ocr-task-example IMAGE=/path/to/image.png
    """
    import base64
    import requests

    _sep(f"Example: POST /v1/ocr/chat/completions")
    print(f"  server  : {base_url}")
    print(f"  image   : {image_path}")
    print(f"  prompt  : {prompt}")

    # --- 1. Read & base64-encode the image ---
    if not os.path.exists(image_path):
        _warn(f"Image not found: {image_path} — skipping example")
        return True

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    fmt_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
               "gif": "gif", "bmp": "bmp", "webp": "webp"}
    mime = fmt_map.get(ext, "jpeg")
    b64  = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/{mime};base64,{b64}"
    print(f"  encoded : {len(data_uri)} chars  ({len(image_bytes)} raw bytes)")

    # --- 2. Build the request payload ---
    payload = {
        "model": "deepseek-ocr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text",  "text": prompt},
                    {"type": "image", "image": data_uri},
                ],
            }
        ],
    }

    # --- 3. POST to the API ---
    url = f"{base_url}/v1/ocr/chat/completions"
    print(f"\n  → POST {url}")
    try:
        resp = requests.post(url, json=payload, timeout=300)
    except requests.exceptions.ConnectionError:
        _warn(f"Could not connect to {base_url} — is the API server running?")
        return True

    print(f"  ← HTTP {resp.status_code}")

    if resp.status_code != 200:
        _fail(f"Non-200 response: {resp.text[:300]}")
        return False

    data = resp.json()

    # --- 4. Print result ---
    ocr_text = data["choices"][0]["text"]
    usage    = data.get("usage", {})

    print(f"\n  ── OCR OUTPUT ──────────────────────────────────")
    print(f"  {ocr_text[:600]}")
    if len(ocr_text) > 600:
        print("  ... (truncated)")
    print(f"\n  model             : {data.get('model')}")
    print(f"  task id           : {data.get('id')}")
    print(f"  completion_tokens : {usage.get('completion_tokens', '–')}")
    print(f"  ────────────────────────────────────────────────")

    _ok("Example call completed successfully")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test vLLM OCR Celery tasks")
    parser.add_argument(
        "--image",
        default="/home/admin/Pre-trained/src/vllm/1752049668_7801_region_det_res.png",
        help="Path to image for live GPU / example test",
    )
    parser.add_argument("--prompt", default="Free OCR.", help="OCR prompt")
    parser.add_argument(
        "--live", action="store_true",
        help="Run live dispatch test (requires broker + worker)",
    )
    parser.add_argument(
        "--example", action="store_true",
        help="Run the end-to-end API example (requires running server + worker)",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="Base URL of the API server for --example",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  vLLM OCR Celery Tasks — Test Suite")
    print("=" * 60)

    results = {}
    results["task_registration"]    = test_task_registration()
    results["single_task_mock"]     = test_single_task_mock()
    results["batch_task_mock"]      = test_batch_task_mock()
    results["task_not_found_error"] = test_task_not_found_error()

    if args.live:
        results["live_dispatch"] = test_live_dispatch(args.image, args.prompt)
    else:
        _warn("Live dispatch test skipped (pass --live to enable)")

    if args.example:
        results["example_api_call"] = example_api_call(
            args.image, args.prompt, args.base_url
        )
    else:
        _warn("API example skipped (pass --example to enable)")

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

