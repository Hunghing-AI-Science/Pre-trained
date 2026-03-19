import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

BASE_DIR = "/home/auusgx10/PycharmProjects/vllm_testing"
import time
import math
from itertools import cycle
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image



# ============================================================
# CONFIG
# ============================================================
MAX_TOKENS_PER_REQUEST = 8192  # must match your SamplingParams max_tokens

# Set BATCH_SIZE = 0 to auto-calculate from GPU KV cache
BATCH_SIZE = 1

# --- 1. Time model loading ---
t0 = time.perf_counter()

llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor],
    gpu_memory_utilization=0.1
)

t1 = time.perf_counter()
print(f"[Timer] Model loading: {t1 - t0:.3f} seconds")

# ============================================================
# Auto-calculate max batch size from GPU KV cache stats
# ============================================================
# In vLLM v1, configs are stored on vllm_config
vllm_config = llm.llm_engine.vllm_config
scheduler_config = vllm_config.scheduler_config

# Max concurrent sequences = scheduler_config.max_num_seqs
max_seqs = scheduler_config.max_num_seqs
print(f"[Info] max_num_seqs (scheduler limit):  {max_seqs}")
print(f"[Info] Auto-calculated max batch size:  {max_seqs}")

auto_batch_size = max_seqs

# if BATCH_SIZE <= 0:
#     BATCH_SIZE = auto_batch_size
#     print(f"[Info] BATCH_SIZE set automatically to: {BATCH_SIZE}")
# else:
#     if BATCH_SIZE > auto_batch_size:
#         print(f"[Warning] BATCH_SIZE={BATCH_SIZE} exceeds scheduler limit {auto_batch_size}. Clamping.")
#         BATCH_SIZE = auto_batch_size
#     print(f"[Info] BATCH_SIZE (user-set): {BATCH_SIZE}")

# --- 2. Time image loading & input preparation ---
t2 = time.perf_counter()

# ============================================================
# Define each image with its own prompt
# ============================================================
IMAGE_CONFIGS = [
    {
        "path": f"/home/admin/Pre-trained/temp.png",
        "prompt": "<image>\nFree OCR.",
    },
    # {
    #     "path": f"/home/admin/Pre-trained/src/vllm/image_2.png",
    #     "prompt": "<image>\nFree OCR.",
    # },
    # {
    #     "path": f"/home/admin/Pre-trained/src/vllm/img_3.png",
    #     "prompt": "<image>\nFree OCR.",
    # },
    # {
    #     "path": f"/home/admin/Pre-trained/src/vllm/img_4.png",
    #     "prompt": "<image>\nFree OCR.",
    # },
    # {
    #     "path": f"/home/admin/Pre-trained/src/vllm/img_5.png",
    #     "prompt": "<image>\nFree OCR.",
    # },
    # {
    #     "path": f"/home/admin/Pre-trained/src/vllm/img_6.png",
    #     "prompt": "<image>\nFree OCR.",
    # },

]

# Load all images once
loaded_configs = []
for cfg in IMAGE_CONFIGS:
    img = Image.open(cfg["path"]).convert("RGB")
    loaded_configs.append({"image": img, "prompt": cfg["prompt"], "path": cfg["path"]})

print(f"Loaded {len(loaded_configs)} unique images. Building batch of {BATCH_SIZE}...")

# Build model_input up to BATCH_SIZE by cycling through configs
model_input = []
image_labels = []

for cfg in cycle(loaded_configs):
    if len(model_input) >= BATCH_SIZE:
        break
    model_input.append({
        "prompt": cfg["prompt"],
        "multi_modal_data": {"image": cfg["image"]},
    })
    image_labels.append(cfg["path"].split("/")[-1])

sampling_param = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_TOKENS_PER_REQUEST,
    extra_args=dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
    ),
    skip_special_tokens=False,
)

t3 = time.perf_counter()
print(f"[Timer] Input preparation: {t3 - t2:.3f} seconds")
print(f"Batch size: {len(model_input)}")

# --- 3. Time inference ---
t4 = time.perf_counter()

model_outputs = llm.generate(model_input, sampling_param)

t5 = time.perf_counter()
print(f"[Timer] Inference (generate): {t5 - t4:.3f} seconds")

# --- 4. Print outputs and per-request stats ---
for i, output in enumerate(model_outputs):
    text = output.outputs[0].text
    num_tokens = len(output.outputs[0].token_ids)
    print(f"\n--- Output {i + 1} | {image_labels[i]} ---")
    print(f"Generated tokens: {num_tokens}")
    print(f"Tokens/sec (this request): {num_tokens / (t5 - t4):.2f}")
    print(text)

# --- 5. Summary ---
total_tokens = sum(len(o.outputs[0].token_ids) for o in model_outputs)
total_time = t5 - t0
print(f"\n{'=' * 50}")
print(f"[Summary]")
print(f"  Model loading:      {t1 - t0:.3f}s")
print(f"  Input preparation:  {t3 - t2:.3f}s")
print(f"  Inference:          {t5 - t4:.3f}s")
print(f"  Total wall time:    {total_time:.3f}s")
print(f"  Total tokens generated: {total_tokens}")
print(f"  Overall throughput: {total_tokens / (t5 - t4):.2f} tokens/sec")
print(f"  Batch size: {len(model_input)}")