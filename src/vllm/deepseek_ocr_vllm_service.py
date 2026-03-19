import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

BASE_DIR = "/home/auusgx10/PycharmProjects/vllm_testing"
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class DeepSeekOCRVllmService:
    """
    Service class for DeepSeek OCR operations using vLLM backend.
    Singleton pattern — model is loaded once and persists for the lifetime of the process.
    """

    _instance: Optional["DeepSeekOCRVllmService"] = None
    _llm = None
    _model_loaded: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Avoid re-initialising config on every __init__ call (singleton)
        if hasattr(self, "_initialised"):
            return
        self._initialised = True

        # ---- Config from environment ----
        self.model_name = os.getenv("VLLM_OCR_MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
        self.gpu_memory_utilization = 0.1
        self.max_tokens = int(os.getenv("VLLM_OCR_MAX_TOKENS", "8192"))
        self.ngram_size = int(os.getenv("VLLM_OCR_NGRAM_SIZE", "30"))
        self.window_size = int(os.getenv("VLLM_OCR_WINDOW_SIZE", "90"))

        # Mirror endpoint (optional)
        hf_endpoint = os.getenv("HF_ENDPOINT")
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint

        if not DeepSeekOCRVllmService._model_loaded:
            self._load_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the vLLM LLM instance (once per process)."""
        if DeepSeekOCRVllmService._model_loaded:
            logger.info("vLLM OCR model already loaded, reusing existing instance.")
            return

        logger.info(f"Loading vLLM DeepSeek OCR model: {self.model_name}")

        from vllm import LLM
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

        DeepSeekOCRVllmService._llm = LLM(
            model=self.model_name,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor],
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        DeepSeekOCRVllmService._model_loaded = True
        logger.info("✓ vLLM OCR model loaded successfully.")

    @property
    def llm(self):
        if not DeepSeekOCRVllmService._model_loaded:
            self._load_model()
        return DeepSeekOCRVllmService._llm

    def _build_sampling_params(self):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=0.0,
            max_tokens=self.max_tokens,
            extra_args=dict(
                ngram_size=self.ngram_size,
                window_size=self.window_size,
                whitelist_token_ids={128821, 128822},
            ),
            skip_special_tokens=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def perform_ocr(
        self,
        image_path: str,
        prompt: str = "Free OCR.",
    ) -> Dict[str, Any]:
        """
        Perform OCR on a single image using the vLLM backend.

        Args:
            image_path: Absolute path to the image file.
            prompt:     OCR instruction prompt.

        Returns:
            dict with keys: text, metadata, usage
        """
        from PIL import Image

        logger.info(f"Loading image: {image_path}")
        image = Image.open(image_path).convert("RGB")

        formatted_prompt = f"<image>\n{prompt}"
        model_input = [
            {
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": image},
            }
        ]

        sampling_params = self._build_sampling_params()
        logger.info("Running vLLM inference...")
        outputs = self.llm.generate(model_input, sampling_params)

        output = outputs[0].outputs[0]
        ocr_text = output.text
        completion_tokens = len(output.token_ids)

        logger.info(f"vLLM OCR completed. Generated {completion_tokens} tokens.")

        return {
            "text": ocr_text,
            "metadata": {
                "image_path": image_path,
                "prompt": prompt,
                "model": self.model_name,
            },
            "usage": {
                "completion_tokens": completion_tokens,
            },
        }

    def perform_ocr_batch(
        self,
        requests: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Perform OCR on a batch of images.

        Args:
            requests: List of dicts, each with keys:
                      - image_path (str)
                      - prompt     (str, optional)

        Returns:
            List of result dicts (same format as perform_ocr).
        """
        from PIL import Image

        model_inputs = []
        for req in requests:
            image = Image.open(req["image_path"]).convert("RGB")
            formatted_prompt = f"<image>\n{req.get('prompt', 'Free OCR.')}"
            model_inputs.append(
                {
                    "prompt": formatted_prompt,
                    "multi_modal_data": {"image": image},
                }
            )

        sampling_params = self._build_sampling_params()
        logger.info(f"Running vLLM batch inference ({len(model_inputs)} images)...")
        outputs = self.llm.generate(model_inputs, sampling_params)

        results = []
        for req, output in zip(requests, outputs):
            out = output.outputs[0]
            results.append(
                {
                    "text": out.text,
                    "metadata": {
                        "image_path": req["image_path"],
                        "prompt": req.get("prompt", "Free OCR."),
                        "model": self.model_name,
                    },
                    "usage": {
                        "completion_tokens": len(out.token_ids),
                    },
                }
            )

        logger.info("vLLM batch OCR completed.")
        return results


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_vllm_ocr_service_instance: Optional[DeepSeekOCRVllmService] = None


def get_vllm_ocr_service() -> DeepSeekOCRVllmService:
    """Return (or create) the global vLLM OCR service singleton."""
    global _vllm_ocr_service_instance
    if _vllm_ocr_service_instance is None:
        _vllm_ocr_service_instance = DeepSeekOCRVllmService()
    return _vllm_ocr_service_instance

