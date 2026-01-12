from transformers import AutoModel
import torch
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DeepSeekOCRService:
    """
    Service class for DeepSeek OCR operations
    Singleton pattern - model is loaded once and persists for the lifetime of the process
    """
    _instance: Optional['DeepSeekOCRService'] = None
    _model = None
    _model_loaded = False

    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(DeepSeekOCRService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration from environment variables"""
        self.model_name = os.getenv('OCR_MODEL_NAME', 'deepseek-ai/DeepSeek-OCR')
        self.cache_dir = os.getenv('OCR_MODEL_CACHE_DIR', None)
        self.attention_implementation = os.getenv('OCR_ATTENTION_IMPLEMENTATION', 'flash_attention_2')
        self.use_safetensors = os.getenv('OCR_USE_SAFETENSORS', 'true').lower() == 'true'
        self.torch_dtype = os.getenv('OCR_TORCH_DTYPE', 'bfloat16')
        self.device = os.getenv('OCR_DEVICE', 'cuda')

        self.tokenizer = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Load model immediately if not already loaded
        if not DeepSeekOCRService._model_loaded:
            self._load_model()

    def _load_model(self):
        """Load the model once and keep it in memory"""
        if DeepSeekOCRService._model_loaded:
            logger.info("Model already loaded, reusing existing instance")
            return

        # Set CUDA device
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        logger.info(f"Loading DeepSeek OCR model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Cache directory: {self.cache_dir or 'default'}")


        # Load model
        model_kwargs = {
            '_attn_implementation': self.attention_implementation,
            'trust_remote_code': True,
            'use_safetensors': self.use_safetensors
        }
        if self.cache_dir:
            model_kwargs['cache_dir'] = os.path.expanduser(self.cache_dir)

        DeepSeekOCRService._model = AutoModel.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        # Convert dtype and move to device
        if self.torch_dtype == 'bfloat16':
            DeepSeekOCRService._model = DeepSeekOCRService._model.to(torch.bfloat16)
        elif self.torch_dtype == 'float16':
            DeepSeekOCRService._model = DeepSeekOCRService._model.to(torch.float16)

        if self.device == 'cuda' and torch.cuda.is_available():
            DeepSeekOCRService._model = DeepSeekOCRService._model.cuda()

        DeepSeekOCRService._model = DeepSeekOCRService._model.eval()
        DeepSeekOCRService._model_loaded = True

        logger.info("✓ Model loaded successfully and will persist in memory")
        logger.info(f"✓ Model device: {next(DeepSeekOCRService._model.parameters()).device}")
        logger.info(f"✓ Model dtype: {next(DeepSeekOCRService._model.parameters()).dtype}")

    @property
    def model(self):
        """Get the loaded model"""
        if not DeepSeekOCRService._model_loaded:
            self._load_model()
        return DeepSeekOCRService._model

    def perform_ocr(
        self,
        image_path: str,
        prompt: str = "Free OCR.",
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        save_results: bool = False,
        test_compress: bool = True
    ) -> Dict[str, Any]:
        """
        Perform OCR on an image

        Args:
            image_path: Path to the image file
            prompt: OCR prompt (default: "Free OCR.")
            base_size: Base size for OCR processing
            image_size: Image size for OCR processing
            crop_mode: Whether to use crop mode
            save_results: Whether to save results to file
            test_compress: Whether to test compression

        Returns:
            Dictionary containing OCR results
        """
        # Format prompt with image tag
        formatted_prompt = f"<image>\n{prompt}"

        # Create temporary output directory
        output_path = os.path.join(os.path.dirname(image_path), "ocr_output")
        os.makedirs(output_path, exist_ok=True)

        # Perform inference using the persistent model
        result = self.model.infer(
            self.tokenizer,
            prompt=formatted_prompt,
            image_file=image_path,
            output_path=output_path,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=save_results,
            test_compress=test_compress
        )

        # Parse and format the result
        ocr_text = result if isinstance(result, str) else str(result)

        # Estimate token usage (rough estimation)
        prompt_tokens = len(formatted_prompt.split()) + 100  # Adding image token estimate
        completion_tokens = len(ocr_text.split())

        return {
            "text": ocr_text,
            "metadata": {
                "image_path": image_path,
                "prompt": prompt,
                "base_size": base_size,
                "image_size": image_size,
                "crop_mode": crop_mode
            },
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }


# Global singleton instance
_ocr_service_instance: Optional[DeepSeekOCRService] = None


def get_ocr_service() -> DeepSeekOCRService:
    """
    Get or create the global OCR service instance
    This ensures the model is loaded once and persists throughout the process lifetime
    """
    return None
    global _ocr_service_instance
    if _ocr_service_instance is None:
        _ocr_service_instance = DeepSeekOCRService()
    return _ocr_service_instance


