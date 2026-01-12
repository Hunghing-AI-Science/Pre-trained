from transformers import pipeline
import torch
import os
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class GPTOSSService:
    """
    Service class for GPT OSS (Open Source) operations
    Supports multiple models - model name is passed as parameter
    """
    _instances: Dict[str, 'GPTOSSService'] = {}
    _pipelines: Dict[str, Any] = {}
    _models_loaded: Dict[str, bool] = {}

    def __new__(cls, model_name: str = None):
        """Implement per-model singleton pattern"""
        model_name = model_name or os.getenv('GPT_MODEL_NAME', 'openai/gpt-oss-20b')

        if model_name not in cls._instances:
            instance = super(GPTOSSService, cls).__new__(cls)
            cls._instances[model_name] = instance
            cls._models_loaded[model_name] = False
        return cls._instances[model_name]

    def __init__(self, model_name: str = None):
        """
        Initialize configuration from environment variables and parameters

        Args:
            model_name: Model identifier (e.g., 'openai/gpt-oss-20b', 'openai/gpt-oss-120b')
        """
        # Model configuration - accept as parameter or from env
        self.model_name = model_name or os.getenv('GPT_MODEL_NAME', 'openai/gpt-oss-20b')

        # Skip if already initialized for this model
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.cache_dir = os.getenv('GPT_MODEL_CACHE_DIR', None)
        self.torch_dtype = os.getenv('GPT_TORCH_DTYPE', 'auto')

        # Smart device detection: cuda -> mps -> cpu
        env_device = os.getenv('GPT_DEVICE', None)
        if env_device:
            self.device = env_device
        else:
            self.device = self._detect_device()

        # Generation defaults
        self.default_max_tokens = int(os.getenv('GPT_MAX_TOKENS', '512'))
        self.default_temperature = float(os.getenv('GPT_TEMPERATURE', '0.7'))
        self.default_top_p = float(os.getenv('GPT_TOP_P', '0.95'))

        self._initialized = True

        # Load model immediately if not already loaded
        if not GPTOSSService._models_loaded.get(self.model_name, False):
            self._load_model()

    def _detect_device(self) -> str:
        """
        Detect the best available device: cuda -> mps -> cpu

        Returns:
            Device string: 'cuda', 'mps', or 'cpu'
        """
        if torch.cuda.is_available():
            logger.info("CUDA available, using GPU")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS available, using Apple Silicon GPU")
            return 'mps'
        else:
            logger.info("No GPU available, using CPU")
            return 'cpu'

    def _load_model(self):
        """Load the model once and keep it in memory"""
        if GPTOSSService._models_loaded.get(self.model_name, False):
            logger.info(f"GPT model {self.model_name} already loaded, reusing existing instance")
            return

        # Set CUDA device if specified
        if self.device == 'cuda':
            cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        logger.info(f"Loading GPT OSS model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Cache directory: {self.cache_dir or 'default'}")
        logger.info(f"Torch dtype: {self.torch_dtype}")

        # Prepare pipeline arguments
        pipeline_kwargs = {
            "task": "text-generation",
            "model": self.model_name,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device if self.device != 'mps' else None,  # MPS doesn't support device_map
        }

        # For MPS, set device directly
        if self.device == 'mps':
            pipeline_kwargs['device'] = 0  # MPS device

        if self.cache_dir:
            pipeline_kwargs['model_kwargs'] = {
                'cache_dir': os.path.expanduser(self.cache_dir)
            }

        try:
            # Load the text-generation pipeline
            GPTOSSService._pipelines[self.model_name] = pipeline(**pipeline_kwargs)
            GPTOSSService._models_loaded[self.model_name] = True

            logger.info("✓ GPT model loaded successfully and will persist in memory")
            logger.info(f"✓ Model: {self.model_name}")
            logger.info(f"✓ Device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to load GPT model: {str(e)}")
            raise

    @property
    def pipe(self):
        """Get the loaded pipeline for this model"""
        if not GPTOSSService._models_loaded.get(self.model_name, False):
            self._load_model()
        return GPTOSSService._pipelines.get(self.model_name)

    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        n: int = 1
    ) -> Dict[str, Any]:
        """
        Generate chat completion using GPT model

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            presence_penalty: Presence penalty (-2.0 to 2.0)
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            n: Number of completions to generate

        Returns:
            Dictionary containing the generated text and metadata
        """
        # Use defaults if not provided
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        logger.info(f"Generating chat completion with {len(messages)} messages")
        logger.debug(f"Parameters: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}")

        # Prepare generation arguments
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_return_sequences": n,
            "do_sample": temperature > 0,  # Use sampling if temperature > 0
        }

        # Add stop sequences if provided
        if stop:
            generation_kwargs["stop_sequence"] = stop

        # Apply penalties if supported
        if presence_penalty != 0.0:
            generation_kwargs["repetition_penalty"] = 1.0 + (presence_penalty / 2.0)

        try:
            # Generate response using the pipeline
            outputs = self.pipe(
                messages,
                **generation_kwargs
            )

            # Extract the generated text
            if n == 1:
                generated_text = outputs[0]["generated_text"][-1]["content"]
            else:
                generated_text = [output["generated_text"][-1]["content"] for output in outputs]

            # Calculate token usage (rough estimation)
            prompt_text = " ".join([msg.get("content", "") for msg in messages])
            prompt_tokens = len(prompt_text.split())

            if isinstance(generated_text, str):
                completion_tokens = len(generated_text.split())
            else:
                completion_tokens = sum(len(text.split()) for text in generated_text)

            return {
                "text": generated_text,
                "messages": messages,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens
            }

        except Exception as e:
            logger.error(f"Error generating chat completion: {str(e)}")
            raise

    def generate_text_completion(
        self,
        prompt: str,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate text completion from a simple prompt (non-chat format)

        Args:
            prompt: The text prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences

        Returns:
            Dictionary containing the generated text and metadata
        """
        # Convert simple prompt to chat format
        messages = [
            {"role": "user", "content": prompt}
        ]

        return self.generate_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop
        )


# Global singleton instances per model
_gpt_service_instances: Dict[str, GPTOSSService] = {}


def get_gpt_service(model_name: str = None) -> GPTOSSService:
    """
    Get or create the GPT service instance for a specific model
    This ensures each model is loaded once and persists throughout the process lifetime

    Args:
        model_name: Model identifier (e.g., 'openai/gpt-oss-20b', 'openai/gpt-oss-120b')
                   If None, uses GPT_MODEL_NAME env var or defaults to 'openai/gpt-oss-20b'

    Returns:
        GPTOSSService instance for the specified model
    """
    model_name = model_name or os.getenv('GPT_MODEL_NAME', 'openai/gpt-oss-20b')

    if model_name not in _gpt_service_instances:
        _gpt_service_instances[model_name] = GPTOSSService(model_name)
    return _gpt_service_instances[model_name]


if __name__ == "__main__":
    """
    Example usage demonstrating:
    1. Multiple model support
    2. Device auto-detection
    3. Chat completions
    4. Text completions
    5. Different generation parameters
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("GPT OSS Service Example")
    print("=" * 80)

    # Example 1: Using default model (20B)
    print("\n" + "=" * 80)
    print("Example 1: Default Model (20B) with Auto Device Detection")
    print("=" * 80)

    try:
        service_20b = get_gpt_service('openai/gpt-oss-20b')
        print(f"✓ Loaded model: {service_20b.model_name}")
        print(f"✓ Device: {service_20b.device}")

        # Chat completion
        print("\n--- Chat Completion ---")
        messages = [
            {"role": "system", "content": "You are a helpful assistant that explains complex topics simply."},
            {"role": "user", "content": "Explain quantum entanglement in one sentence."}
        ]

        result = service_20b.generate_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )

        print(f"Response: {result['text']}")
        print(f"Tokens: {result['prompt_tokens']} prompt + {result['completion_tokens']} completion = {result['total_tokens']} total")

    except Exception as e:
        print(f"✗ Error with 20B model: {str(e)}")
        print("Note: Make sure the model is available and you have enough GPU memory")

    # Example 2: Using 120B model (if available)
    print("\n" + "=" * 80)
    print("Example 2: Larger Model (120B) - Requires More GPU Memory")
    print("=" * 80)

    try:
        service_120b = get_gpt_service('openai/gpt-oss-120b')
        print(f"✓ Loaded model: {service_120b.model_name}")
        print(f"✓ Device: {service_120b.device}")

        # Text completion
        print("\n--- Text Completion ---")
        result = service_120b.generate_text_completion(
            prompt="The three laws of robotics are:",
            temperature=0.5,
            max_tokens=150
        )

        print(f"Response: {result['text']}")
        print(f"Tokens: {result['total_tokens']} total")

    except Exception as e:
        print(f"✗ Error with 120B model: {str(e)}")
        print("Note: 120B model requires ~240GB GPU memory (multi-GPU setup)")

    # Example 3: Multiple completions
    print("\n" + "=" * 80)
    print("Example 3: Multiple Completions (n=3)")
    print("=" * 80)

    try:
        service = get_gpt_service('openai/gpt-oss-20b')

        messages = [
            {"role": "user", "content": "Write a creative opening line for a sci-fi story."}
        ]

        result = service.generate_chat_completion(
            messages=messages,
            temperature=0.9,  # Higher temperature for more creativity
            max_tokens=50,
            n=3  # Generate 3 different completions
        )

        print("Generated 3 different responses:")
        for i, text in enumerate(result['text'], 1):
            print(f"\n{i}. {text}")

    except Exception as e:
        print(f"✗ Error: {str(e)}")

    # Example 4: Different temperature values
    print("\n" + "=" * 80)
    print("Example 4: Temperature Comparison (Low vs High)")
    print("=" * 80)

    try:
        service = get_gpt_service('openai/gpt-oss-20b')
        prompt_msg = [{"role": "user", "content": "What is Python?"}]

        # Low temperature (more focused/deterministic)
        print("\n--- Low Temperature (0.1) - More Focused ---")
        result_low = service.generate_chat_completion(
            messages=prompt_msg,
            temperature=0.1,
            max_tokens=100
        )
        print(f"Response: {result_low['text']}")

        # High temperature (more creative/random)
        print("\n--- High Temperature (1.5) - More Creative ---")
        result_high = service.generate_chat_completion(
            messages=prompt_msg,
            temperature=1.5,
            max_tokens=100
        )
        print(f"Response: {result_high['text']}")

    except Exception as e:
        print(f"✗ Error: {str(e)}")

    # Example 5: Using stop sequences
    print("\n" + "=" * 80)
    print("Example 5: Stop Sequences")
    print("=" * 80)

    try:
        service = get_gpt_service('openai/gpt-oss-20b')

        result = service.generate_chat_completion(
            messages=[{"role": "user", "content": "List 3 programming languages:\n1."}],
            temperature=0.7,
            max_tokens=100,
            stop=["\n4.", "END"]  # Stop at 4th item or "END"
        )

        print(f"Response: {result['text']}")

    except Exception as e:
        print(f"✗ Error: {str(e)}")

    # Example 6: Device detection info
    print("\n" + "=" * 80)
    print("Example 6: Device Detection Information")
    print("=" * 80)

    service = get_gpt_service('openai/gpt-oss-20b')

    print(f"Model: {service.model_name}")
    print(f"Device: {service.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")

    print(f"Cache directory: {service.cache_dir or 'default'}")
    print(f"Torch dtype: {service.torch_dtype}")

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
    print("\nTips:")
    print("- Use lower temperature (0.1-0.3) for factual/focused responses")
    print("- Use higher temperature (0.7-1.5) for creative/diverse responses")
    print("- Device auto-detection priority: CUDA -> MPS -> CPU")
    print("- Multiple models can be loaded simultaneously (if memory allows)")
    print("- Each model instance is a singleton and persists in memory")



