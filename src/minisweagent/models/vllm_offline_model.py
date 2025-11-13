"""vLLM Offline Model for direct GPU inference without a server."""

import logging
from dataclasses import dataclass, field
from typing import Any

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("vllm_offline_model")

# Try to import vLLM, but make it optional
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning(
        "vLLM is not installed. To use VLLMOfflineModel, install vLLM: "
        "pip install vllm"
    )


@dataclass
class VLLMOfflineModelConfig:
    model_name: str
    """Path to the model (local path or HuggingFace model ID)"""

    model_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs passed to vLLM's LLM class (e.g., tensor_parallel_size, gpu_memory_utilization)"""

    sampling_kwargs: dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.0,
        "max_tokens": 4096,
        "top_p": 1.0,
    })
    """Sampling parameters for generation"""

    cost_per_input_token: float = 0.0
    """Cost per input token (for tracking purposes)"""

    cost_per_output_token: float = 0.0
    """Cost per output token (for tracking purposes)"""


class VLLMOfflineModel:
    """Model class that uses vLLM's offline inference API directly on GPU."""

    def __init__(self, *, config_class: type = VLLMOfflineModelConfig, **kwargs):
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm"
            )

        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0

        # Initialize the vLLM engine
        logger.info(f"Loading vLLM model: {self.config.model_name}")
        logger.info(f"Model kwargs: {self.config.model_kwargs}")

        self.llm = LLM(
            model=self.config.model_name,
            **self.config.model_kwargs,
        )

        # Create sampling params
        self.sampling_params = SamplingParams(**self.config.sampling_kwargs)

        logger.info("vLLM model loaded successfully")

    def _format_messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Convert OpenAI-style messages to a prompt string.

        This is a simple implementation that concatenates messages.
        You may want to customize this based on your model's chat template.
        """
        # Try to use the model's built-in chat template if available
        try:
            # vLLM 0.4.0+ supports get_tokenizer()
            tokenizer = self.llm.get_tokenizer()
            if hasattr(tokenizer, 'apply_chat_template'):
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        except Exception as e:
            logger.debug(f"Could not use chat template: {e}")

        # Fallback: simple concatenation
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        # Add final assistant prompt
        prompt_parts.append("Assistant:")

        return "\n\n".join(prompt_parts)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Query the model with a list of messages."""
        # Convert messages to prompt
        prompt = self._format_messages_to_prompt(messages)

        # Override sampling params if provided
        sampling_params = self.sampling_params
        if kwargs:
            merged_params = self.config.sampling_kwargs.copy()
            merged_params.update(kwargs)
            sampling_params = SamplingParams(**merged_params)

        # Generate
        logger.debug(f"Generating with prompt length: {len(prompt)} chars")
        outputs = self.llm.generate([prompt], sampling_params)

        # Extract the generated text
        generated_text = outputs[0].outputs[0].text

        # Calculate cost (if specified)
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = len(outputs[0].outputs[0].token_ids)

        instance_cost = (
            prompt_tokens * self.config.cost_per_input_token +
            completion_tokens * self.config.cost_per_output_token
        )

        self.cost += instance_cost
        self.n_calls += 1
        GLOBAL_MODEL_STATS.add(instance_cost)

        logger.info(
            f"Generated {completion_tokens} tokens "
            f"(prompt: {prompt_tokens} tokens, cost: ${instance_cost:.6f})"
        )

        return {
            "content": generated_text,
            "extra": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": instance_cost,
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        """Return template variables for config rendering."""
        return {
            "model_name": self.config.model_name,
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
        }
