from logging import getLogger

from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_bayesian_reasoning.problog_models.problog_models import ProblogAtom

logger = getLogger(__name__)


class BaseEstimator:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "microsoft/phi-2",
        device: str = "cuda",
        **kwargs,
    ) -> "BaseEstimator":
        """Load a pretrained model with proper device handling.

        Args:
            model_name: HuggingFace model identifier
            device: Target device (cuda/cpu) - note: device_map="auto" is always used
            **kwargs: Additional kwargs like quantization_config

        Returns:
            Initialized BaseEstimator
        """
        import os

        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Optimize device_map strategy based on available VRAM
        available_vram_gb = 0
        if torch.cuda.is_available():
            available_vram_gb = torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )
            logger.debug(f"Available GPU VRAM: {available_vram_gb:.1f} GB")

        # Use device_map="auto" with CPU offloading for better memory management
        # This prevents model from staying entirely on CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            offload_folder=os.path.expanduser("~/.cache/huggingface/offload"),
            **kwargs,
        )

        # Diagnostic: warn if model is on CPU
        if hasattr(model, "device"):
            logger.warning(
                f"WARNING: Model device is {model.device}. If this is 'cpu', GPU is not being used!"
            )
        else:
            # Check first layer device
            first_param = next(model.parameters(), None)
            if first_param is not None:
                first_device = first_param.device
                if first_device.type == "cpu":
                    logger.warning(
                        "WARNING: Model parameters are on CPU. GPU VRAM may be insufficient."
                    )
                else:
                    logger.debug(
                        f"Model loaded with device_map='auto'. First layer on: {first_device}"
                    )

        return cls(model=model, tokenizer=tokenizer, device=device)

    def score_probability(
        self,
        predicates: list[ProblogAtom] | list[tuple[ProblogAtom, ProblogAtom]],
        entity: str,
    ) -> list[ProblogAtom]:
        raise NotImplementedError("Subclasses must implement this method.")
