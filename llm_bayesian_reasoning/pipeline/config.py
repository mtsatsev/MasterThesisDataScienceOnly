from enum import Enum
from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from transformers import BitsAndBytesConfig


class EstimatorType(str, Enum):
    """Enumeration of supported estimator types."""

    TRUE_FALSE_LLM = "TrueFalseLLM"
    LIKELIHOOD_BASED_PERPLEXITY = "Perplexity"
    LIKELIHOOD_BASED_CONTRASTIVE = "Contrastive"


class LogicBackendType(str, Enum):
    """Enumeration of supported logic evaluation backends."""

    PROBLOG = "ProbLog"
    DEEPPROBLOG = "DeepProbLog"


class EstimatorConfig(BaseModel):
    """Configuration for probability estimation.

    Attributes:
        model_name: Name of the pre-trained LLM model.
        device: Device to run the model on (cuda, cpu, etc).
        true_token: Token representing True (default: " True").
        false_token: Token representing False (default: " False").
        estimator_type: Type of estimator to use (e.g., "true_false_llm").
    """

    model_name: str = Field(
        default="microsoft/phi-2",
        description="HuggingFace model identifier",
        min_length=1,
    )
    device: str = Field(
        default="cuda",
        description="Device to run the model on (cuda, cpu, etc)",
        min_length=1,
    )
    true_token: str = Field(
        default=" True", description="Token representing True", min_length=1
    )
    false_token: str = Field(
        default=" False", description="Token representing False", min_length=1
    )
    estimator_type: EstimatorType = Field(
        default=EstimatorType.TRUE_FALSE_LLM, description="Type of estimator to use"
    )
    contrastive_temperature: float = Field(
        default=1.0,
        le=1.0,
        ge=0.0,
        description="Temperature for contrastive estimation",
    )
    include_retrieved_text: bool = Field(
        default=False,
        description="Include retrieved document text in Problog atom context",
    )
    quantization: BitsAndBytesConfig = Field(
        default_factory=lambda: BitsAndBytesConfig(load_in_4bit=True)
    )

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_estimator_config(self) -> Self:
        """Validate estimator configuration."""
        if self.estimator_type == EstimatorType.LIKELIHOOD_BASED_CONTRASTIVE:
            if self.contrastive_temperature is None:
                raise ValueError(
                    "contrastive_temperature is required for likelihood-based estimators"
                )
        return self


class PipelineConfig(BaseModel):
    """Configuration for the full retrieval + scoring pipeline.

    Attributes:
        top_n: Number of candidate entities to retrieve with BM25.
        top_k: Number of top entities to keep after Problog reranking.
        batch_size: Number of documents per batch when building the BM25 index.
        index_path: Directory for storing the BM25 index.
        output_path: File path to write results (JSONL).
        estimator_config: Configuration for the LLM estimator.
    """

    top_n: int = Field(default=50, ge=1, description="BM25 retrieval pool size")
    top_k: int = Field(default=10, ge=1, description="Reranked output size")
    batch_size: int = Field(
        default=1000, ge=1, description="Document batch size for index building"
    )
    index_path: Path = Field(description="Directory path for BM25 index storage")
    output_path: Path = Field(description="JSONL file path for results output")
    estimator_config: EstimatorConfig = Field(
        default_factory=EstimatorConfig,
        description="LLM estimator configuration",
    )
    logic_backend: LogicBackendType = Field(
        default=LogicBackendType.PROBLOG,
        description="Logic backend used to evaluate the final program",
    )
    mlflow_experiment: str | None = Field(
        default=None,
        description=(
            "MLflow experiment name. When set, pipeline params and per-record "
            "metrics are tracked in an MLflow run."
        ),
    )

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_top_n_k(self) -> Self:
        if self.top_k > self.top_n:
            raise ValueError(f"top_k ({self.top_k}) must be <= top_n ({self.top_n})")
        return self
