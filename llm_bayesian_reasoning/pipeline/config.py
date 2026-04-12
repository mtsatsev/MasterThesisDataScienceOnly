from enum import Enum
from pathlib import Path
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
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


class RetrieverType(str, Enum):
    """Enumeration of supported retrievers."""

    BM25 = "BM25"
    E5 = "E5"


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

    @field_validator("quantization", mode="before")
    @classmethod
    def parse_quantization(
        cls,
        value: BitsAndBytesConfig | dict[str, Any] | None,
    ) -> BitsAndBytesConfig:
        """Allow quantization to be passed as a JSON object."""
        if value is None:
            return BitsAndBytesConfig(load_in_4bit=True)
        if isinstance(value, BitsAndBytesConfig):
            return value
        if isinstance(value, dict):
            return BitsAndBytesConfig(**value)
        raise TypeError("quantization must be a BitsAndBytesConfig or a JSON object")

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
    retriever_type: RetrieverType = Field(
        default=RetrieverType.BM25,
        description="Retriever used to construct the candidate pool",
    )
    retriever_model_name: str = Field(
        default="intfloat/e5-base-v2",
        description="Dense retriever model identifier when applicable",
        min_length=1,
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


class RetrieverConfig(BaseModel):
    """Configuration for a reusable retriever in an experiment suite."""

    name: str = Field(min_length=1, description="Unique identifier for the retriever")
    retriever_type: RetrieverType = Field(description="Retriever implementation")
    index_path: Path = Field(description="Path where the retriever index is stored")
    documents_path: Path = Field(description="Corpus used to build or load the index")
    batch_size: int = Field(default=1000, ge=1, description="Index build batch size")
    index_limit: int | None = Field(
        default=None,
        ge=1,
        description="Optional limit on indexed documents",
    )
    retriever_model_name: str = Field(
        default="intfloat/e5-base-v2",
        description="Dense retriever model identifier when applicable",
        min_length=1,
    )
    retrieval_pool_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional cached candidate-pool size. When omitted, the suite runner "
            "uses the maximum top_n requested by variants that reference this retriever."
        ),
    )

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class ExperimentVariantConfig(BaseModel):
    """Single experiment variant within a suite run."""

    name: str = Field(min_length=1, description="Variant name used in output paths")
    retriever_name: str = Field(
        min_length=1,
        description="Name of the retriever configuration to use",
    )
    top_n: int = Field(default=50, ge=1, description="Candidate pool size to rerank")
    top_k: int = Field(default=10, ge=1, description="Final ranked output size")
    logic_backend: LogicBackendType = Field(
        default=LogicBackendType.PROBLOG,
        description="Logic backend used for this variant",
    )
    estimator_config: EstimatorConfig = Field(
        default_factory=EstimatorConfig,
        description="Estimator configuration for this variant",
    )

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_top_n_k(self) -> Self:
        if self.top_k > self.top_n:
            raise ValueError(f"top_k ({self.top_k}) must be <= top_n ({self.top_n})")
        return self


class ExperimentSuiteConfig(BaseModel):
    """Configuration for a multi-variant experiment run."""

    data_file: Path = Field(description="Preprocessed JSONL file with logical records")
    results_root: Path = Field(
        default=Path("llm_bayesian_reasoning/results"),
        description="Root directory where suite outputs are written",
    )
    retrieval_cache_dir: Path = Field(
        default=Path("llm_bayesian_reasoning/data/retrieval_cache"),
        description="Directory used for persistent retrieval caches",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Optional limit on loaded records",
    )
    use_metadata_ground_truth: bool = Field(
        default=False,
        description="Extract ground truth from metadata.relevance_ratings when present",
    )
    mlflow_experiment: str | None = Field(
        default=None,
        description=(
            "MLflow experiment name. When set, the suite logs a parent run and "
            "nested per-variant runs with metrics and artifacts."
        ),
    )
    retrievers: list[RetrieverConfig] = Field(
        min_length=1,
        description="Retrievers available to the suite",
    )
    variants: list[ExperimentVariantConfig] = Field(
        min_length=1,
        description="Experiment variants to execute",
    )

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_suite(self) -> Self:
        retriever_names = [retriever.name for retriever in self.retrievers]
        unique_retriever_names = set(retriever_names)
        if len(unique_retriever_names) != len(retriever_names):
            raise ValueError(
                "Retriever names in the suite configuration must be unique"
            )

        variant_names = [variant.name for variant in self.variants]
        unique_variant_names = set(variant_names)
        if len(unique_variant_names) != len(variant_names):
            raise ValueError("Variant names in the suite configuration must be unique")

        missing = sorted(
            {
                variant.retriever_name
                for variant in self.variants
                if variant.retriever_name not in unique_retriever_names
            }
        )
        if missing:
            raise ValueError(
                "Variants reference unknown retrievers: " + ", ".join(missing)
            )
        return self
