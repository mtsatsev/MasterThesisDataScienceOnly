import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from llm_bayesian_reasoning.estimators.factory import (
    create_estimator_from_components,
    load_model_and_tokenizer_from_config,
)
from llm_bayesian_reasoning.pipeline.config import (
    EstimatorType,
    ExperimentSuiteConfig,
    ExperimentVariantConfig,
    LogicBackendType,
    RetrieverConfig,
)
from llm_bayesian_reasoning.pipeline.metrics import compute_metrics
from llm_bayesian_reasoning.pipeline.pipeline import (
    build_record_result,
    create_logic_backend,
    score_candidate_documents,
)
from llm_bayesian_reasoning.pipeline.record_loader import (
    build_atoms_for_estimator,
    load_preprocessed_records,
)
from llm_bayesian_reasoning.pipeline.results_io import safe_name, write_result_row
from llm_bayesian_reasoning.pipeline.retrieval_service import CachedRetrieverService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelResourceKey:
    model_name: str
    device: str
    quantization_repr: str

    @classmethod
    def from_variant(cls, variant: ExperimentVariantConfig) -> "ModelResourceKey":
        estimator_config = variant.estimator_config
        return cls(
            model_name=estimator_config.model_name,
            device=estimator_config.device,
            quantization_repr=repr(estimator_config.quantization),
        )


@dataclass(frozen=True)
class ScoringPolicyKey:
    logic_backend: LogicBackendType
    estimator_type: EstimatorType
    include_retrieved_text: bool
    true_token: str
    false_token: str
    contrastive_temperature: float

    @classmethod
    def from_variant(cls, variant: ExperimentVariantConfig) -> "ScoringPolicyKey":
        estimator_config = variant.estimator_config
        return cls(
            logic_backend=variant.logic_backend,
            estimator_type=estimator_config.estimator_type,
            include_retrieved_text=estimator_config.include_retrieved_text,
            true_token=estimator_config.true_token,
            false_token=estimator_config.false_token,
            contrastive_temperature=estimator_config.contrastive_temperature,
        )


@dataclass(frozen=True)
class PlannedScoringFamily:
    key: ScoringPolicyKey
    reference_variant: ExperimentVariantConfig
    variants: tuple[ExperimentVariantConfig, ...]
    scoring_top_n: int


@dataclass(frozen=True)
class PlannedModelFamily:
    key: ModelResourceKey
    reference_variant: ExperimentVariantConfig
    scoring_families: tuple[PlannedScoringFamily, ...]


@dataclass(frozen=True)
class PlannedRetrieverExecution:
    retriever_config: RetrieverConfig
    retrieval_pool_size: int
    model_families: tuple[PlannedModelFamily, ...]


@dataclass(frozen=True)
class SuiteExecutionPlan:
    retriever_executions: tuple[PlannedRetrieverExecution, ...]


@dataclass
class VariantOutputState:
    handle: TextIO
    results: dict[int | str, dict[str, Any]]
    results_path: Path
    metrics_path: Path
    variant: ExperimentVariantConfig


def make_suite_results_folder(root: Path) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_dir = root / f"{timestamp}_experiment_suite"
    results_dir.mkdir(parents=True, exist_ok=False)
    return results_dir


def get_candidate_pool_size(
    retriever_config: RetrieverConfig,
    variants: list[ExperimentVariantConfig],
) -> int:
    requested_top_n = max(variant.top_n for variant in variants)
    if retriever_config.retrieval_pool_size is None:
        return requested_top_n
    if retriever_config.retrieval_pool_size < requested_top_n:
        logger.info(
            "Expanding retrieval pool for %s from configured %d to required %d",
            retriever_config.name,
            retriever_config.retrieval_pool_size,
            requested_top_n,
        )
    return max(retriever_config.retrieval_pool_size, requested_top_n)


def plan_suite_execution(suite_config: ExperimentSuiteConfig) -> SuiteExecutionPlan:
    variants_by_retriever: dict[str, list[ExperimentVariantConfig]] = defaultdict(list)
    for variant in suite_config.variants:
        variants_by_retriever[variant.retriever_name].append(variant)

    retriever_executions: list[PlannedRetrieverExecution] = []
    for retriever_config in suite_config.retrievers:
        retriever_variants = variants_by_retriever.get(retriever_config.name, [])
        if not retriever_variants:
            continue

        variants_by_model: dict[ModelResourceKey, list[ExperimentVariantConfig]] = defaultdict(list)
        for variant in retriever_variants:
            variants_by_model[ModelResourceKey.from_variant(variant)].append(variant)

        model_families: list[PlannedModelFamily] = []
        for model_key, model_variants in variants_by_model.items():
            variants_by_scoring: dict[ScoringPolicyKey, list[ExperimentVariantConfig]] = defaultdict(list)
            for variant in model_variants:
                variants_by_scoring[ScoringPolicyKey.from_variant(variant)].append(variant)

            scoring_families: list[PlannedScoringFamily] = []
            for scoring_key, scoring_variants in variants_by_scoring.items():
                scoring_families.append(
                    PlannedScoringFamily(
                        key=scoring_key,
                        reference_variant=scoring_variants[0],
                        variants=tuple(scoring_variants),
                        scoring_top_n=max(variant.top_n for variant in scoring_variants),
                    )
                )

            model_families.append(
                PlannedModelFamily(
                    key=model_key,
                    reference_variant=model_variants[0],
                    scoring_families=tuple(scoring_families),
                )
            )

        retriever_executions.append(
            PlannedRetrieverExecution(
                retriever_config=retriever_config,
                retrieval_pool_size=get_candidate_pool_size(
                    retriever_config,
                    retriever_variants,
                ),
                model_families=tuple(model_families),
            )
        )

    return SuiteExecutionPlan(retriever_executions=tuple(retriever_executions))


def prepare_variant_outputs(
    results_dir: Path,
    variants: list[ExperimentVariantConfig],
) -> tuple[dict[str, VariantOutputState], dict[str, dict[str, Any]]]:
    variant_states: dict[str, VariantOutputState] = {}
    manifest: dict[str, dict[str, Any]] = {}

    for variant in variants:
        variant_dir = results_dir / safe_name(variant.name)
        variant_dir.mkdir(parents=True, exist_ok=False)
        results_path = variant_dir / "results.jsonl"
        metrics_path = variant_dir / "metrics.json"
        variant_states[variant.name] = VariantOutputState(
            handle=results_path.open("w", encoding="utf-8"),
            results={},
            results_path=results_path,
            metrics_path=metrics_path,
            variant=variant,
        )
        manifest[variant.name] = {
            "results_path": str(results_path),
            "metrics_path": str(metrics_path),
            "retriever_name": variant.retriever_name,
            "top_n": variant.top_n,
            "top_k": variant.top_k,
            "logic_backend": variant.logic_backend.value,
            "estimator_type": variant.estimator_config.estimator_type.value,
            "include_retrieved_text": variant.estimator_config.include_retrieved_text,
            "model_name": variant.estimator_config.model_name,
        }

    return variant_states, manifest


def suite_mlflow_params(suite_config: ExperimentSuiteConfig) -> dict[str, Any]:
    return {
        "limit": suite_config.limit,
        "use_metadata_ground_truth": suite_config.use_metadata_ground_truth,
        "num_retrievers": len(suite_config.retrievers),
        "num_variants": len(suite_config.variants),
    }


def variant_mlflow_params(
    variant: ExperimentVariantConfig,
    retriever_config: RetrieverConfig,
) -> dict[str, Any]:
    estimator_config = variant.estimator_config
    return {
        "retriever_name": retriever_config.name,
        "retriever_type": retriever_config.retriever_type.value,
        "index_path": str(retriever_config.index_path),
        "retriever_model_name": retriever_config.retriever_model_name,
        "retrieval_pool_size": retriever_config.retrieval_pool_size,
        "top_n": variant.top_n,
        "top_k": variant.top_k,
        "logic_backend": variant.logic_backend.value,
        "estimator_type": estimator_config.estimator_type.value,
        "include_retrieved_text": estimator_config.include_retrieved_text,
        "model_name": estimator_config.model_name,
        "device": estimator_config.device,
        "true_token": estimator_config.true_token,
        "false_token": estimator_config.false_token,
        "contrastive_temperature": estimator_config.contrastive_temperature,
    }


def log_variant_to_mlflow(
    mlflow: Any,
    variant_name: str,
    state: VariantOutputState,
    variant: ExperimentVariantConfig,
    retriever_config: RetrieverConfig,
    metrics: dict[str, float],
) -> None:
    with mlflow.start_run(run_name=variant_name, nested=True):
        mlflow.log_params(variant_mlflow_params(variant, retriever_config))
        mlflow.set_tag("variant_name", variant_name)
        mlflow.set_tag("retriever_name", retriever_config.name)
        mlflow.set_tag("results_path", str(state.results_path))
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(state.results_path))
        if state.metrics_path.exists():
            mlflow.log_artifact(str(state.metrics_path))


def run_experiment_suite(
    suite_config: ExperimentSuiteConfig,
    config_payload: dict[str, Any],
) -> Path:
    suite_config.results_root.mkdir(parents=True, exist_ok=True)
    suite_results_dir = make_suite_results_folder(suite_config.results_root)
    suite_config_path = suite_results_dir / "suite_config.json"
    suite_config_path.write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )

    mlflow = None
    mlflow_active = False
    if suite_config.mlflow_experiment is not None:
        try:
            import mlflow as _mlflow

            mlflow = _mlflow
            mlflow.set_experiment(suite_config.mlflow_experiment)
            mlflow.start_run(run_name=suite_results_dir.name)
            mlflow.log_params(suite_mlflow_params(suite_config))
            mlflow.set_tag("results_path", str(suite_results_dir))
            mlflow.set_tag("suite_config_path", str(suite_config_path))
            mlflow_active = True
            logger.info(
                "MLflow tracking enabled for suite (experiment=%r)",
                suite_config.mlflow_experiment,
            )
        except ImportError:
            logger.warning(
                "mlflow is not installed; skipping suite tracking. pip install mlflow"
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "MLflow initialisation failed for suite — continuing without tracking.",
                exc_info=True,
            )

    logger.info("Loading preprocessed data from %s", suite_config.data_file)
    data, ground_truth = load_preprocessed_records(
        suite_config.data_file,
        limit=suite_config.limit,
        use_metadata_ground_truth=suite_config.use_metadata_ground_truth,
    )
    ground_truth_map: dict[int | str, list[str]] = ground_truth or {}
    logger.info("Loaded %d records for the experiment suite", len(data))

    variant_states, manifest = prepare_variant_outputs(
        suite_results_dir,
        suite_config.variants,
    )
    plan = plan_suite_execution(suite_config)
    retrievers_by_name = {
        retriever_config.name: retriever_config
        for retriever_config in suite_config.retrievers
    }

    try:
        for retriever_execution in plan.retriever_executions:
            retriever_config = retriever_execution.retriever_config
            logger.info(
                "Preparing retriever %s (%s) with retrieval pool size %d",
                retriever_config.name,
                retriever_config.retriever_type.value,
                retriever_execution.retrieval_pool_size,
            )
            retrieval_service = CachedRetrieverService(
                retriever_config=retriever_config,
                cache_root=suite_config.retrieval_cache_dir,
                retrieval_pool_size=retriever_execution.retrieval_pool_size,
            )

            for model_family in retriever_execution.model_families:
                reference_variant = model_family.reference_variant
                logger.info(
                    "Loading shared model resources for %s",
                    reference_variant.estimator_config.model_name,
                )
                model, tokenizer = load_model_and_tokenizer_from_config(
                    reference_variant.estimator_config
                )

                for scoring_family in model_family.scoring_families:
                    scoring_variant = scoring_family.reference_variant
                    logger.info(
                        "Scoring variant family rooted at %s (retriever=%s, top_n=%d)",
                        scoring_variant.name,
                        retriever_config.name,
                        scoring_family.scoring_top_n,
                    )
                    estimator = create_estimator_from_components(
                        scoring_variant.estimator_config,
                        model,
                        tokenizer,
                    )
                    logic_backend = create_logic_backend(scoring_variant.logic_backend)

                    for record_id, record in data.items():
                        query = record["query"]
                        candidate_pool = retrieval_service.get_candidate_pool(
                            record_id=record_id,
                            query=query,
                        )
                        candidate_documents = candidate_pool.documents[
                            : scoring_family.scoring_top_n
                        ]
                        atoms = build_atoms_for_estimator(
                            record,
                            scoring_variant.estimator_config.estimator_type,
                        )
                        entity_scores = score_candidate_documents(
                            atoms=atoms,
                            formula=record["problog_formula"],
                            candidate_documents=candidate_documents,
                            estimator=estimator,
                            logic_backend=logic_backend,
                            include_retrieved_text=scoring_variant.estimator_config.include_retrieved_text,
                            record_id=record_id,
                        )

                        for variant in scoring_family.variants:
                            variant_candidate_documents = candidate_pool.documents[
                                : variant.top_n
                            ]
                            relevant = None
                            if record_id in ground_truth_map:
                                relevant = set(ground_truth_map[record_id])
                            record_result, _ = build_record_result(
                                query=query,
                                atoms=atoms,
                                retrieved_documents=variant_candidate_documents,
                                entity_scores=entity_scores,
                                top_k=variant.top_k,
                                relevant=relevant,
                            )
                            variant_state = variant_states[variant.name]
                            variant_state.results[record_id] = record_result
                            write_result_row(
                                variant_state.handle,
                                record_id,
                                record_result,
                            )

        for variant_name, state in variant_states.items():
            state.handle.close()
            variant = state.variant
            metrics: dict[str, float] = {}
            if ground_truth is not None:
                metrics = compute_metrics(
                    state.results,
                    ground_truth_map,
                    variant.top_k,
                )
                state.metrics_path.write_text(
                    json.dumps(metrics, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            manifest[variant_name]["metrics"] = metrics
            if mlflow_active and mlflow is not None:
                try:
                    retriever_config = retrievers_by_name[variant.retriever_name]
                    log_variant_to_mlflow(
                        mlflow=mlflow,
                        variant_name=variant_name,
                        state=state,
                        variant=variant,
                        retriever_config=retriever_config,
                        metrics=metrics,
                    )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "MLflow variant logging failed for %s — skipping.",
                        variant_name,
                        exc_info=True,
                    )

    finally:
        for state in variant_states.values():
            if not state.handle.closed:
                state.handle.close()

    manifest_path = suite_results_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if mlflow_active and mlflow is not None:
        try:
            mlflow.log_artifact(str(suite_config_path))
            mlflow.log_artifact(str(manifest_path))
            mlflow.end_run()
            logger.info("MLflow suite run ended. View with: mlflow ui")
        except Exception:  # noqa: BLE001
            logger.warning(
                "MLflow suite cleanup failed — skipping.",
                exc_info=True,
            )

    logger.info("Experiment suite finished. Results folder: %s", suite_results_dir)
    return suite_results_dir