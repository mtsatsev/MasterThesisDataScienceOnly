import json
import logging
from pathlib import Path
from typing import Any

import mlflow

from llm_bayesian_reasoning.estimators.base import BaseEstimator
from llm_bayesian_reasoning.estimators.factory import create_estimator_from_config
from llm_bayesian_reasoning.estimators.scoring_inputs import (
    ScoringInput,
    clone_scoring_inputs_with_document_context,
)
from llm_bayesian_reasoning.pipeline.config import (
    LogicBackendType,
    PipelineConfig,
    RetrieverType,
)
from llm_bayesian_reasoning.pipeline.logic_backends import (
    DeepProbLogBackend,
    LogicBackend,
    ProbLogBackend,
)
from llm_bayesian_reasoning.pipeline.metrics import (
    compute_metrics,
    compute_record_metrics,
)
from llm_bayesian_reasoning.pipeline.results_io import write_result_row
from llm_bayesian_reasoning.problog_models.problog_models import (
    ProblogAtom,
    ProblogFormula,
)
from llm_bayesian_reasoning.retrievers.base_retriever import BaseRetriever
from llm_bayesian_reasoning.retrievers.document import ScoredDocument
from llm_bayesian_reasoning.retrievers.factory import build_or_load_retriever

logger = logging.getLogger(__name__)


def create_logic_backend(backend_type: LogicBackendType) -> LogicBackend:
    if backend_type == LogicBackendType.PROBLOG:
        return ProbLogBackend()
    if backend_type == LogicBackendType.DEEPPROBLOG:
        return DeepProbLogBackend()
    raise ValueError(f"Unsupported logic backend: {backend_type}")


def build_or_load_index(
    documents_path: str | Path,
    index_path: str | Path,
    batch_size: int = 1000,
    limit: int | None = None,
) -> BaseRetriever:
    """Backward-compatible wrapper that builds or loads a BM25 retriever."""
    return build_or_load_retriever(
        documents_path=documents_path,
        index_path=index_path,
        retriever_type=RetrieverType.BM25,
        batch_size=batch_size,
        limit=limit,
    )


def score_candidate_documents(
    atoms: list[ScoringInput],
    formula: ProblogFormula,
    candidate_documents: list[ScoredDocument],
    estimator: BaseEstimator,
    logic_backend: LogicBackend,
    include_retrieved_text: bool = False,
    record_id: int | str | None = None,
) -> dict[str, float]:
    """Score an ordered candidate pool for a single record."""
    entity_scores: dict[str, float] = {}
    for document in candidate_documents:
        entity = document.title
        scoring_atoms = atoms
        if include_retrieved_text:
            scoring_atoms = clone_scoring_inputs_with_document_context(
                atoms,
                document.text,
            )
        try:
            scored_atoms = estimator.score_probability(scoring_atoms, entity)
            probability = logic_backend.evaluate(scored_atoms, formula, entity)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Scoring failed for record %s, entity %r — skipping entity",
                record_id,
                entity,
            )
            continue
        entity_scores[entity] = probability
        logger.debug("  entity=%r  prob=%.4f", entity, probability)
    return entity_scores


def build_record_result(
    query: str,
    atoms: list[ScoringInput],
    retrieved_documents: list[ScoredDocument],
    entity_scores: dict[str, float],
    top_k: int,
    relevant: set[str] | None = None,
) -> tuple[dict, list[tuple[str, float]]]:
    """Create a persisted record result from scored candidate entities."""
    retrieved_entities = [document.title for document in retrieved_documents]
    retrieval_scores = {
        document.title: document.score for document in retrieved_documents
    }
    candidate_entity_set = set(retrieved_entities)
    ranked = sorted(
        (
            (entity, score)
            for entity, score in entity_scores.items()
            if entity in candidate_entity_set
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    ranked_entities = [entity for entity, _ in ranked]

    record_result: dict[str, Any] = {
        "query": query,
        "retrieved_entities": retrieved_entities,
        "retrieval_scores": retrieval_scores,
        "ranked_entities": ranked_entities,
        "scores": {entity: score for entity, score in ranked},
        "num_atoms": len(atoms),
        "reranked_pool_size": len(ranked_entities),
        "metric_top_k": top_k,
    }
    if relevant is not None:
        record_result["ground_truth"] = list(relevant)
        relevant_ranks = {
            entity: index + 1
            for index, (entity, _score) in enumerate(ranked)
            if entity in relevant
        }
        record_result["relevant_ranks"] = relevant_ranks
        record_result["first_relevant_rank"] = (
            min(relevant_ranks.values()) if relevant_ranks else None
        )
        record_metrics = compute_record_metrics(
            retrieved_entities,
            relevant,
            top_k,
            prefix="retrieval",
        )
        record_metrics.update(
            compute_record_metrics(
                ranked_entities,
                relevant,
                top_k,
                prefix="reranked",
            )
        )
        record_result["record_metrics"] = record_metrics
    return record_result, ranked


def run_pipeline(
    data: dict,
    retriever: BaseRetriever,
    estimator: BaseEstimator | None,
    config: PipelineConfig,
    ground_truth: dict[int | str, list[str]] | None = None,
) -> dict:
    """Run the full retrieval → scoring → reranking pipeline.

    For every record in ``data`` the pipeline performs the following steps:

    1. Retrieve the top-``top_n`` entity titles using the full query string
       with BM25.
    2. For each candidate entity, score every atom with the LLM estimator to
       obtain per-atom probabilities.
    3. Assemble a ProbLog program from the scored atoms and the logical formula,
       then evaluate it to obtain an entity-level probability.
    4. Sort entities by that probability (descending) across the full top-``top_n`` pool.
    5. Append the full reranked pool to ``output_path`` immediately (checkpointing).

    Args:
        data: Mapping of record id → ``{"query": str, "atoms": atoms_input,
            "problog_formula": ProblogFormula}``, where ``atoms_input`` is either
            a list of positive atoms or a list of ``(positive_atom, negated_atom)``
            tuples for contrastive scoring.
        retriever: A loaded :class:`BM25Retriever`.
        estimator: A loaded :class:`TrueFalseLLMEstimator`.
        config: Pipeline hyper-parameters.

    Returns:
        Dict with keys ``"results"`` (per-record output) and ``"metrics"``
        (aggregated evaluation metrics; empty dict when no ground_truth).
    """
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional MLflow tracking
    _mlflow_active = False
    if config.mlflow_experiment is not None:
        try:
            mlflow.set_experiment(config.mlflow_experiment)
            mlflow.start_run(nested=True)
            mlflow.log_params(
                {
                    "top_n": config.top_n,
                    "top_k": config.top_k,
                    "batch_size": config.batch_size,
                    "model_name": config.estimator_config.model_name,
                    "device": config.estimator_config.device,
                    "include_retrieved_text": config.estimator_config.include_retrieved_text,
                }
            )
            # Tag the run with the on-disk results folder for easy lookup
            try:
                mlflow.set_tag("results_path", str(output_path.parent))
            except Exception:
                # best-effort; do not fail the run if tagging fails
                logger.debug("Could not set mlflow tag 'results_path'")
            _mlflow_active = True
            logger.info(
                "MLflow tracking enabled (experiment=%r)", config.mlflow_experiment
            )
        except ImportError:
            logger.warning(
                "mlflow is not installed; skipping tracking. pip install mlflow"
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "MLflow initialisation failed — continuing without tracking.",
                exc_info=True,
            )

    # Track already-processed ids to support resumption; reload their results
    # so that metrics can be computed over the full dataset even when resuming.
    processed_ids: set = set()
    results: dict = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    record_id = row["id"]
                    processed_ids.add(record_id)
                    results[record_id] = {k: v for k, v in row.items() if k != "id"}
                except (json.JSONDecodeError, KeyError):
                    pass
        if processed_ids:
            logger.info(
                "Resuming: skipping %d already-processed records", len(processed_ids)
            )
    record_step = len(processed_ids)

    # Instantiate estimator from config when not provided (convenience for CLI)
    if estimator is None:
        try:
            estimator = create_estimator_from_config(config.estimator_config)
            logger.info(
                "Instantiated estimator %r from config",
                config.estimator_config.model_name,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to instantiate estimator from config")
            raise

    logic_backend = create_logic_backend(config.logic_backend)
    logger.info("Using logic backend: %s", config.logic_backend.value)

    with open(output_path, "a", encoding="utf-8") as out_f:
        for record_id, record in data.items():
            if record_id in processed_ids:
                logger.debug("Skipping already-processed record %s", record_id)
                continue

            try:
                query: str = record["query"]
                atoms: list[ScoringInput] = record["atoms"]
                formula: ProblogFormula = record["problog_formula"]

                # --- Step 1: BM25 retrieval (top-N) ---
                top_n_results = retriever.retrieve(query, top_k=config.top_n).documents
                if not top_n_results:
                    logger.warning("No BM25 results for record %s", record_id)
                    results[record_id] = {
                        "query": query,
                        "retrieved_entities": [],
                        "retrieval_scores": {},
                        "ranked_entities": [],
                        "scores": {},
                        "num_atoms": len(atoms),
                        "reranked_pool_size": 0,
                        "metric_top_k": config.top_k,
                    }
                    write_result_row(out_f, record_id, results[record_id])
                    continue

                # --- Steps 2–3: LLM scoring + Problog evaluation ---
                entity_scores = score_candidate_documents(
                    atoms=atoms,
                    formula=formula,
                    candidate_documents=top_n_results,
                    estimator=estimator,
                    logic_backend=logic_backend,
                    include_retrieved_text=config.estimator_config.include_retrieved_text,
                    record_id=record_id,
                )

                # --- Step 4: Rerank to top-K ---
                relevant = None
                if ground_truth is not None and record_id in ground_truth:
                    relevant = set(ground_truth[record_id])

                record_result, ranked = build_record_result(
                    query=query,
                    atoms=atoms,
                    retrieved_documents=top_n_results,
                    entity_scores=entity_scores,
                    top_k=config.top_k,
                    relevant=relevant,
                )
                ranked_entities = record_result["ranked_entities"]
                retrieved_entities = record_result["retrieved_entities"]

                results[record_id] = record_result

                # --- Step 5: Checkpoint ---
                write_result_row(out_f, record_id, record_result)
                logger.info(
                    "Record %s done. Top entity: %r (%.4f)",
                    record_id,
                    ranked_entities[0] if ranked_entities else None,
                    ranked[0][1] if ranked else 0.0,
                )

                # --- MLflow per-record logging ---
                if _mlflow_active:
                    try:
                        top_score = ranked[0][1] if ranked else 0.0
                        avg_score = (
                            sum(entity_scores.values()) / len(entity_scores)
                            if entity_scores
                            else 0.0
                        )
                        # Step metrics for time-series charts
                        mlflow.log_metrics(
                            {
                                "top_entity_score": top_score,
                                "avg_candidate_score": avg_score,
                                "num_candidates": len(entity_scores),
                            },
                            step=record_step,
                        )
                        top_k_entities = ranked_entities[: config.top_k]
                        # Structured per-query table row
                        row: dict = {
                            "record_id": [record_id],
                            "logic_backend": config.logic_backend.value,
                            "query": [query],
                            "model": [config.estimator_config.model_name],
                            "ground_truth": [
                                ", ".join(record_result.get("ground_truth", []))
                            ],
                            "top_n_candidates": [", ".join(retrieved_entities)],
                            "top_k_results": [", ".join(top_k_entities)],
                        }
                        for metric_name, metric_val in record_result.get(
                            "record_metrics", {}
                        ).items():
                            row[metric_name] = [metric_val]
                        mlflow.log_table(data=row, artifact_file="query_results.json")
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "MLflow per-record logging failed — skipping.",
                            exc_info=True,
                        )

            except Exception:  # noqa: BLE001
                logger.exception(
                    "Unhandled error processing record %s — skipping", record_id
                )

            record_step += 1

    # --- Evaluation metrics ---
    metrics: dict[str, float] = {}
    if ground_truth is not None:
        metrics = compute_metrics(results, ground_truth, config.top_k)
        logger.info("=== Evaluation Metrics (K=%d) ===", config.top_k)
        for name, value in metrics.items():
            if name == "num_evaluated":
                logger.info("  %-30s %d", name, int(value))
            else:
                logger.info("  %-30s %.4f", name, value)

        metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as mf:
            json.dump(metrics, mf, indent=2, ensure_ascii=False)
        logger.info("Metrics saved to %s", metrics_path)

        if _mlflow_active:
            try:
                mlflow.log_metrics(metrics)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "MLflow metric logging failed — skipping.", exc_info=True
                )

    if _mlflow_active:
        try:
            mlflow.log_artifact(str(output_path))
            mlflow.end_run()
            logger.info("MLflow run ended. View with: mlflow ui")
        except Exception:  # noqa: BLE001
            logger.warning("MLflow end-run cleanup failed — skipping.", exc_info=True)

    return {"results": results, "metrics": metrics}
