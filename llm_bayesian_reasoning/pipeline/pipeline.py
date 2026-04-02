import json
import logging
from pathlib import Path

from problog import get_evaluatable
from problog.program import PrologString

from llm_bayesian_reasoning.estimators.base import BaseEstimator
from llm_bayesian_reasoning.estimators.factory import create_estimator_from_config
from llm_bayesian_reasoning.pipeline.config import PipelineConfig
from llm_bayesian_reasoning.pipeline.metrics import (
    compute_metrics,
    compute_record_metrics,
)
from llm_bayesian_reasoning.problog_models.problog_models import (
    ProblogAtom,
    ProblogFormula,
)
from llm_bayesian_reasoning.retrievers.retrievers import BM25Retriever

logger = logging.getLogger(__name__)


def build_or_load_index(
    documents_path: str | Path,
    index_path: str | Path,
    batch_size: int = 1000,
    limit: int | None = None,
) -> BM25Retriever:
    """Build a BM25 index from a JSONL document corpus, or load it if it already exists.

    Documents are streamed and appended in batches so that large corpora do not
    need to be fully loaded into memory at build time.  After all batches are
    appended the BM25 model is rebuilt once via ``finalize_index``.

    Args:
        documents_path: Path to a JSONL file where each line is
            ``{"title": str, "text": str}``.
        index_path: Directory in which to persist (or load) the index.
        batch_size: Number of documents processed per append call.
        limit: If set, only index the first ``limit`` documents. Useful for
            testing without building a full index.

    Returns:
        A loaded :class:`BM25Retriever` instance ready for retrieval.
    """
    index_dir = Path(index_path)
    retriever = BM25Retriever()

    if (index_dir / "bm25.pkl").exists():
        logger.info("Loading existing BM25 index from %s", index_dir)
        retriever.load_index(index_dir)
        return retriever

    logger.info(
        "Building BM25 index from %s into %s (batch_size=%d)",
        documents_path,
        index_dir,
        batch_size,
    )
    index_dir.mkdir(parents=True, exist_ok=True)

    entities_batch: list[str] = []
    titles_batch: list[str] = []
    total = 0

    with open(documents_path, encoding="utf-8") as f:
        for line in f:
            if limit is not None and total + len(entities_batch) >= limit:
                break
            doc = json.loads(line)
            entities_batch.append(doc["text"])
            titles_batch.append(doc["title"])

            if len(entities_batch) >= batch_size:
                retriever.append_batch(
                    entities=entities_batch,
                    index_path=index_dir,
                    titles=titles_batch,
                )
                total += len(entities_batch)
                logger.debug("Appended batch; total so far: %d", total)
                entities_batch = []
                titles_batch = []

    # Flush remaining docs
    if entities_batch:
        retriever.append_batch(
            entities=entities_batch,
            index_path=index_dir,
            titles=titles_batch,
        )
        total += len(entities_batch)

    logger.info("Finalizing index with %d documents", total)
    retriever.finalize_index(index_dir)
    retriever.load_index(index_dir)
    return retriever


def evaluate_problog(problog_str: str) -> float:
    """Evaluate a ProbLog program and return the probability of the query.

    Args:
        problog_str: A valid ProbLog program string containing exactly one
            ``query(...)`` directive.

    Returns:
        The probability of the query, or ``0.0`` if evaluation fails.
    """
    try:
        db = PrologString(problog_str)
        result = get_evaluatable().create_from(db).evaluate()
        # ``result`` is a dict mapping Term -> float; take the single value
        if result:
            return float(next(iter(result.values())))
        return 0.0
    except Exception:  # noqa: BLE001
        # Log full traceback and the program to help debugging parsing/evaluation issues
        logger.exception("Problog evaluation failed for program:\n%s", problog_str)
        return 0.0


def run_pipeline(
    data: dict,
    retriever: BM25Retriever,
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
    4. Sort entities by that probability (descending) and keep top-``top_k``.
    5. Append the result to ``output_path`` immediately (checkpointing).

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
            import mlflow  # lazy import — only required when tracking is enabled

            mlflow.set_experiment(config.mlflow_experiment)
            mlflow.start_run(nested=True)
            mlflow.log_params(
                {
                    "top_n": config.top_n,
                    "top_k": config.top_k,
                    "batch_size": config.batch_size,
                    "model_name": config.estimator_config.model_name,
                    "device": config.estimator_config.device,
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

    with open(output_path, "a", encoding="utf-8") as out_f:
        for record_id, record in data.items():
            if record_id in processed_ids:
                logger.debug("Skipping already-processed record %s", record_id)
                continue

            try:
                query: str = record["query"]
                atoms: list[ProblogAtom] | list[tuple[ProblogAtom, ProblogAtom]] = record["atoms"]
                formula: ProblogFormula = record["problog_formula"]

                # --- Step 1: BM25 retrieval (top-N) ---
                top_n_results = retriever.retrieve(query, top_k=config.top_n)
                if not top_n_results:
                    logger.warning("No BM25 results for record %s", record_id)
                    results[record_id] = {
                        "query": query,
                        "ranked_entities": [],
                        "scores": {},
                        "num_atoms": len(atoms),
                    }
                    _write_result(out_f, record_id, results[record_id])
                    continue

                # Resolve entity titles; fall back to entity text when no titles
                candidate_entities: list[str] = []
                for idx, _score in top_n_results:
                    if retriever.titles is not None:
                        candidate_entities.append(retriever.titles[idx])
                    else:
                        candidate_entities.append(retriever.entities[idx])

                # --- Steps 2–3: LLM scoring + Problog evaluation ---
                entity_scores: dict[str, float] = {}
                for entity in candidate_entities:
                    try:
                        scored_atoms = estimator.score_probability(atoms, entity)
                        problog_str = formula.to_problog(scored_atoms, entity)
                        prob = evaluate_problog(problog_str)
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "Scoring failed for record %s, entity %r — skipping entity",
                            record_id,
                            entity,
                        )
                        continue
                    entity_scores[entity] = prob
                    logger.debug("  entity=%r  prob=%.4f", entity, prob)

                # --- Step 4: Rerank to top-K ---
                ranked = sorted(
                    entity_scores.items(), key=lambda kv: kv[1], reverse=True
                )
                ranked_entities = [e for e, _ in ranked[: config.top_k]]

                record_result = {
                    "query": query,
                    "ranked_entities": ranked_entities,
                    "scores": {e: s for e, s in ranked[: config.top_k]},
                    "num_atoms": len(atoms),
                }

                # Per-record metrics (when ground truth is available)
                if ground_truth is not None and record_id in ground_truth:
                    relevant = set(ground_truth[record_id])
                    record_result["ground_truth"] = list(relevant)
                    record_result["record_metrics"] = compute_record_metrics(
                        ranked_entities, relevant, config.top_k
                    )

                results[record_id] = record_result

                # --- Step 5: Checkpoint ---
                _write_result(out_f, record_id, record_result)
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
                        # Structured per-query table row
                        row: dict = {
                            "record_id": [record_id],
                            "query": [query],
                            "model": [config.estimator_config.model_name],
                            "ground_truth": [
                                ", ".join(record_result.get("ground_truth", []))
                            ],
                            "top_n_candidates": [", ".join(candidate_entities)],
                            "top_k_results": [", ".join(ranked_entities)],
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


def _write_result(file_handle, record_id, record_result: dict) -> None:
    row = {"id": record_id, **record_result}
    file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    file_handle.flush()
