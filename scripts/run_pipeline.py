#!/usr/bin/env python3
"""Cloud-friendly CLI to run the retrieval + scoring pipeline.

Creates a timestamped results folder under `llm_bayesian_reasoning/results/` by default
and checkpoints progress to a JSONL file so runs can be resumed.

Example:
  python scripts/run_pipeline.py --data-file llm_bayesian_reasoning/data/preprocessed_data/parsed_test.jsonl

"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

# Using Python 3.12 native generic types: `list`, `dict`, `tuple`, `| None`
from llm_bayesian_reasoning.pipeline.config import (
    EstimatorConfig,
    EstimatorType,
    LogicBackendType,
    PipelineConfig,
    RetrieverType,
)
from llm_bayesian_reasoning.pipeline.record_loader import (
    load_preprocessed_records,
    materialize_records_for_estimator,
)
from llm_bayesian_reasoning.pipeline.pipeline import run_pipeline
from llm_bayesian_reasoning.retrievers.base_retriever import BaseRetriever
from llm_bayesian_reasoning.retrievers.factory import build_or_load_retriever

logger = logging.getLogger("run_pipeline")


def _make_results_folder(
    root: Path,
    model_name: str,
    retriever_type: RetrieverType,
    estimator_type: EstimatorType,
    logic_backend: LogicBackendType,
) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_model = Path(model_name).stem.replace("/", "_")
    run_name = (
        f"{ts}_{retriever_type.value}_{safe_model}_"
        f"{estimator_type.value}_{logic_backend.value}"
    )
    out = root / run_name
    out.mkdir(parents=True, exist_ok=False)
    return out


def main():
    p = argparse.ArgumentParser(
        description="Run retrieval + ProbLog scoring pipeline (cloud-friendly CLI)"
    )

    p.add_argument(
        "--data-file",
        type=Path,
        default=Path("llm_bayesian_reasoning/data/preprocessed_data/parsed_test.jsonl"),
    )
    p.add_argument(
        "--index-path",
        type=Path,
        default=Path("llm_bayesian_reasoning/data/index_data/bm25_index"),
    )
    p.add_argument(
        "--index-documents",
        type=Path,
        default=Path("llm_bayesian_reasoning/data/index_data/documents.jsonl"),
    )
    p.add_argument(
        "--results-root", type=Path, default=Path("llm_bayesian_reasoning/results")
    )

    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1000)

    p.add_argument("--model-name", type=str, default=None)
    p.add_argument(
        "--estimator-type",
        type=str,
        choices=[e.value for e in EstimatorType],
        default=EstimatorType.TRUE_FALSE_LLM.value,
    )
    p.add_argument(
        "--logic-backend",
        type=str,
        choices=[e.value for e in LogicBackendType],
        default=LogicBackendType.PROBLOG.value,
    )
    p.add_argument(
        "--retriever-type",
        type=str,
        choices=[e.value for e in RetrieverType],
        default=RetrieverType.BM25.value,
    )
    p.add_argument(
        "--retriever-model-name",
        type=str,
        default="intfloat/e5-base-v2",
        help="Dense retriever model identifier for retrievers that need one",
    )
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--mlflow-experiment", type=str, default=None)
    p.add_argument(
        "--include-retrieved-text",
        action="store_true",
        help="Include retrieved document text in estimator prompt context",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of input records (useful for smoke tests)",
    )
    p.add_argument(
        "--index-limit",
        type=int,
        default=None,
        help="Limit documents indexed when building BM25",
    )
    p.add_argument(
        "--use-metadata-ground-truth",
        action="store_true",
        help="Extract ground-truth from metadata.relevance_ratings if present",
    )

    p.add_argument("--debug", action="store_true")

    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Prepare results folder
    results_root = args.results_root
    results_root.mkdir(parents=True, exist_ok=True)

    # Prepare estimator config
    estimator_cfg: EstimatorConfig
    if args.model_name is not None:
        estimator_cfg = EstimatorConfig(
            model_name=args.model_name,
            device=args.device,
            estimator_type=EstimatorType(args.estimator_type),
            include_retrieved_text=args.include_retrieved_text,
        )
    else:
        estimator_cfg = EstimatorConfig(
            estimator_type=EstimatorType(args.estimator_type),
            device=args.device,
            include_retrieved_text=args.include_retrieved_text,
        )

    # Create a temporary PipelineConfig to determine run naming
    pipe_cfg = PipelineConfig(
        top_n=args.top_n,
        top_k=args.top_k,
        batch_size=args.batch_size,
        index_path=args.index_path,
        output_path=Path("/tmp/dummy.jsonl"),
        estimator_config=estimator_cfg,
        retriever_type=RetrieverType(args.retriever_type),
        retriever_model_name=args.retriever_model_name,
        logic_backend=LogicBackendType(args.logic_backend),
        mlflow_experiment=args.mlflow_experiment,
    )

    try:
        results_dir = _make_results_folder(
            results_root,
            estimator_cfg.model_name,
            pipe_cfg.retriever_type,
            estimator_cfg.estimator_type,
            pipe_cfg.logic_backend,
        )
    except FileExistsError:
        # Unlikely because of timestamp but handle gracefully
        results_dir = results_root / (
            datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_run"
        )
        results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "results.jsonl"

    # Save chosen config for reproducibility
    cfg_out = results_dir / "pipeline_config.json"
    with cfg_out.open("w", encoding="utf-8") as cf:
        json.dump(
            {
                "top_n": args.top_n,
                "top_k": args.top_k,
                "batch_size": args.batch_size,
                "index_path": str(args.index_path),
                "index_documents": str(args.index_documents),
                "retriever_type": args.retriever_type,
                "retriever_model_name": args.retriever_model_name,
                "model_name": estimator_cfg.model_name,
                "estimator_type": estimator_cfg.estimator_type.value,
                "logic_backend": args.logic_backend,
                "device": estimator_cfg.device,
                "include_retrieved_text": estimator_cfg.include_retrieved_text,
                "mlflow_experiment": args.mlflow_experiment,
            },
            cf,
            indent=2,
        )

    # (Re-)build or load retriever index
    logger.info(
        "Loading or building %s index (index_path=%s)",
        args.retriever_type,
        args.index_path,
    )
    retriever: BaseRetriever = build_or_load_retriever(
        documents_path=args.index_documents,
        index_path=args.index_path,
        retriever_type=RetrieverType(args.retriever_type),
        batch_size=args.batch_size,
        limit=args.index_limit,
        retriever_model_name=args.retriever_model_name,
    )

    # Load data
    logger.info("Loading preprocessed data from %s", args.data_file)
    loaded_records, ground_truth = load_preprocessed_records(
        args.data_file,
        limit=args.limit,
        use_metadata_ground_truth=args.use_metadata_ground_truth,
    )
    data = materialize_records_for_estimator(
        loaded_records,
        estimator_cfg.estimator_type,
    )
    if args.limit is not None:
        logger.info("Loaded %d records (limited)", len(data))
    else:
        logger.info("Loaded %d records", len(data))

    # Final pipeline config with real output path
    final_cfg = PipelineConfig(
        top_n=args.top_n,
        top_k=args.top_k,
        batch_size=args.batch_size,
        index_path=args.index_path,
        output_path=output_path,
        estimator_config=pipe_cfg.estimator_config,
        retriever_type=pipe_cfg.retriever_type,
        retriever_model_name=pipe_cfg.retriever_model_name,
        logic_backend=pipe_cfg.logic_backend,
        mlflow_experiment=args.mlflow_experiment,
    )

    # Run pipeline (estimator will be instantiated from config inside run_pipeline)
    logger.info("Starting pipeline; results will be written to %s", output_path)
    result = run_pipeline(
        data=data,
        retriever=retriever,
        estimator=None,
        config=final_cfg,
        ground_truth=ground_truth,
    )

    logger.info("Pipeline finished. Metrics: %s", result.get("metrics"))
    logger.info("Results JSONL: %s", output_path)
    logger.info("Results folder: %s", results_dir)


if __name__ == "__main__":
    main()
