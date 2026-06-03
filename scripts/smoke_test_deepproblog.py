#!/usr/bin/env python3

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from llm_bayesian_reasoning.data.deepproblog_dataset import (
    group_deepproblog_rows,
    read_deepproblog_rows,
)
from llm_bayesian_reasoning.pipeline.config import (
    EstimatorConfig,
    EstimatorType,
    LogicBackendType,
    PipelineConfig,
    RetrieverType,
)
from llm_bayesian_reasoning.pipeline.pipeline import run_pipeline
from llm_bayesian_reasoning.retrievers.factory import build_or_load_retriever
from llm_bayesian_reasoning.training.deepproblog_module import (
    AtomClassifier,
    DeepProbLogModelConfig,
    load_tokenizer,
    run_deepproblog_training,
    save_model_bundle,
    train_atom_classifier,
)
from scripts.run_pipeline import _load_preprocessed

logger = logging.getLogger("smoke_test_deepproblog")


def _make_smoke_dir(root: Path) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = root / f"deepproblog_smoke_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and smoke-test DeepProbLog reranking"
    )
    parser.add_argument(
        "--training-dataset",
        type=Path,
        default=Path(
            "llm_bayesian_reasoning/data/preprocessed_data/parsed_test_with_negs_dpp_smoke.jsonl"
        ),
    )
    parser.add_argument(
        "--pipeline-data-file",
        type=Path,
        default=Path(
            "llm_bayesian_reasoning/data/preprocessed_data/parsed_test_with_negs.jsonl"
        ),
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("index/bm25_cli_smoke"),
    )
    parser.add_argument(
        "--index-documents",
        type=Path,
        default=Path("llm_bayesian_reasoning/data/index_data/documents.jsonl"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("llm_bayesian_reasoning/results"),
    )
    parser.add_argument("--model-name", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--training-limit", type=int, default=8)
    parser.add_argument("--pipeline-limit", type=int, default=1)
    parser.add_argument("--stage1-epochs", type=int, default=1)
    parser.add_argument("--stage2-epochs", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    smoke_dir = _make_smoke_dir(args.results_root)
    model_dir = smoke_dir / "model"
    results_path = smoke_dir / "rerank_results.jsonl"

    rows = read_deepproblog_rows(args.training_dataset, limit=args.training_limit)
    grouped_examples = group_deepproblog_rows(rows)
    logger.info(
        "Loaded %d training rows across %d grouped queries",
        len(rows),
        len(grouped_examples),
    )

    model_config = DeepProbLogModelConfig(model_name=args.model_name)
    tokenizer = load_tokenizer(model_config.model_name)
    model = AtomClassifier(
        model_name=model_config.model_name,
        dropout=model_config.dropout,
    )

    stage1_summary = train_atom_classifier(
        grouped_examples=grouped_examples,
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        device=args.device,
        epochs=args.stage1_epochs,
        batch_size=4,
        learning_rate=2e-5,
    )
    stage2_summary = run_deepproblog_training(
        grouped_examples=grouped_examples,
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        device=args.device,
        epochs=args.stage2_epochs,
        batch_size=2,
        learning_rate=1e-5,
    )

    save_model_bundle(
        output_dir=model_dir,
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        metadata={
            "stage1_losses": stage1_summary.losses,
            "stage2_losses": stage2_summary.losses,
        },
    )

    estimator_config = EstimatorConfig(
        model_name=args.model_name,
        estimator_type=EstimatorType.DEEP_PROBLOG,
        device=args.device,
        include_retrieved_text=True,
        deepproblog_model_dir=model_dir,
    )
    pipeline_config = PipelineConfig(
        top_n=5,
        top_k=3,
        batch_size=1000,
        index_path=args.index_path,
        output_path=results_path,
        estimator_config=estimator_config,
        retriever_type=RetrieverType.BM25,
        logic_backend=LogicBackendType.DEEPPROBLOG,
    )

    retriever = build_or_load_retriever(
        documents_path=args.index_documents,
        index_path=args.index_path,
        retriever_type=RetrieverType.BM25,
        batch_size=1000,
    )
    data, _ground_truth = _load_preprocessed(
        args.pipeline_data_file,
        estimator_type=EstimatorType.DEEP_PROBLOG,
        limit=args.pipeline_limit,
    )
    result = run_pipeline(
        data=data,
        retriever=retriever,
        estimator=None,
        config=pipeline_config,
        ground_truth=None,
    )

    if not results_path.exists():
        raise RuntimeError(f"Expected smoke-test results at {results_path}")

    lines = results_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise RuntimeError("Smoke test completed but produced no reranked records")

    summary = {
        "model_dir": str(model_dir),
        "results_path": str(results_path),
        "stage1_losses": stage1_summary.losses,
        "stage2_losses": stage2_summary.losses,
        "num_result_rows": len(lines),
        "metrics": result.get("metrics"),
    }
    (smoke_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    logger.info("Smoke summary: %s", summary)


if __name__ == "__main__":
    main()
