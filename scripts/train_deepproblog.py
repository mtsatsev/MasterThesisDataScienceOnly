#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from llm_bayesian_reasoning.data.deepproblog_dataset import (
    group_deepproblog_rows,
    read_deepproblog_rows,
)
from llm_bayesian_reasoning.training.deepproblog_module import (
    AtomClassifier,
    DeepProbLogModelConfig,
    load_tokenizer,
    run_deepproblog_training,
    save_model_bundle,
    train_atom_classifier,
)

logger = logging.getLogger("train_deepproblog")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DeepProbLog atom scorer")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(
            "llm_bayesian_reasoning/data/preprocessed_data/parsed_test_with_negs_dpp_smoke.jsonl"
        ),
        help="DeepProbLog row dataset produced by build_dpp_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the trained model bundle will be written",
    )
    parser.add_argument("--model-name", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stage1-epochs", type=int, default=1)
    parser.add_argument("--stage2-epochs", type=int, default=1)
    parser.add_argument("--stage1-batch-size", type=int, default=8)
    parser.add_argument("--stage2-batch-size", type=int, default=4)
    parser.add_argument("--stage1-learning-rate", type=float, default=2e-5)
    parser.add_argument("--stage2-learning-rate", type=float, default=1e-5)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rows = read_deepproblog_rows(args.dataset_path, limit=args.limit)
    grouped_examples = group_deepproblog_rows(rows)
    logger.info("Loaded %d rows grouped into %d queries", len(rows), len(grouped_examples))

    model_config = DeepProbLogModelConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        dropout=args.dropout,
    )
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
        batch_size=args.stage1_batch_size,
        epochs=args.stage1_epochs,
        learning_rate=args.stage1_learning_rate,
    )
    logger.info("Stage-1 losses: %s", stage1_summary.losses)

    stage2_summary = run_deepproblog_training(
        grouped_examples=grouped_examples,
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        device=args.device,
        batch_size=args.stage2_batch_size,
        epochs=args.stage2_epochs,
        learning_rate=args.stage2_learning_rate,
    )
    logger.info("Stage-2 losses: %s", stage2_summary.losses)

    output_dir = save_model_bundle(
        output_dir=args.output_dir,
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        metadata={
            "dataset_path": str(args.dataset_path),
            "num_rows": len(rows),
            "num_queries": len(grouped_examples),
            "stage1_losses": stage1_summary.losses,
            "stage2_losses": stage2_summary.losses,
        },
    )

    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "stage1_losses": stage1_summary.losses,
                "stage2_losses": stage2_summary.losses,
                "program_preview": "\n".join(stage2_summary.program.splitlines()[:12]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote model bundle to %s", output_dir)
    logger.info("Wrote training summary to %s", summary_path)


if __name__ == "__main__":
    main()
