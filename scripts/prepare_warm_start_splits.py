#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path

from llm_bayesian_reasoning.data.deepproblog_dataset import (
    DeepProbLogGroupedExample,
    group_deepproblog_rows,
    read_deepproblog_rows,
    rows_from_grouped_examples,
    select_grouped_example_subset,
    split_grouped_examples,
)

logger = logging.getLogger("prepare_warm_start_splits")


def _write_jsonl(path: Path, grouped_examples: list[DeepProbLogGroupedExample]) -> int:
    rows = rows_from_grouped_examples(grouped_examples)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")
    return len(rows)


def _split_stage_train_pool(
    train_pool: list[DeepProbLogGroupedExample],
    seed: int,
    stage2_queries: int | None,
    stage2_fraction: float | None,
) -> tuple[list[DeepProbLogGroupedExample], list[DeepProbLogGroupedExample]]:
    if not train_pool:
        raise ValueError("Training pool is empty; nothing to split")
    if len(train_pool) < 2:
        raise ValueError(
            "Training pool must contain at least 2 grouped queries so both stage1 and stage2 are non-empty"
        )

    if stage2_queries is not None and stage2_fraction is not None:
        raise ValueError("Use either --stage2-queries or --stage2-fraction, not both")

    if stage2_fraction is not None:
        if not 0.0 < stage2_fraction < 1.0:
            raise ValueError("stage2_fraction must be strictly between 0 and 1")
        stage1_examples, stage2_examples, _ = split_grouped_examples(
            train_pool,
            train_fraction=1.0 - stage2_fraction,
            val_fraction=stage2_fraction,
            test_fraction=0.0,
            seed=seed,
        )
        return stage1_examples, stage2_examples

    effective_stage2_queries = stage2_queries if stage2_queries is not None else 5000
    if effective_stage2_queries < 1:
        raise ValueError("stage2_queries must be >= 1")
    if effective_stage2_queries >= len(train_pool):
        capped_stage2_queries = len(train_pool) - 1
        logger.warning(
            "Requested stage2_queries=%d but the global training pool only contains %d grouped queries; capping stage2_queries to %d to keep stage1 non-empty",
            effective_stage2_queries,
            len(train_pool),
            capped_stage2_queries,
        )
        effective_stage2_queries = capped_stage2_queries

    stage2_examples = select_grouped_example_subset(
        train_pool,
        limit=effective_stage2_queries,
        seed=seed,
    )
    stage2_ids = {example.id for example in stage2_examples}
    stage1_examples = [
        example for example in train_pool if example.id not in stage2_ids
    ]
    if not stage1_examples:
        raise ValueError(
            "stage1 training split is empty after selecting stage2 queries"
        )
    return stage1_examples, stage2_examples


def _split_summary(grouped_examples: list[DeepProbLogGroupedExample]) -> dict[str, int]:
    positive_candidates = 0
    total_candidates = 0
    for example in grouped_examples:
        total_candidates += len(example.candidates)
        positive_candidates += sum(
            int(candidate.relevance > 0) for candidate in example.candidates
        )
    return {
        "queries": len(grouped_examples),
        "candidate_rows": total_candidates,
        "positive_candidate_rows": positive_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create disjoint query-level splits for warm-start training: "
            "stage1_train, stage2_train, validation, and test"
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("llm_bayesian_reasoning/data/preprocessed_data/dpppl.jsonl"),
        help="DeepProbLog-style JSONL source dataset",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional limit on grouped queries before global splitting",
    )
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument(
        "--stage2-queries",
        type=int,
        default=None,
        help="Number of grouped queries to reserve for symbolic warm-start fine-tuning",
    )
    parser.add_argument(
        "--stage2-fraction",
        type=float,
        default=None,
        help="Alternative to --stage2-queries; fraction of the global training pool used for symbolic fine-tuning",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    grouped_examples = group_deepproblog_rows(read_deepproblog_rows(args.data_path))
    grouped_examples = select_grouped_example_subset(
        grouped_examples,
        limit=args.max_queries,
        seed=args.seed,
    )
    train_pool, validation_examples, test_examples = split_grouped_examples(
        grouped_examples,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    stage1_examples, stage2_examples = _split_stage_train_pool(
        train_pool,
        seed=args.seed + 1,
        stage2_queries=args.stage2_queries,
        stage2_fraction=args.stage2_fraction,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_path = output_dir / "stage1_train.jsonl"
    stage2_path = output_dir / "stage2_symbolic_train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    test_path = output_dir / "test.jsonl"
    manifest_path = output_dir / "warm_start_split_manifest.json"

    stage1_rows = _write_jsonl(stage1_path, stage1_examples)
    stage2_rows = _write_jsonl(stage2_path, stage2_examples)
    validation_rows = _write_jsonl(validation_path, validation_examples)
    test_rows = _write_jsonl(test_path, test_examples)

    manifest = {
        "source_data_path": str(args.data_path),
        "seed": args.seed,
        "max_queries": args.max_queries,
        "global_split": {
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
        },
        "stage2_policy": {
            "stage2_queries": args.stage2_queries,
            "stage2_fraction": args.stage2_fraction,
        },
        "paths": {
            "stage1_train": str(stage1_path),
            "stage2_symbolic_train": str(stage2_path),
            "validation": str(validation_path),
            "test": str(test_path),
        },
        "counts": {
            "stage1_train": {
                **_split_summary(stage1_examples),
                "written_rows": stage1_rows,
            },
            "stage2_symbolic_train": {
                **_split_summary(stage2_examples),
                "written_rows": stage2_rows,
            },
            "validation": {
                **_split_summary(validation_examples),
                "written_rows": validation_rows,
            },
            "test": {
                **_split_summary(test_examples),
                "written_rows": test_rows,
            },
        },
        "example_usage": {
            "atom_only": [
                "python3 scripts/train_dpl_atom_only.py",
                f"--train-data-path {stage1_path}",
                f"--val-data-path {validation_path}",
                f"--test-data-path {test_path}",
                "--output-dir llm_bayesian_reasoning/results/dpl_atom_only",
            ],
            "symbolic": [
                "Set train_data_path/val_data_path/test_data_path in your train_dpl_pipeline config",
                f"train_data_path={stage2_path}",
                f"val_data_path={validation_path}",
                f"test_data_path={test_path}",
            ],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Wrote warm-start split bundle to %s", output_dir)
    logger.info("stage1_train queries=%d rows=%d", len(stage1_examples), stage1_rows)
    logger.info(
        "stage2_symbolic_train queries=%d rows=%d", len(stage2_examples), stage2_rows
    )
    logger.info(
        "validation queries=%d rows=%d", len(validation_examples), validation_rows
    )
    logger.info("test queries=%d rows=%d", len(test_examples), test_rows)


if __name__ == "__main__":
    main()
