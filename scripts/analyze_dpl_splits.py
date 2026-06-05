#!/usr/bin/env python3

import argparse
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path

from llm_bayesian_reasoning.data.deepproblog_dataset import (
    DeepProbLogCandidate,
    DeepProbLogGroupedExample,
    group_deepproblog_rows,
    read_deepproblog_rows,
    select_grouped_example_subset,
    split_grouped_examples,
)

logger = logging.getLogger("analyze_dpl_splits")

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _normalize_text(value: str) -> str:
    normalized = value.replace("{x}", "{X}")
    normalized = normalized.replace("{X}", "")
    normalized = normalized.replace("'", "")
    normalized = normalized.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(_normalize_text(text)))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _quantile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = position - lower
    return float(lower_value + (upper_value - lower_value) * weight)


def _summarize_numeric(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(float(value) for value in values)
    return {
        "mean": _safe_mean(sorted_values),
        "median": _quantile(sorted_values, 0.5),
        "p90": _quantile(sorted_values, 0.9),
        "min": float(sorted_values[0]) if sorted_values else float("nan"),
        "max": float(sorted_values[-1]) if sorted_values else float("nan"),
    }


def _summarize_counter(counter: Counter[str], top_n: int = 10) -> dict[str, object]:
    total = sum(counter.values())
    top_items = [
        {
            "label": label,
            "count": count,
            "share": _safe_ratio(count, total),
        }
        for label, count in counter.most_common(top_n)
    ]
    entropy = 0.0
    if total > 0:
        for count in counter.values():
            probability = count / total
            entropy -= probability * math.log2(probability)
    return {
        "unique": len(counter),
        "top_1_share": top_items[0]["share"] if top_items else float("nan"),
        "entropy": float(entropy),
        "top_items": top_items,
    }


def _candidate_query_overlap(query: str, candidate: DeepProbLogCandidate) -> float:
    query_tokens = _tokenize(query)
    candidate_tokens = _tokenize(candidate.text)
    if not query_tokens:
        return float("nan")
    return _safe_ratio(len(query_tokens & candidate_tokens), len(query_tokens))


def _attributed_atom_fraction(
    example: DeepProbLogGroupedExample,
    candidate: DeepProbLogCandidate,
) -> float:
    if not example.atoms:
        return float("nan")
    attribution_keys: list[str] = []
    for attribution in candidate.attributions or []:
        for key in attribution:
            if isinstance(key, str):
                attribution_keys.append(_normalize_text(key))
    if not attribution_keys:
        return 0.0

    attributed_atoms = 0
    for atom in example.atoms:
        atom_norm = _normalize_text(atom)
        if any(
            atom_norm == key or atom_norm in key or key in atom_norm
            for key in attribution_keys
        ):
            attributed_atoms += 1
    return _safe_ratio(attributed_atoms, len(example.atoms))


def _evidence_bucket_flags(candidate: DeepProbLogCandidate) -> tuple[bool, bool]:
    evidence_strings = [
        _normalize_text(value)
        for value in (candidate.evidence_ratings or [])
        if isinstance(value, str)
    ]
    has_partial = any("partial" in value for value in evidence_strings)
    has_complete = any("complete" in value for value in evidence_strings)
    return has_partial, has_complete


def _summarize_split(
    grouped_examples: list[DeepProbLogGroupedExample],
) -> dict[str, object]:
    query_lengths = [len(_tokenize(example.query)) for example in grouped_examples]
    atoms_per_query = [len(example.atoms) for example in grouped_examples]
    candidates_per_query = [len(example.candidates) for example in grouped_examples]
    template_counts = Counter(
        (example.template or "unknown").strip() or "unknown"
        for example in grouped_examples
    )
    domain_counts = Counter(
        (example.domain or "unknown").strip() or "unknown"
        for example in grouped_examples
    )

    total_candidate_rows = 0
    positive_candidate_rows = 0
    positive_target_means: list[float] = []
    positive_query_overlap: list[float] = []
    negative_query_overlap: list[float] = []
    positive_has_evidence = 0
    positive_has_attribution = 0
    positive_partial_evidence = 0
    positive_complete_evidence = 0
    positive_attributed_atom_fraction: list[float] = []

    for example in grouped_examples:
        for candidate in example.candidates:
            total_candidate_rows += 1
            overlap = _candidate_query_overlap(example.query, candidate)
            if candidate.relevance > 0:
                positive_candidate_rows += 1
                positive_query_overlap.append(overlap)
                if candidate.atom_targets:
                    positive_target_means.append(_safe_mean(candidate.atom_targets))
                if candidate.evidence_ratings:
                    positive_has_evidence += 1
                if candidate.attributions:
                    positive_has_attribution += 1
                has_partial, has_complete = _evidence_bucket_flags(candidate)
                if has_partial:
                    positive_partial_evidence += 1
                if has_complete:
                    positive_complete_evidence += 1
                positive_attributed_atom_fraction.append(
                    _attributed_atom_fraction(example, candidate)
                )
            else:
                negative_query_overlap.append(overlap)

    return {
        "queries": len(grouped_examples),
        "candidate_rows": total_candidate_rows,
        "positive_candidate_rows": positive_candidate_rows,
        "positive_rate": _safe_ratio(positive_candidate_rows, total_candidate_rows),
        "atoms_per_query": _summarize_numeric(atoms_per_query),
        "candidates_per_query": _summarize_numeric(candidates_per_query),
        "query_length_tokens": _summarize_numeric(query_lengths),
        "template_distribution": _summarize_counter(template_counts),
        "domain_distribution": _summarize_counter(domain_counts),
        "difficulty_proxies": {
            "positive_query_text_overlap": _summarize_numeric(positive_query_overlap),
            "negative_query_text_overlap": _summarize_numeric(negative_query_overlap),
        },
        "weak_label_quality": {
            "positive_has_evidence_rate": _safe_ratio(
                positive_has_evidence, positive_candidate_rows
            ),
            "positive_has_attribution_rate": _safe_ratio(
                positive_has_attribution, positive_candidate_rows
            ),
            "positive_partial_evidence_rate": _safe_ratio(
                positive_partial_evidence, positive_candidate_rows
            ),
            "positive_complete_evidence_rate": _safe_ratio(
                positive_complete_evidence, positive_candidate_rows
            ),
            "positive_mean_atom_target": _safe_mean(positive_target_means),
            "positive_attributed_atom_fraction": _summarize_numeric(
                positive_attributed_atom_fraction
            ),
        },
    }


def _safe_delta(left: float, right: float) -> float:
    if math.isnan(left) or math.isnan(right):
        return float("nan")
    return float(left - right)


def _build_alerts(report: dict[str, object]) -> list[str]:
    split_summaries = report["split_summaries"]
    train_summary = split_summaries.get("train")
    validation_summary = split_summaries.get("validation")
    if train_summary is None or validation_summary is None:
        return []

    alerts: list[str] = []

    positive_rate_delta = _safe_delta(
        train_summary["positive_rate"],
        validation_summary["positive_rate"],
    )
    if not math.isnan(positive_rate_delta) and abs(positive_rate_delta) >= 0.05:
        alerts.append(
            f"Positive rate differs materially between train and validation ({positive_rate_delta:+.3f})."
        )

    train_overlap = train_summary["difficulty_proxies"]["positive_query_text_overlap"][
        "mean"
    ]
    val_overlap = validation_summary["difficulty_proxies"][
        "positive_query_text_overlap"
    ]["mean"]
    overlap_delta = _safe_delta(train_overlap, val_overlap)
    if not math.isnan(overlap_delta) and overlap_delta >= 0.05:
        alerts.append(
            "Train positives have noticeably higher query-text lexical overlap than validation positives; validation may be harder or less lexically direct."
        )

    train_template_top1 = train_summary["template_distribution"]["top_1_share"]
    val_template_top1 = validation_summary["template_distribution"]["top_1_share"]
    template_delta = _safe_delta(train_template_top1, val_template_top1)
    if not math.isnan(template_delta) and template_delta >= 0.10:
        alerts.append(
            "Train template distribution is more concentrated than validation, which can make train easier and more repetitive."
        )

    train_evidence = train_summary["weak_label_quality"]["positive_has_evidence_rate"]
    val_evidence = validation_summary["weak_label_quality"][
        "positive_has_evidence_rate"
    ]
    evidence_delta = _safe_delta(train_evidence, val_evidence)
    if not math.isnan(evidence_delta) and evidence_delta >= 0.05:
        alerts.append(
            "Validation positives have weaker evidence coverage than train positives."
        )

    train_attr = train_summary["weak_label_quality"]["positive_has_attribution_rate"]
    val_attr = validation_summary["weak_label_quality"]["positive_has_attribution_rate"]
    attr_delta = _safe_delta(train_attr, val_attr)
    if not math.isnan(attr_delta) and attr_delta >= 0.05:
        alerts.append(
            "Validation positives have less attribution support than train positives."
        )

    train_target = train_summary["weak_label_quality"]["positive_mean_atom_target"]
    val_target = validation_summary["weak_label_quality"]["positive_mean_atom_target"]
    target_delta = _safe_delta(train_target, val_target)
    if not math.isnan(target_delta) and target_delta >= 0.05:
        alerts.append(
            "Validation weak labels are softer on average than train weak labels."
        )

    return alerts


def _load_splits_from_source(
    data_path: Path,
    max_queries: int | None,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[DeepProbLogGroupedExample]]:
    grouped_examples = group_deepproblog_rows(read_deepproblog_rows(data_path))
    grouped_examples = select_grouped_example_subset(
        grouped_examples,
        limit=max_queries,
        seed=seed,
    )
    train_examples, validation_examples, test_examples = split_grouped_examples(
        grouped_examples,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    return {
        "train": train_examples,
        "validation": validation_examples,
        "test": test_examples,
    }


def _load_grouped_examples(path: Path) -> list[DeepProbLogGroupedExample]:
    return group_deepproblog_rows(read_deepproblog_rows(path))


def _load_explicit_splits(
    train_data_path: Path,
    val_data_path: Path,
    test_data_path: Path | None,
) -> dict[str, list[DeepProbLogGroupedExample]]:
    split_payload = {
        "train": _load_grouped_examples(train_data_path),
        "validation": _load_grouped_examples(val_data_path),
    }
    if test_data_path is not None:
        split_payload["test"] = _load_grouped_examples(test_data_path)
    return split_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze query-level train/validation/test splits for class balance, "
            "difficulty proxies, and weak-label quality"
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="DeepProbLog-style JSONL source dataset used to create internal splits",
    )
    parser.add_argument("--train-data-path", type=Path, default=None)
    parser.add_argument("--val-data-path", type=Path, default=None)
    parser.add_argument("--test-data-path", type=Path, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    using_explicit = any(
        path is not None
        for path in (args.train_data_path, args.val_data_path, args.test_data_path)
    )
    if using_explicit:
        if args.train_data_path is None or args.val_data_path is None:
            raise ValueError(
                "When using explicit split files, provide --train-data-path and --val-data-path"
            )
        split_examples = _load_explicit_splits(
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            test_data_path=args.test_data_path,
        )
        split_source = "explicit_split_files"
    else:
        if args.data_path is None:
            args.data_path = Path(
                "llm_bayesian_reasoning/data/preprocessed_data/dpppl.jsonl"
            )
        split_examples = _load_splits_from_source(
            data_path=args.data_path,
            max_queries=args.max_queries,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        split_source = "internal_query_split"

    split_summaries = {
        split_name: _summarize_split(examples)
        for split_name, examples in split_examples.items()
    }
    report = {
        "split_source": split_source,
        "input": {
            "data_path": None if args.data_path is None else str(args.data_path),
            "train_data_path": (
                None if args.train_data_path is None else str(args.train_data_path)
            ),
            "val_data_path": (
                None if args.val_data_path is None else str(args.val_data_path)
            ),
            "test_data_path": (
                None if args.test_data_path is None else str(args.test_data_path)
            ),
            "max_queries": args.max_queries,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "seed": args.seed,
        },
        "split_summaries": split_summaries,
    }
    report["alerts"] = _build_alerts(report)

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(rendered + "\n", encoding="utf-8")
        logger.info("Wrote split diagnostics to %s", args.output_path)


if __name__ == "__main__":
    main()
