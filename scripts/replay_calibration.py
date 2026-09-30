#!/usr/bin/env python3
"""Re-rank saved pipeline results under an atom-level calibration, without the LLM.

Reads a results JSONL that contains ``atom_scores`` and recomputes each entity's
ProbLog probability after shifting every atom logit by a per-atom prior:

- ``none``: no shift; reproduces the stored ranking (sanity check).
- ``contextual``: subtract the logit of the atom scored on the content-free
  input "N/A" (contextual calibration, Zhao et al., 2021).
- ``pool``: subtract the atom's mean logit over the query's candidate pool.

Example:
    python scripts/replay_calibration.py results.jsonl \\
        llm_bayesian_reasoning/data/preprocessed_data/parsed_test_with_negs.jsonl \\
        --method contextual
"""

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

from llm_bayesian_reasoning.pipeline.logic_backends import ProbLogBackend
from llm_bayesian_reasoning.pipeline.metrics import compute_metrics
from llm_bayesian_reasoning.pipeline.pipeline import CONTENT_FREE_KEY
from llm_bayesian_reasoning.problog_models.problog_models import (
    ProblogAtom,
    ProblogFormula,
)

EPS = 1e-12


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def _atom_priors(atom_scores: dict, entities: list[str], method: str) -> dict:
    atoms = atom_scores[entities[0]].keys()
    if method == "contextual":
        return {a: _logit(atom_scores[CONTENT_FREE_KEY][a]) for a in atoms}
    if method == "pool":
        return {
            a: sum(_logit(atom_scores[e][a]) for e in entities) / len(entities)
            for a in atoms
        }
    return {a: 0.0 for a in atoms}


def _replay_record(args: tuple[dict, str, str]) -> tuple:
    row, formula_text, method = args
    atom_scores = row["atom_scores"]
    entities = [e for e in row["retrieved_entities"] if e in atom_scores]
    if not entities:
        return row["id"], row
    formula = ProblogFormula(formula=formula_text)
    priors = _atom_priors(atom_scores, entities, method)
    backend = ProbLogBackend()
    scores = {}
    for entity in entities:
        atoms = [
            ProblogAtom(
                atom=a,
                probability=1 / (1 + math.exp(-(_logit(p) - priors[a]))),
            )
            for a, p in atom_scores[entity].items()
        ]
        scores[entity] = backend.evaluate(atoms, formula, entity)
    ranked = sorted(entities, key=lambda e: scores[e], reverse=True)
    return row["id"], {**row, "ranked_entities": ranked, "scores": scores}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="Pipeline results JSONL")
    parser.add_argument("data", type=Path, help="Preprocessed data JSONL")
    parser.add_argument(
        "--method", choices=["none", "contextual", "pool"], default="contextual"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    formulas = {}
    with args.data.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            parsed = record.get("parsed", {})
            logical = parsed.get("logical query") or parsed.get("logical_query")
            if logical:
                formulas[record["id"]] = logical.replace("{x}", "{X}")

    with args.results.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    jobs = [
        (row, formulas[row["id"]], args.method)
        for row in rows
        if "atom_scores" in row and row["id"] in formulas
    ]

    with Pool(args.workers) as pool:
        results = dict(pool.imap_unordered(_replay_record, jobs, chunksize=4))
    ground_truth = {
        rid: row["ground_truth"]
        for rid, row in results.items()
        if "ground_truth" in row
    }
    metrics = compute_metrics(results, ground_truth, args.top_k)

    out_path = args.results.with_name(
        f"{args.results.stem}_replay_{args.method}_metrics.json"
    )
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Replayed {len(results)}/{len(rows)} records -> {out_path}")


if __name__ == "__main__":
    main()
