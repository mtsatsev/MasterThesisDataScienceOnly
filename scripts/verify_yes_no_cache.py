#!/usr/bin/env python3
"""Check that the cached single-pass yes/no estimator matches the old two-pass one.

Loads the old ``LikelihoodBasedYesNoEstimator`` from git (the commit before the
single-pass change), scores the same atoms/entities with both versions on one
shared model, and prints the largest probability difference and both timings.
Differences around 1e-2 are expected under 4-bit/fp16 numerics; anything much
larger, or a changed atom ordering, points to a bug.

Run on the server (loads the full model):
    python scripts/verify_yes_no_cache.py --model meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import argparse
import importlib.util
import json
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_bayesian_reasoning.estimators.likelihood_based_yes_no_estimator import (
    LikelihoodBasedYesNoEstimator as NewEstimator,
)
from llm_bayesian_reasoning.problog_models.problog_models import ProblogAtom

ESTIMATOR_PATH = (
    "llm_bayesian_reasoning/estimators/likelihood_based_yes_no_estimator.py"
)


def _load_old_estimator(ref: str):
    source = subprocess.run(
        ["git", "show", f"{ref}:{ESTIMATOR_PATH}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    path = Path(tempfile.mkdtemp()) / "old_yes_no.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("old_yes_no", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LikelihoodBasedYesNoEstimator


def _first_lines(path: Path, n: int) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for _, line in zip(range(n), f)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--old-ref", default="b48c2d7~1")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "llm_bayesian_reasoning/data/preprocessed_data/parsed_test_with_negs.jsonl"
        ),
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("llm_bayesian_reasoning/data/index_data/documents.jsonl"),
    )
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--docs-per-query", type=int, default=3)
    parser.add_argument(
        "--max-chars", type=int, default=40000, help="Truncate pages to this length"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
    ).eval()
    old = _load_old_estimator(args.old_ref)(model, tokenizer)
    new = NewEstimator(model, tokenizer)

    records = _first_lines(args.data, args.queries)
    docs = _first_lines(args.documents, args.docs_per_query)
    max_diff, t_old, t_new, n_cases, order_changes = 0.0, 0.0, 0.0, 0, 0
    for record in records:
        atom_texts = [a.replace("{x}", "{X}") for a in record["parsed"]["atoms"]]
        for doc in docs:
            for context in (doc["text"].strip()[: args.max_chars], None):
                atoms = [ProblogAtom(atom=a, context=context) for a in atom_texts]
                start = time.perf_counter()
                p_old = [
                    a.probability for a in old.score_probability(atoms, doc["title"])
                ]
                t_old += time.perf_counter() - start
                start = time.perf_counter()
                p_new = [
                    a.probability for a in new.score_probability(atoms, doc["title"])
                ]
                t_new += time.perf_counter() - start
                max_diff = max(max_diff, *(abs(a - b) for a, b in zip(p_old, p_new)))
                order_changes += sorted(
                    range(len(p_old)), key=p_old.__getitem__
                ) != sorted(range(len(p_new)), key=p_new.__getitem__)
                n_cases += 1

    print(f"cases (query x page x ctx/noctx): {n_cases}")
    print(f"max |p_old - p_new|:              {max_diff:.2e}")
    print(f"cases with changed atom ordering: {order_changes}")
    print(f"time old / new: {t_old:.1f}s / {t_new:.1f}s  ({t_old / t_new:.1f}x faster)")


if __name__ == "__main__":
    main()
