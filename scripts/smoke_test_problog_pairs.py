#!/usr/bin/env python3
"""Smoke test ProbLog formula generation and evaluation for dataset records."""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from problog import get_evaluatable
from problog.program import PrologString

from llm_bayesian_reasoning.problog_models.problog_models import (
    ProblogAtom,
    ProblogFormula,
)

logger = logging.getLogger("smoke_test_problog_pairs")
FORMULA_ATOM_PATTERN = re.compile(
    r"\{X\}[^()]*?(?=(\bAND\b|\bOR\b|\bNOT\b|\)|$))",
    flags=re.IGNORECASE,
)


def _normalize_placeholder(value: str) -> str:
    return value.replace("{x}", "{X}")


def _build_zero_probability_atom(atom: str) -> ProblogAtom:
    return ProblogAtom(
        atom=_normalize_placeholder(atom),
        probability=0.0,
    )


def _evaluate_program(program: str) -> float:
    db = PrologString(program)
    result = get_evaluatable().create_from(db).evaluate()
    if result:
        return float(next(iter(result.values())))
    return 0.0


def _default_output_path(results_root: Path) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return results_root / f"problog_smoke_failures_{timestamp}.jsonl"


def _count_parenthesis_imbalance(expression: str) -> dict[str, int]:
    unmatched_closing = 0
    open_count = 0
    for char in expression:
        if char == "(":
            open_count += 1
        elif char == ")":
            if open_count == 0:
                unmatched_closing += 1
            else:
                open_count -= 1
    return {
        "unmatched_opening": open_count,
        "unmatched_closing": unmatched_closing,
    }


def _extract_formula_phrases(formula_text: str) -> list[str]:
    phrases: list[str] = []
    for match in FORMULA_ATOM_PATTERN.finditer(formula_text):
        phrase = match.group(0).strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def _build_failure_reason(
    error: Exception,
    *,
    phase: str,
    atoms: list[ProblogAtom] | None,
    formula: ProblogFormula | None,
    program: str | None,
) -> tuple[str, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}
    message = str(error)

    if formula is not None:
        diagnostics["formula_parentheses"] = _count_parenthesis_imbalance(
            formula.formula
        )
        diagnostics["formula_atom_phrases"] = _extract_formula_phrases(formula.formula)

    if atoms is not None and formula is not None:
        atom_texts = {atom.atom for atom in atoms}
        missing_formula_atoms = [
            phrase
            for phrase in diagnostics.get("formula_atom_phrases", [])
            if phrase not in atom_texts
        ]
        if missing_formula_atoms:
            diagnostics["missing_formula_atoms"] = missing_formula_atoms

    if phase == "parse_json":
        return "invalid_json", diagnostics

    if phase == "load_record":
        return "invalid_record_structure", diagnostics

    if "not present in the supplied atom list" in message:
        return "formula_atom_mismatch", diagnostics

    if "unresolved placeholder text" in message:
        return "unresolved_formula_text", diagnostics

    if program is not None and "Expected binary operator" in message:
        diagnostics["program_contains_raw_text"] = " is " in program
        return "invalid_raw_formula_text", diagnostics

    if program is not None and "Unmatched character" in message:
        diagnostics["program_parentheses"] = _count_parenthesis_imbalance(program)
        return "unbalanced_formula_parentheses", diagnostics

    if diagnostics.get("missing_formula_atoms"):
        return "formula_atom_mismatch", diagnostics

    if formula is not None:
        formula_parentheses = diagnostics.get("formula_parentheses", {})
        if formula_parentheses.get("unmatched_opening") or formula_parentheses.get(
            "unmatched_closing"
        ):
            return "unbalanced_formula_parentheses", diagnostics

    return "problog_evaluation_error", diagnostics


def _load_record(
    doc: dict[str, Any],
) -> tuple[str | int, str, list[ProblogAtom], ProblogFormula]:
    record_id = doc.get("id")
    if record_id is None:
        raise ValueError("Record is missing 'id'")

    query = doc.get("query") or doc.get("original_query") or ""
    parsed = doc.get("parsed") or {}

    atoms_raw = parsed.get("atoms") or []
    if not isinstance(atoms_raw, list):
        raise TypeError("parsed.atoms must be a list")

    logical_query = parsed.get("logical query") or parsed.get("logical_query")
    if not isinstance(logical_query, str) or not logical_query.strip():
        raise ValueError("parsed.logical query is missing or empty")

    atoms = [
        _build_zero_probability_atom(atom)
        for atom in atoms_raw
        if isinstance(atom, str) and atom.strip()
    ]
    if not atoms:
        raise ValueError("Record has no valid atoms")

    formula = ProblogFormula(formula=_normalize_placeholder(logical_query))
    return record_id, query, atoms, formula


def _write_failure(
    handle,
    *,
    record_id: str | int | None,
    query: str,
    phase: str,
    error: Exception,
    atoms: list[ProblogAtom] | None = None,
    formula: ProblogFormula | None = None,
    program: str | None = None,
) -> None:
    reason, diagnostics = _build_failure_reason(
        error,
        phase=phase,
        atoms=atoms,
        formula=formula,
        program=program,
    )
    row = {
        "id": record_id,
        "query": query,
        "status": "failed",
        "phase": phase,
        "reason": reason,
        "error_type": type(error).__name__,
        "error": str(error),
        "num_atoms": len(atoms) if atoms is not None else None,
        "atoms": [atom.model_dump() for atom in atoms] if atoms is not None else None,
        "formula": formula.formula if formula is not None else None,
        "program": program,
        "diagnostics": diagnostics,
    }
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Load a preprocessed dataset, set all atom probabilities to 0, and "
            "smoke test ProbLog program generation/evaluation for a fake entity."
        )
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("llm_bayesian_reasoning/data/preprocessed_data/parsed_test.jsonl"),
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="__smoke_test_entity__",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional record limit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL file to write failures to",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results_root = Path("llm_bayesian_reasoning/results")
    results_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output or _default_output_path(results_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    succeeded = 0
    failed = 0

    with (
        args.data_file.open(encoding="utf-8") as in_f,
        output_path.open("w", encoding="utf-8") as out_f,
    ):
        for line_number, line in enumerate(in_f, start=1):
            if args.limit is not None and processed >= args.limit:
                break

            try:
                doc = json.loads(line)
            except json.JSONDecodeError as error:
                failed += 1
                processed += 1
                _write_failure(
                    out_f,
                    record_id=None,
                    query="",
                    phase="parse_json",
                    error=error,
                )
                logger.warning("Line %s failed during JSON parsing", line_number)
                continue

            record_id = doc.get("id")
            query = doc.get("query") or doc.get("original_query") or ""

            try:
                record_id, query, atoms, formula = _load_record(doc)
            except Exception as error:  # noqa: BLE001
                failed += 1
                processed += 1
                _write_failure(
                    out_f,
                    record_id=record_id,
                    query=query,
                    phase="load_record",
                    error=error,
                )
                logger.warning("Record %r failed during loading", record_id)
                continue

            try:
                program = formula.to_problog(atoms, args.entity)
            except Exception as error:  # noqa: BLE001
                failed += 1
                processed += 1
                _write_failure(
                    out_f,
                    record_id=record_id,
                    query=query,
                    phase="build_program",
                    error=error,
                    atoms=atoms,
                    formula=formula,
                )
                logger.warning("Record %r failed during program build", record_id)
                continue

            try:
                _evaluate_program(program)
            except Exception as error:  # noqa: BLE001
                failed += 1
                processed += 1
                _write_failure(
                    out_f,
                    record_id=record_id,
                    query=query,
                    phase="evaluate_program",
                    error=error,
                    atoms=atoms,
                    formula=formula,
                    program=program,
                )
                logger.warning("Record %r failed during ProbLog evaluation", record_id)
                continue

            processed += 1
            succeeded += 1

    logger.info("Processed records: %d", processed)
    logger.info("Succeeded: %d", succeeded)
    logger.info("Failed: %d", failed)
    logger.info("Failure report: %s", output_path)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
