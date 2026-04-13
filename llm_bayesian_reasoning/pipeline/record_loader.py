import json
from pathlib import Path
from typing import TypedDict

from llm_bayesian_reasoning.estimators.scoring_inputs import (
    ContrastiveScoringInput,
    PositiveScoringInput,
    ScoringInput,
)
from llm_bayesian_reasoning.pipeline.config import EstimatorType
from llm_bayesian_reasoning.problog_models.problog_models import (
    ProblogAtom,
    ProblogFormula,
)


class LoadedPreprocessedRecord(TypedDict):
    query: str
    atoms: list[ProblogAtom]
    negated_atoms: list[ProblogAtom]
    problog_formula: ProblogFormula


class MaterializedPreprocessedRecord(TypedDict):
    query: str
    atoms: list[ScoringInput]
    problog_formula: ProblogFormula


def normalize_placeholder(value: str) -> str:
    return value.replace("{x}", "{X}")


def build_atom(atom: str) -> ProblogAtom:
    return ProblogAtom(atom=normalize_placeholder(atom))


def load_preprocessed_records(
    path: Path,
    limit: int | None = None,
    use_metadata_ground_truth: bool = False,
) -> tuple[dict[int | str, LoadedPreprocessedRecord], dict[int | str, list[str]] | None]:
    """Load preprocessed JSONL records into a shared in-memory representation."""
    data: dict[int | str, LoadedPreprocessedRecord] = {}
    ground_truth: dict[int | str, list[str]] = {}

    with path.open(encoding="utf-8") as file_handle:
        for index, line in enumerate(file_handle):
            if limit is not None and index >= limit:
                break
            try:
                document = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_id = document.get("id")
            if record_id is None:
                continue

            query = document.get("query") or document.get("original_query") or ""
            parsed = document.get("parsed", {})
            atoms_raw = parsed.get("atoms", [])
            negated_atoms_raw = parsed.get("negated_atoms", [])
            logical = parsed.get("logical query") or parsed.get("logical_query") or ""

            atoms_text = [atom for atom in atoms_raw if isinstance(atom, str)]
            negated_atoms_text = [
                atom for atom in negated_atoms_raw if isinstance(atom, str)
            ]

            data[record_id] = {
                "query": query,
                "atoms": [build_atom(atom) for atom in atoms_text],
                "negated_atoms": [build_atom(atom) for atom in negated_atoms_text],
                "problog_formula": ProblogFormula(
                    formula=normalize_placeholder(logical)
                ),
            }

            if use_metadata_ground_truth:
                metadata = document.get("metadata", {})
                relevance = metadata.get("relevance_ratings") or {}
                if isinstance(relevance, dict):
                    ground_truth[record_id] = list(relevance.keys())

    return data, ground_truth if use_metadata_ground_truth else None


def build_atoms_for_estimator(
    record: LoadedPreprocessedRecord,
    estimator_type: EstimatorType,
) -> list[ScoringInput]:
    positive_atoms = record["atoms"]
    negated_atoms = record["negated_atoms"]

    if estimator_type == EstimatorType.LIKELIHOOD_BASED_CONTRASTIVE:
        if not negated_atoms:
            raise ValueError(
                "Contrastive estimators require parsed.negated_atoms in the dataset"
            )
        if len(positive_atoms) != len(negated_atoms):
            raise ValueError(
                "Contrastive estimators require the same number of atoms and negated_atoms"
            )
        return [
            ContrastiveScoringInput(atom=atom, negated_atom=negated_atom)
            for atom, negated_atom in zip(positive_atoms, negated_atoms)
        ]

    return [PositiveScoringInput(atom=atom) for atom in positive_atoms]


def materialize_records_for_estimator(
    records: dict[int | str, LoadedPreprocessedRecord],
    estimator_type: EstimatorType,
) -> dict[int | str, MaterializedPreprocessedRecord]:
    materialized: dict[int | str, MaterializedPreprocessedRecord] = {}

    for record_id, record in records.items():
        materialized[record_id] = {
            "query": record["query"],
            "atoms": build_atoms_for_estimator(record, estimator_type),
            "problog_formula": record["problog_formula"],
        }

    return materialized