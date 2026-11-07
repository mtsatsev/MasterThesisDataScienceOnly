from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llm_bayesian_reasoning.logic.formula import evaluate_formula
from llm_bayesian_reasoning.records.atoms import Atom


def _reject_repeats(kind: str, values: list[str]) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"Query repeats {kind} {value!r}")
        seen.add(value)


class QueryRecord(BaseModel):
    """A dataset query with its atoms, ProbLog formula and relevant entities."""

    model_config = ConfigDict(extra="forbid")

    query_id: int
    query: str = Field(min_length=1)
    atoms: list[Atom] = Field(min_length=1)
    formula: str = Field(min_length=1)
    relevant_entities: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    template: str = Field(min_length=1)
    domain: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Reject repeated values and formulas that the atoms cannot evaluate."""
        _reject_repeats("relevant entity", self.relevant_entities)
        _reject_repeats("atom name", [atom.name for atom in self.atoms])
        _reject_repeats("atom text", [atom.text for atom in self.atoms])
        evaluate_formula(self.formula, {atom.name: 0.5 for atom in self.atoms})
        return self


def load_query_records(path: Path) -> list[QueryRecord]:
    """
    Return the query records of a JSONL file, in file order.

    Args:
        path (Path):
            The JSONL file, with one query record per line.

    Returns:
        list[QueryRecord]: The validated records.
    """
    records: list[QueryRecord] = []
    seen_ids: set[int] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = QueryRecord.model_validate_json(line)
            except ValidationError as error:
                raise ValueError(
                    f"{path}: invalid record on line {line_number}: {error}"
                ) from error
            if record.query_id in seen_ids:
                raise ValueError(
                    f"{path}: line {line_number} repeats query id {record.query_id}"
                )
            seen_ids.add(record.query_id)
            records.append(record)
    if not records:
        raise ValueError(f"{path}: file has no records")
    return records
