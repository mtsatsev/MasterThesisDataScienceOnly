from typing import Any

from pydantic import BaseModel

from llm_bayesian_reasoning.records.atoms import build_atoms
from llm_bayesian_reasoning.records.query_record import QueryRecord
from llm_bayesian_reasoning.records.translate_formula import translate_logical_query


class RecordRepair(BaseModel):
    """A correction to one raw query, replacing its text and optionally its atoms."""

    logical_query: str
    atoms: list[str] | None = None
    negated_atoms: list[str] | None = None


def prepare_record(
    raw: dict[str, Any],
    relevant_entities: dict[str, list[str]],
    repairs: dict[int, RecordRepair],
) -> QueryRecord:
    """
    Return the validated record for one raw dataset line.

    Args:
        raw (dict[str, Any]):
            One parsed line of the raw dataset.
        relevant_entities (dict[str, list[str]]):
            A dictionary mapping each original query to its QUEST docs.
        repairs (dict[int, RecordRepair]):
            A dictionary mapping query ids to their corrections.

    Returns:
        QueryRecord: The record, with its formula translated to ProbLog.
    """
    query_id = raw["id"]
    try:
        parsed = raw["parsed"]
        atom_texts = parsed["atoms"]
        negated_texts = parsed["negated_atoms"]
        logical_query = parsed["logical query"]

        repair = repairs.get(query_id)
        if repair is not None:
            logical_query = repair.logical_query
            if repair.atoms is not None:
                atom_texts = repair.atoms
                negated_texts = repair.negated_atoms

        atoms = build_atoms(atom_texts, negated_texts)
        formula = translate_logical_query(
            logical_query.replace("{x}", "{X}"),
            {atom.text: atom.name for atom in atoms},
        )

        original_query = raw["original_query"]
        if original_query not in relevant_entities:
            raise ValueError(f"{original_query!r} has no QUEST entry")

        return QueryRecord(
            query_id=query_id,
            query=raw["query"],
            atoms=atoms,
            formula=formula,
            relevant_entities=relevant_entities[original_query],
            template=raw["metadata"]["template"],
            domain=raw["metadata"]["domain"],
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"Query {query_id}: {error}") from error
