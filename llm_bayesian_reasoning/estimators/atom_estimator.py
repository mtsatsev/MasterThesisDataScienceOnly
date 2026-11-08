from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from llm_bayesian_reasoning.records.atoms import Atom


class Candidate(BaseModel):
    """A retrieved entity with its text and retrieval score, before any scoring."""

    model_config = ConfigDict(extra="forbid")

    entity: str = Field(min_length=1)
    text: str
    retrieval_score: float = Field(allow_inf_nan=False)


class AtomEstimator(Protocol):
    """Anything that scores all atoms of a query for one candidate in one call."""

    def estimate_atoms(
        self, atoms: list[Atom], candidate: Candidate
    ) -> dict[str, float]:
        """Return the probability of each atom, keyed by atom name."""
        ...
