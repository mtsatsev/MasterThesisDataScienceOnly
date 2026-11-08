from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Probability = Annotated[float, Field(ge=0, le=1)]


class ScoredCandidate(BaseModel):
    """A retrieved entity with its formula probability and the atom scores behind it."""

    model_config = ConfigDict(extra="forbid")

    entity: str = Field(min_length=1)
    probability: Probability
    atom_probabilities: dict[str, Probability] = Field(min_length=1)
