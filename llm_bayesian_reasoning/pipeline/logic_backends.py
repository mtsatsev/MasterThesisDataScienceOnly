import logging
from abc import ABC, abstractmethod

from problog import get_evaluatable
from problog.program import PrologString

from llm_bayesian_reasoning.problog_models.problog_models import (
    ProblogAtom,
    ProblogFormula,
)

logger = logging.getLogger(__name__)

_deepproblog_compat_warned = False


def evaluate_problog_program(program: str) -> float:
    try:
        db = PrologString(program)
        result = get_evaluatable().create_from(db).evaluate()
        if result:
            return float(next(iter(result.values())))
        return 0.0
    except Exception:  # noqa: BLE001
        logger.exception("ProbLog evaluation failed for program:\n%s", program)
        return 0.0


class LogicBackend(ABC):
    @abstractmethod
    def build_program(
        self, atoms: list[ProblogAtom], formula: ProblogFormula, entity: str
    ) -> str:
        """Build a logic program for a single entity."""

    @abstractmethod
    def evaluate(
        self, atoms: list[ProblogAtom], formula: ProblogFormula, entity: str
    ) -> float:
        """Evaluate a single entity under the backend."""


class ProbLogBackend(LogicBackend):
    def build_program(
        self, atoms: list[ProblogAtom], formula: ProblogFormula, entity: str
    ) -> str:
        return formula.to_problog(atoms, entity)

    def evaluate(
        self, atoms: list[ProblogAtom], formula: ProblogFormula, entity: str
    ) -> float:
        return evaluate_problog_program(self.build_program(atoms, formula, entity))


class DeepProbLogBackend(LogicBackend):
    """Compatibility backend for DeepProbLog integration.

    This first implementation keeps the current fixed-probability facts and
    delegates evaluation to the ProbLog engine. It provides a dedicated backend
    boundary so neural predicates can replace these facts in a follow-up step.
    """

    def build_program(
        self, atoms: list[ProblogAtom], formula: ProblogFormula, entity: str
    ) -> str:
        return formula.to_deepproblog(atoms, entity)

    def evaluate(
        self, atoms: list[ProblogAtom], formula: ProblogFormula, entity: str
    ) -> float:
        global _deepproblog_compat_warned
        if not _deepproblog_compat_warned:
            logger.warning(
                "DeepProbLog backend is running in compatibility mode: "
                "fixed probabilistic facts are still evaluated with the ProbLog "
                "engine. Neural predicates are the next implementation step."
            )
            _deepproblog_compat_warned = True
        return evaluate_problog_program(self.build_program(atoms, formula, entity))
