import unittest

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.ranking.rerank import ScoredCandidate
from llm_bayesian_reasoning.ranking.score_candidate import score_candidate
from llm_bayesian_reasoning.records.atoms import Atom
from llm_bayesian_reasoning.records.query_record import QueryRecord

BOOK = Atom(name="a0", text="{X} is a book", negated_text="{X} is not a book")
NOZICK = Atom(
    name="a1",
    text="{X} is written by Robert Nozick",
    negated_text="{X} is not written by Robert Nozick",
)
CANDIDATE = Candidate(
    entity="Invariances",
    text="Invariances is a 2001 book by the philosopher Robert Nozick.",
    retrieval_score=12.5,
)


def record(formula: str) -> QueryRecord:
    return QueryRecord(
        query_id=664,
        query="Robert Nozick books",
        atoms=[BOOK, NOZICK],
        formula=formula,
        relevant_entities=["Invariances"],
        template="_",
        domain="books",
    )


class FakeEstimator:
    """Return fixed atom probabilities and remember every call."""

    def __init__(self, atom_probabilities: dict[str, float]):
        self.atom_probabilities = atom_probabilities
        self.calls = []

    def estimate_atoms(
        self, atoms: list[Atom], candidate: Candidate
    ) -> dict[str, float]:
        self.calls.append((atoms, candidate))
        return dict(self.atom_probabilities)


class TestScoreCandidate(unittest.TestCase):
    def test_when_candidate_is_scored_then_estimator_gets_all_atoms_in_one_call(self):
        estimator = FakeEstimator({"a0": 0.6, "a1": 0.5})

        score_candidate(record("((a0) , (a1))"), CANDIDATE, estimator)

        self.assertEqual(estimator.calls, [([BOOK, NOZICK], CANDIDATE)])

    def test_when_candidate_is_scored_then_result_keeps_entity_and_atom_scores(self):
        estimator = FakeEstimator({"a0": 0.6, "a1": 0.5})

        scored = score_candidate(record("((a0) , (a1))"), CANDIDATE, estimator)

        self.assertIsInstance(scored, ScoredCandidate)
        self.assertEqual(
            (scored.entity, scored.atom_probabilities),
            ("Invariances", {"a0": 0.6, "a1": 0.5}),
        )

    def test_when_formula_is_a_conjunction_then_probability_is_the_product(self):
        estimator = FakeEstimator({"a0": 0.6, "a1": 0.5})

        scored = score_candidate(record("((a0) , (a1))"), CANDIDATE, estimator)

        self.assertAlmostEqual(scored.probability, 0.3)

    def test_when_formula_is_a_disjunction_then_probability_is_either_holding(self):
        estimator = FakeEstimator({"a0": 0.6, "a1": 0.5})

        scored = score_candidate(record("((a0) ; (a1))"), CANDIDATE, estimator)

        self.assertAlmostEqual(scored.probability, 0.8)

    def test_when_formula_negates_an_atom_then_its_complement_is_used(self):
        estimator = FakeEstimator({"a0": 0.6, "a1": 0.25})

        scored = score_candidate(record("((a0) , (\\+(a1)))"), CANDIDATE, estimator)

        self.assertAlmostEqual(scored.probability, 0.45)

    def test_when_estimator_misses_an_atom_then_error_names_entity_and_atom(self):
        for formula in ("((a0) , (a1))", "(a0)"):
            with self.subTest(formula=formula):
                estimator = FakeEstimator({"a0": 0.6})

                with self.assertRaisesRegex(ValueError, "'Invariances'.*'a1'"):
                    score_candidate(record(formula), CANDIDATE, estimator)

    def test_when_estimator_adds_an_unknown_atom_then_error_names_entity_and_atom(
        self,
    ):
        estimator = FakeEstimator({"a0": 0.6, "a1": 0.5, "a7": 0.9})

        with self.assertRaisesRegex(ValueError, "'Invariances'.*'a7'"):
            score_candidate(record("((a0) , (a1))"), CANDIDATE, estimator)

    def test_when_an_atom_score_is_outside_zero_and_one_then_error_names_entity_and_atom(
        self,
    ):
        estimator = FakeEstimator({"a0": 0.6, "a1": 1.5})

        with self.assertRaisesRegex(ValueError, "'Invariances'.*'a1'"):
            score_candidate(record("((a0) , (a1))"), CANDIDATE, estimator)
