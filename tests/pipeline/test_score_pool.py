import math
import unittest
from typing import Any

from pydantic import ValidationError

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.pipeline.score_pool import QueryResult, score_pool
from llm_bayesian_reasoning.ranking.rerank import ScoredCandidate
from llm_bayesian_reasoning.records.atoms import Atom
from llm_bayesian_reasoning.records.query_record import QueryRecord

INVARIANCES = ScoredCandidate(
    entity="Invariances", probability=0.3, atom_probabilities={"a0": 0.6, "a1": 0.5}
)
ZETTEL = ScoredCandidate(
    entity="Zettel", probability=0.02, atom_probabilities={"a0": 0.9, "a1": 0.02}
)


def result_fields(**changes: Any) -> dict[str, Any]:
    return {
        "query_id": 664,
        "retrieved": ["Zettel", "Invariances"],
        "reranked": [INVARIANCES, ZETTEL],
        "content_free_scores": {"a0": 0.4, "a1": 0.3},
        **changes,
    }


class TestQueryResult(unittest.TestCase):
    def test_when_all_fields_are_valid_then_result_keeps_their_values(self):
        fields = result_fields()

        result = QueryResult(**fields)

        self.assertEqual(
            {name: getattr(result, name) for name in fields},
            fields,
        )

    def test_when_result_is_written_to_json_then_reading_it_back_gives_an_equal_one(
        self,
    ):
        result = QueryResult(**result_fields())

        self.assertEqual(
            QueryResult.model_validate_json(result.model_dump_json()), result
        )

    def test_when_nothing_was_retrieved_then_empty_pool_is_accepted(self):
        result = QueryResult(**result_fields(retrieved=[], reranked=[]))

        self.assertEqual((result.retrieved, result.reranked), ([], []))

    def test_when_retrieved_repeats_an_entity_then_error_names_the_entity(self):
        with self.assertRaisesRegex(ValidationError, "'Zettel'"):
            QueryResult(**result_fields(retrieved=["Zettel", "Invariances", "Zettel"]))

    def test_when_reranked_repeats_an_entity_then_error_names_the_entity(self):
        with self.assertRaisesRegex(ValidationError, "'Zettel'"):
            QueryResult(
                **result_fields(
                    retrieved=["Zettel", "Invariances"],
                    reranked=[INVARIANCES, ZETTEL, ZETTEL],
                )
            )

    def test_when_a_reranked_entity_was_not_retrieved_then_error_names_the_entity(
        self,
    ):
        with self.assertRaisesRegex(ValidationError, "'Zettel'"):
            QueryResult(**result_fields(retrieved=["Invariances"]))

    def test_when_a_retrieved_entity_was_not_reranked_then_error_names_the_entity(
        self,
    ):
        with self.assertRaisesRegex(ValidationError, "'Zettel'"):
            QueryResult(**result_fields(reranked=[INVARIANCES]))

    def test_when_a_content_free_score_is_outside_zero_and_one_then_error_names_the_atom(
        self,
    ):
        for value in (-0.1, 1.1, math.nan):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValidationError, "content_free_scores.a1"):
                    QueryResult(
                        **result_fields(content_free_scores={"a0": 0.4, "a1": value})
                    )

    def test_when_no_content_free_score_is_given_then_error_names_the_field(self):
        with self.assertRaisesRegex(ValidationError, "content_free_scores"):
            QueryResult(**result_fields(content_free_scores={}))

    def test_when_a_candidate_has_other_atoms_than_content_free_scores_then_error_names_it(
        self,
    ):
        for content_free_scores in ({"a0": 0.4}, {"a0": 0.4, "a1": 0.3, "a2": 0.5}):
            with self.subTest(content_free_scores=content_free_scores):
                with self.assertRaisesRegex(ValidationError, "'Invariances'"):
                    QueryResult(
                        **result_fields(content_free_scores=content_free_scores)
                    )

    def test_when_a_field_is_missing_then_error_names_the_field(self):
        for field in result_fields():
            with self.subTest(field=field):
                fields = {k: v for k, v in result_fields().items() if k != field}

                with self.assertRaisesRegex(ValidationError, field):
                    QueryResult(**fields)

    def test_when_an_unknown_field_is_given_then_error_names_the_field(self):
        with self.assertRaisesRegex(ValidationError, "estimator"):
            QueryResult(**result_fields(estimator="yes_no"))


BOOK = Atom(name="a0", text="{X} is a book", negated_text="{X} is not a book")
NOZICK = Atom(
    name="a1",
    text="{X} is written by Robert Nozick",
    negated_text="{X} is not written by Robert Nozick",
)
RECORD = QueryRecord(
    query_id=664,
    query="Robert Nozick books",
    atoms=[BOOK, NOZICK],
    formula="((a0) , (a1))",
    relevant_entities=["Invariances"],
    template="_",
    domain="books",
)


def candidate(entity: str, retrieval_score: float = 10.0) -> Candidate:
    return Candidate(
        entity=entity, text=f"{entity} is a text.", retrieval_score=retrieval_score
    )


class FakeEstimator:
    """Return fixed atom probabilities per entity and remember every call."""

    def __init__(self, scores: dict[str, dict[str, float]]):
        self.scores = scores
        self.calls = []

    def estimate_atoms(
        self, atoms: list[Atom], candidate: Candidate
    ) -> dict[str, float]:
        self.calls.append((atoms, candidate))
        return dict(self.scores[candidate.entity])


SCORES = {
    "Zettel": {"a0": 0.9, "a1": 0.02},
    "Invariances": {"a0": 0.6, "a1": 0.5},
    "Socratic Puzzles": {"a0": 0.8, "a1": 0.25},
    "N/A": {"a0": 0.4, "a1": 0.3},
}
POOL = [candidate("Zettel", 14.0), candidate("Invariances", 12.5)]


class TestScorePool(unittest.TestCase):
    def test_when_pool_is_scored_then_each_candidate_gets_one_call_with_all_atoms(
        self,
    ):
        estimator = FakeEstimator(SCORES)

        score_pool(RECORD, POOL, estimator)

        self.assertEqual(
            [call for call in estimator.calls if call[1].entity != "N/A"],
            [([BOOK, NOZICK], POOL[0]), ([BOOK, NOZICK], POOL[1])],
        )

    def test_when_pool_is_scored_then_atoms_are_scored_once_on_content_free_input(
        self,
    ):
        estimator = FakeEstimator(SCORES)

        score_pool(RECORD, POOL, estimator)

        content_free_calls = [
            (atoms, scored.text)
            for atoms, scored in estimator.calls
            if scored.entity == "N/A"
        ]
        self.assertEqual(content_free_calls, [([BOOK, NOZICK], "N/A")])

    def test_when_pool_is_scored_then_result_names_the_query(self):
        result = score_pool(RECORD, POOL, FakeEstimator(SCORES))

        self.assertEqual(result.query_id, 664)

    def test_when_pool_is_scored_then_retrieved_keeps_retrieval_order(self):
        result = score_pool(RECORD, POOL, FakeEstimator(SCORES))

        self.assertEqual(result.retrieved, ["Zettel", "Invariances"])

    def test_when_pool_is_scored_then_reranked_puts_highest_probability_first(self):
        result = score_pool(RECORD, POOL, FakeEstimator(SCORES))

        self.assertEqual(len(result.reranked), 2)
        self.assertEqual(
            [(scored.entity, scored.atom_probabilities) for scored in result.reranked],
            [
                ("Invariances", {"a0": 0.6, "a1": 0.5}),
                ("Zettel", {"a0": 0.9, "a1": 0.02}),
            ],
        )
        self.assertAlmostEqual(result.reranked[0].probability, 0.3)
        self.assertAlmostEqual(result.reranked[1].probability, 0.018)

    def test_when_probabilities_tie_then_reranked_keeps_retrieval_order(self):
        scores = {
            "Zettel": {"a0": 0.5, "a1": 0.5},
            "Invariances": {"a0": 0.5, "a1": 0.5},
            "N/A": {"a0": 0.4, "a1": 0.3},
        }

        result = score_pool(RECORD, POOL, FakeEstimator(scores))

        self.assertEqual(
            [scored.entity for scored in result.reranked], ["Zettel", "Invariances"]
        )

    def test_when_pool_is_scored_then_content_free_scores_are_kept(self):
        result = score_pool(RECORD, POOL, FakeEstimator(SCORES))

        self.assertEqual(result.content_free_scores, {"a0": 0.4, "a1": 0.3})

    def test_when_pool_is_empty_then_only_content_free_scores_are_computed(self):
        estimator = FakeEstimator(SCORES)

        result = score_pool(RECORD, [], estimator)

        self.assertEqual(
            (result.retrieved, result.reranked, result.content_free_scores),
            ([], [], {"a0": 0.4, "a1": 0.3}),
        )
        self.assertEqual(len(estimator.calls), 1)

    def test_when_a_candidate_is_scored_badly_then_error_names_the_entity(self):
        scores = {**SCORES, "Invariances": {"a0": 0.6, "a1": 1.5}}

        with self.assertRaisesRegex(ValueError, "'Invariances'.*'a1'"):
            score_pool(RECORD, POOL, FakeEstimator(scores))

    def test_when_content_free_input_is_scored_badly_then_error_names_it(self):
        for content_free_scores in ({"a0": 0.4}, {"a0": 0.4, "a1": 1.5}):
            with self.subTest(content_free_scores=content_free_scores):
                scores = {**SCORES, "N/A": content_free_scores}

                with self.assertRaisesRegex(ValueError, "'N/A'.*'a1'"):
                    score_pool(RECORD, [], FakeEstimator(scores))

    def test_when_an_entity_is_retrieved_twice_then_error_names_the_entity(self):
        pool = [candidate("Zettel"), candidate("Invariances"), candidate("Zettel")]

        with self.assertRaisesRegex(ValueError, "'Zettel'"):
            score_pool(RECORD, pool, FakeEstimator(SCORES))
