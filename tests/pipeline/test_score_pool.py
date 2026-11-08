import math
import unittest
from typing import Any

from pydantic import ValidationError

from llm_bayesian_reasoning.pipeline.score_pool import QueryResult
from llm_bayesian_reasoning.ranking.rerank import ScoredCandidate

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
