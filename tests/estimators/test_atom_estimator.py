import math
import unittest

from pydantic import ValidationError

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate


class TestCandidate(unittest.TestCase):
    VALID_FIELDS = {
        "entity": "Invariances",
        "text": "Invariances is a 2001 book by the philosopher Robert Nozick.",
        "retrieval_score": 12.5,
    }

    def test_when_all_fields_are_valid_then_candidate_keeps_their_values(self):
        candidate = Candidate(**self.VALID_FIELDS)

        self.assertEqual(candidate.model_dump(), self.VALID_FIELDS)

    def test_when_candidate_is_written_to_json_then_reading_it_back_gives_an_equal_one(
        self,
    ):
        candidate = Candidate(**self.VALID_FIELDS)

        self.assertEqual(
            Candidate.model_validate_json(candidate.model_dump_json()), candidate
        )

    def test_when_retrieval_score_is_negative_then_it_is_accepted(self):
        candidate = Candidate(**{**self.VALID_FIELDS, "retrieval_score": -0.25})

        self.assertEqual(candidate.retrieval_score, -0.25)

    def test_when_retrieval_score_is_not_finite_then_error_names_the_field(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValidationError, "retrieval_score"):
                    Candidate(**{**self.VALID_FIELDS, "retrieval_score": value})

    def test_when_entity_is_empty_then_error_names_the_field(self):
        with self.assertRaisesRegex(ValidationError, "entity"):
            Candidate(**{**self.VALID_FIELDS, "entity": ""})

    def test_when_a_field_is_missing_then_error_names_the_field(self):
        for field in self.VALID_FIELDS:
            with self.subTest(field=field):
                fields = {k: v for k, v in self.VALID_FIELDS.items() if k != field}

                with self.assertRaisesRegex(ValidationError, field):
                    Candidate(**fields)

    def test_when_an_unknown_field_is_given_then_error_names_the_field(self):
        with self.assertRaisesRegex(ValidationError, "probability"):
            Candidate(**self.VALID_FIELDS, probability=0.5)
