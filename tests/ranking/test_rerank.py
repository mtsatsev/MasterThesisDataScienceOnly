import math
import unittest

from pydantic import ValidationError

from llm_bayesian_reasoning.ranking.rerank import ScoredCandidate, rerank


def candidate(entity: str, probability: float) -> ScoredCandidate:
    return ScoredCandidate(
        entity=entity, probability=probability, atom_probabilities={"a0": probability}
    )


def entities(candidates: list[ScoredCandidate]) -> list[str]:
    return [scored.entity for scored in candidates]


class TestScoredCandidate(unittest.TestCase):
    VALID_FIELDS = {
        "entity": "Invariances",
        "probability": 0.3,
        "atom_probabilities": {"a0": 0.6, "a1": 0.5},
    }

    def test_when_all_fields_are_valid_then_candidate_keeps_their_values(self):
        candidate = ScoredCandidate(**self.VALID_FIELDS)

        self.assertEqual(candidate.model_dump(), self.VALID_FIELDS)

    def test_when_candidate_is_written_to_json_then_reading_it_back_gives_an_equal_one(
        self,
    ):
        candidate = ScoredCandidate(**self.VALID_FIELDS)

        self.assertEqual(
            ScoredCandidate.model_validate_json(candidate.model_dump_json()), candidate
        )

    def test_when_probabilities_are_exactly_zero_or_one_then_they_are_accepted(self):
        for value in (0.0, 1.0):
            with self.subTest(value=value):
                candidate = ScoredCandidate(
                    entity="Invariances",
                    probability=value,
                    atom_probabilities={"a0": value},
                )

                self.assertEqual(
                    (candidate.probability, candidate.atom_probabilities["a0"]),
                    (value, value),
                )

    def test_when_probability_is_outside_zero_and_one_then_error_names_the_field(self):
        for value in (-0.1, 1.1, math.nan):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValidationError, "probability"):
                    ScoredCandidate(**{**self.VALID_FIELDS, "probability": value})

    def test_when_an_atom_probability_is_outside_zero_and_one_then_error_names_the_atom(
        self,
    ):
        for value in (-0.1, 1.1, math.nan):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValidationError, "atom_probabilities.a1"):
                    ScoredCandidate(
                        **{
                            **self.VALID_FIELDS,
                            "atom_probabilities": {"a0": 0.6, "a1": value},
                        }
                    )

    def test_when_entity_is_empty_then_error_names_the_field(self):
        with self.assertRaisesRegex(ValidationError, "entity"):
            ScoredCandidate(**{**self.VALID_FIELDS, "entity": ""})

    def test_when_no_atom_is_scored_then_error_names_the_field(self):
        with self.assertRaisesRegex(ValidationError, "atom_probabilities"):
            ScoredCandidate(**{**self.VALID_FIELDS, "atom_probabilities": {}})

    def test_when_a_field_is_missing_then_error_names_the_field(self):
        for field in self.VALID_FIELDS:
            with self.subTest(field=field):
                fields = {k: v for k, v in self.VALID_FIELDS.items() if k != field}

                with self.assertRaisesRegex(ValidationError, field):
                    ScoredCandidate(**fields)

    def test_when_an_unknown_field_is_given_then_error_names_the_field(self):
        with self.assertRaisesRegex(ValidationError, "retrieval_score"):
            ScoredCandidate(**self.VALID_FIELDS, retrieval_score=12.5)


class TestRerank(unittest.TestCase):
    def test_when_no_candidates_are_given_then_result_is_empty(self):
        self.assertEqual(rerank([]), [])

    def test_when_one_candidate_is_given_then_result_is_that_candidate(self):
        scored = candidate("Invariances", 0.3)

        self.assertEqual(rerank([scored]), [scored])

    def test_when_probabilities_differ_then_highest_probability_comes_first(self):
        candidates = [
            candidate("Socratic Puzzles", 0.2),
            candidate("Invariances", 0.9),
            candidate("Anarchy, State, and Utopia", 0.5),
        ]

        self.assertEqual(
            entities(rerank(candidates)),
            ["Invariances", "Anarchy, State, and Utopia", "Socratic Puzzles"],
        )

    def test_when_more_candidates_than_any_cutoff_are_given_then_all_are_kept(self):
        candidates = [candidate(f"Entity {index}", index / 100) for index in range(60)]

        self.assertEqual(
            entities(rerank(candidates)),
            [f"Entity {index}" for index in reversed(range(60))],
        )

    def test_when_probabilities_tie_then_input_order_is_kept(self):
        candidates = [
            candidate("Zettel", 0.5),
            candidate("Invariances", 0.9),
            candidate("Anarchy", 0.5),
            candidate("Middle", 0.5),
        ]

        self.assertEqual(
            entities(rerank(candidates)), ["Invariances", "Zettel", "Anarchy", "Middle"]
        )

    def test_when_every_probability_is_zero_then_input_order_is_kept(self):
        candidates = [
            candidate(entity, 0.0) for entity in ("Zettel", "Anarchy", "Middle")
        ]

        self.assertEqual(entities(rerank(candidates)), ["Zettel", "Anarchy", "Middle"])

    def test_when_candidates_are_reranked_then_each_keeps_its_own_scores(self):
        candidates = [candidate("Socratic Puzzles", 0.2), candidate("Invariances", 0.9)]

        self.assertEqual(rerank(candidates), [candidates[1], candidates[0]])

    def test_when_candidates_are_reranked_then_input_list_is_unchanged(self):
        candidates = [candidate("Socratic Puzzles", 0.2), candidate("Invariances", 0.9)]
        original = list(candidates)

        rerank(candidates)

        self.assertEqual(candidates, original)

    def test_when_an_entity_repeats_then_error_names_the_entity(self):
        candidates = [
            candidate("Invariances", 0.9),
            candidate("Socratic Puzzles", 0.2),
            candidate("Invariances", 0.4),
        ]

        with self.assertRaisesRegex(ValueError, "'Invariances'"):
            rerank(candidates)
