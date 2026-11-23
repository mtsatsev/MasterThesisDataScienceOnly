import math
import unittest

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.negation_consistent_yes_no import (
    NegationConsistentYesNoEstimator,
)
from llm_bayesian_reasoning.records.atoms import Atom

FILM = Atom(name="a0", text="{X} is a film", negated_text="{X} is not a film")
AMERICAN = Atom(
    name="a1",
    text="{X} is an American film.",
    negated_text="{X} is not an American film.",
)
CANDIDATE = Candidate(
    entity="Blade Runner",
    text="Blade Runner is a 1982 science fiction film.",
    retrieval_score=12.5,
)


def prompt(statement: str, document: str = CANDIDATE.text, entity=None) -> str:
    """Return the yes/no estimator's prompt for a statement."""
    entity = entity or "'Blade Runner'"
    return (
        f"Context:\n{document}\n\n"
        f"Entity: {entity}\nStatement: {statement}\n\n"
        "Is the statement supported by the context?\nAnswer:"
    )


FILM_PROMPT = prompt("'Blade Runner' is a film")
NOT_FILM_PROMPT = prompt("'Blade Runner' is not a film")
AMERICAN_PROMPT = prompt("'Blade Runner' is an American film.")
NOT_AMERICAN_PROMPT = prompt("'Blade Runner' is not an American film.")


class FakeLanguageModel:
    """Return a stored yes/no gap per prompt (0 when unknown), and remember calls."""

    def __init__(self, gaps: dict[str, float] | None = None):
        self.gaps = gaps or {}
        self.calls = []

    def next_token_logprobs(
        self, prompts: list[str], tokens: list[str]
    ) -> list[dict[str, float]]:
        self.calls.append((prompts, tokens))
        return [
            {" yes": self.gaps.get(text, 0.0) - 5.0, " no": -5.0} for text in prompts
        ]


def sigmoid(logit: float) -> float:
    return 1 / (1 + math.exp(-logit))


class TestNegationConsistentYesNoEstimator(unittest.TestCase):
    def estimate(
        self,
        atoms: list[Atom],
        gaps: dict[str, float] | None = None,
        scored: Candidate = CANDIDATE,
        temperature: float = 1.0,
    ) -> tuple[dict[str, float], FakeLanguageModel]:
        language_model = FakeLanguageModel(gaps)
        estimator = NegationConsistentYesNoEstimator(
            language_model, temperature=temperature
        )
        return estimator.estimate_atoms(atoms, scored), language_model

    def test_when_atoms_are_scored_then_one_call_asks_about_each_atom_and_negation(
        self,
    ):
        _, language_model = self.estimate([FILM, AMERICAN])

        self.assertEqual(
            language_model.calls,
            [
                (
                    [
                        FILM_PROMPT,
                        NOT_FILM_PROMPT,
                        AMERICAN_PROMPT,
                        NOT_AMERICAN_PROMPT,
                    ],
                    [" yes", " no"],
                )
            ],
        )

    def test_when_content_free_input_is_scored_then_na_fills_document_and_entity(
        self,
    ):
        scored = Candidate(entity="N/A", text="N/A", retrieval_score=0.0)

        _, language_model = self.estimate([FILM], scored=scored)

        self.assertEqual(
            language_model.calls[0][0],
            [
                prompt("'N/A' is a film", document="N/A", entity="'N/A'"),
                prompt("'N/A' is not a film", document="N/A", entity="'N/A'"),
            ],
        )

    def test_when_atoms_are_scored_then_each_gets_a_probability_under_its_name(self):
        result, _ = self.estimate([FILM, AMERICAN])

        self.assertEqual(set(result), {"a0", "a1"})

    def test_when_atom_and_negation_are_asked_then_gaps_are_averaged_in_logit_space(
        self,
    ):
        result, _ = self.estimate(
            [FILM], gaps={FILM_PROMPT: 3.0, NOT_FILM_PROMPT: -1.0}
        )

        self.assertAlmostEqual(result["a0"], sigmoid((3.0 - -1.0) / 2))

    def test_when_model_says_yes_to_both_then_the_bias_cancels(self):
        result, _ = self.estimate([FILM], gaps={FILM_PROMPT: 4.0, NOT_FILM_PROMPT: 4.0})

        self.assertEqual(result["a0"], 0.5)

    def test_when_the_negation_is_better_supported_then_probability_is_below_half(
        self,
    ):
        result, _ = self.estimate(
            [FILM], gaps={FILM_PROMPT: -2.0, NOT_FILM_PROMPT: 2.0}
        )

        self.assertAlmostEqual(result["a0"], sigmoid(-2.0))

    def test_when_each_atom_has_its_own_answers_then_they_are_not_mixed(self):
        result, _ = self.estimate(
            [FILM, AMERICAN],
            gaps={
                FILM_PROMPT: 2.0,
                NOT_FILM_PROMPT: 0.0,
                AMERICAN_PROMPT: 0.0,
                NOT_AMERICAN_PROMPT: 4.0,
            },
        )

        self.assertAlmostEqual(result["a0"], sigmoid(1.0))
        self.assertAlmostEqual(result["a1"], sigmoid(-2.0))

    def test_when_temperature_is_given_then_it_divides_each_gap(self):
        gaps = {FILM_PROMPT: 3.0, NOT_FILM_PROMPT: -1.0}
        for temperature, expected_logit in ((2.0, 1.0), (0.5, 4.0)):
            with self.subTest(temperature=temperature):
                result, _ = self.estimate([FILM], gaps, temperature=temperature)

                self.assertAlmostEqual(result["a0"], sigmoid(expected_logit))

    def test_when_no_temperature_is_given_then_it_is_one(self):
        language_model = FakeLanguageModel({FILM_PROMPT: 3.0, NOT_FILM_PROMPT: -1.0})

        result = NegationConsistentYesNoEstimator(language_model).estimate_atoms(
            [FILM], CANDIDATE
        )

        self.assertAlmostEqual(result["a0"], sigmoid(2.0))

    def test_when_both_answers_are_near_certain_then_their_difference_still_counts(
        self,
    ):
        result, _ = self.estimate(
            [FILM], gaps={FILM_PROMPT: 40.0, NOT_FILM_PROMPT: 38.0}
        )

        self.assertAlmostEqual(result["a0"], sigmoid(1.0))

    def test_when_gaps_are_huge_then_probability_does_not_overflow(self):
        for gaps, expected in (
            ({FILM_PROMPT: 1000.0, NOT_FILM_PROMPT: -1000.0}, 1.0),
            ({FILM_PROMPT: -1000.0, NOT_FILM_PROMPT: 1000.0}, 0.0),
        ):
            with self.subTest(expected=expected):
                result, _ = self.estimate([FILM], gaps, temperature=0.1)

                self.assertAlmostEqual(result["a0"], expected)

    def test_when_temperature_is_not_positive_and_finite_then_error_names_it(self):
        for temperature in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    NegationConsistentYesNoEstimator(
                        FakeLanguageModel(), temperature=temperature
                    )
