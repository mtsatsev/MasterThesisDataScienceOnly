import math
import unittest

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.channel_bayes_factor import (
    ChannelBayesFactorEstimator,
)
from llm_bayesian_reasoning.records.atoms import Atom

FILM = Atom(name="a0", text="{X} is a film", negated_text="{X} is not a film")
AMERICAN = Atom(
    name="a1", text="{X} is an American film", negated_text="{X} is not American"
)
CANDIDATE = Candidate(
    entity="Blade Runner",
    text="Blade Runner is a 1982 science fiction film.",
    retrieval_score=12.5,
)
ARTICLE = "\n\nWikipedia article:"
LEAD = "\nBlade Runner is a 1982 science fiction film."


def prior_prompt(statement: str, entity: str = "'Blade Runner'") -> str:
    return (
        f"Entity: {entity}\nStatement: {statement}\n\nIs the statement true?\nAnswer:"
    )


FILM_PRIOR = prior_prompt("'Blade Runner' is a film")
FILM_EVIDENCE = (f"Fact: 'Blade Runner' is a film.{ARTICLE}", LEAD)
NOT_FILM_EVIDENCE = (f"Fact: 'Blade Runner' is not a film.{ARTICLE}", LEAD)


class FakeLanguageModel:
    """Return stored yes/no gaps and continuation scores, cut text by characters."""

    def __init__(
        self,
        gaps: dict[str, float] | None = None,
        token_logprobs: dict[tuple[str, str], list[float]] | None = None,
    ):
        self.gaps = gaps or {}
        self.token_logprobs = token_logprobs or {}
        self.next_token_calls = []
        self.continuation_calls = []
        self.truncate_calls = []

    def next_token_logprobs(
        self, prompts: list[str], tokens: list[str]
    ) -> list[dict[str, float]]:
        self.next_token_calls.append((prompts, tokens))
        return [
            {" yes": self.gaps.get(text, 0.0) - 5.0, " no": -5.0} for text in prompts
        ]

    def continuation_token_logprobs(
        self, requests: list[tuple[str, str]]
    ) -> list[list[float]]:
        self.continuation_calls.append(requests)
        return [self.token_logprobs.get(request, [-1.0]) for request in requests]

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        self.truncate_calls.append((text, max_tokens))
        return text[:max_tokens]


def candidate(entity: str = "Blade Runner", text: str = CANDIDATE.text) -> Candidate:
    return Candidate(entity=entity, text=text, retrieval_score=1.0)


def sigmoid(logit: float) -> float:
    return 1 / (1 + math.exp(-logit))


class TestChannelBayesFactorEstimator(unittest.TestCase):
    def estimate(
        self,
        atoms: list[Atom],
        language_model: FakeLanguageModel | None = None,
        scored: Candidate = CANDIDATE,
        **settings: float,
    ) -> tuple[dict[str, float], FakeLanguageModel]:
        language_model = language_model or FakeLanguageModel()
        estimator = ChannelBayesFactorEstimator(language_model, **settings)
        return estimator.estimate_atoms(atoms, scored), language_model

    def test_when_atoms_are_scored_then_one_call_asks_the_prior_question_per_atom(
        self,
    ):
        _, language_model = self.estimate([FILM, AMERICAN])

        self.assertEqual(
            language_model.next_token_calls,
            [
                (
                    [FILM_PRIOR, prior_prompt("'Blade Runner' is an American film")],
                    [" yes", " no"],
                )
            ],
        )

    def test_when_atoms_are_scored_then_one_call_scores_the_lead_under_both_facts(
        self,
    ):
        _, language_model = self.estimate([FILM, AMERICAN])

        self.assertEqual(
            language_model.continuation_calls,
            [
                [
                    FILM_EVIDENCE,
                    NOT_FILM_EVIDENCE,
                    (f"Fact: 'Blade Runner' is an American film.{ARTICLE}", LEAD),
                    (f"Fact: 'Blade Runner' is not American.{ARTICLE}", LEAD),
                ]
            ],
        )

    def test_when_document_is_long_then_only_its_first_tokens_are_scored(self):
        _, language_model = self.estimate([FILM], max_document_tokens=12)

        self.assertEqual(language_model.continuation_calls[0][0][1], "\nBlade Runner")

    def test_when_document_is_cut_then_it_is_stripped_first_and_cut_once(self):
        _, language_model = self.estimate(
            [FILM, AMERICAN], scored=candidate(text="  \nBlade Runner is a film.\n ")
        )

        self.assertEqual(
            language_model.truncate_calls, [("Blade Runner is a film.", 512)]
        )

    def test_when_no_document_length_is_given_then_512_tokens_are_kept(self):
        _, language_model = self.estimate([FILM])

        self.assertEqual(language_model.truncate_calls[0][1], 512)

    def test_when_document_is_empty_then_no_evidence_is_scored(self):
        for text in ("", "  \n "):
            with self.subTest(text=text):
                _, language_model = self.estimate([FILM], scored=candidate(text=text))

                self.assertEqual(language_model.continuation_calls, [])

    def test_when_document_is_empty_then_probability_is_the_prior(self):
        language_model = FakeLanguageModel(gaps={FILM_PRIOR: 1.5})

        result, _ = self.estimate([FILM], language_model, scored=candidate(text=""))

        self.assertAlmostEqual(result["a0"], sigmoid(1.5))

    def test_when_content_free_input_is_scored_then_na_fills_document_and_entity(
        self,
    ):
        _, language_model = self.estimate(
            [FILM], scored=candidate(entity="N/A", text="N/A")
        )

        self.assertEqual(
            language_model.continuation_calls,
            [
                [
                    (f"Fact: 'N/A' is a film.{ARTICLE}", "\nN/A"),
                    (f"Fact: 'N/A' is not a film.{ARTICLE}", "\nN/A"),
                ]
            ],
        )
        self.assertEqual(
            language_model.next_token_calls[0][0],
            [prior_prompt("'N/A' is a film", entity="'N/A'")],
        )

    def test_when_atoms_are_scored_then_each_gets_a_probability_under_its_name(self):
        result, _ = self.estimate([FILM, AMERICAN])

        self.assertEqual(set(result), {"a0", "a1"})

    def test_when_prior_and_evidence_are_given_then_they_add_in_logit_space(self):
        language_model = FakeLanguageModel(
            gaps={FILM_PRIOR: 0.5},
            token_logprobs={
                FILM_EVIDENCE: [-1.0, -1.0],
                NOT_FILM_EVIDENCE: [-2.0, -1.5],
            },
        )

        result, _ = self.estimate([FILM], language_model)

        self.assertAlmostEqual(result["a0"], sigmoid(0.5 + (-2.0 - -3.5)))

    def test_when_temperature_is_given_then_it_divides_only_the_evidence(self):
        language_model = FakeLanguageModel(
            gaps={FILM_PRIOR: 0.5},
            token_logprobs={FILM_EVIDENCE: [-1.0], NOT_FILM_EVIDENCE: [-5.0]},
        )

        result, _ = self.estimate([FILM], language_model, temperature=4.0)

        self.assertAlmostEqual(result["a0"], sigmoid(0.5 + 4.0 / 4.0))

    def test_when_the_negation_explains_the_document_better_then_it_is_below_half(
        self,
    ):
        language_model = FakeLanguageModel(
            token_logprobs={FILM_EVIDENCE: [-6.0], NOT_FILM_EVIDENCE: [-2.0]}
        )

        result, _ = self.estimate([FILM], language_model)

        self.assertAlmostEqual(result["a0"], sigmoid(-4.0))

    def test_when_each_atom_has_its_own_scores_then_they_are_not_mixed(self):
        american_prior = prior_prompt("'Blade Runner' is an American film")
        language_model = FakeLanguageModel(
            gaps={FILM_PRIOR: 1.0, american_prior: -1.0},
            token_logprobs={FILM_EVIDENCE: [-1.0], NOT_FILM_EVIDENCE: [-3.0]},
        )

        result, _ = self.estimate([FILM, AMERICAN], language_model)

        self.assertAlmostEqual(result["a0"], sigmoid(1.0 + 2.0))
        self.assertAlmostEqual(result["a1"], sigmoid(-1.0))

    def test_when_evidence_is_huge_then_probability_does_not_overflow(self):
        for positive, negative, expected in ((0.0, -5000.0, 1.0), (-5000.0, 0.0, 0.0)):
            with self.subTest(expected=expected):
                language_model = FakeLanguageModel(
                    token_logprobs={
                        FILM_EVIDENCE: [positive],
                        NOT_FILM_EVIDENCE: [negative],
                    }
                )

                result, _ = self.estimate([FILM], language_model)

                self.assertAlmostEqual(result["a0"], expected)

    def test_when_temperature_is_not_positive_and_finite_then_error_names_it(self):
        for temperature in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    ChannelBayesFactorEstimator(
                        FakeLanguageModel(), temperature=temperature
                    )

    def test_when_document_length_is_below_one_then_error_names_it(self):
        for max_document_tokens in (0, -5):
            with self.subTest(max_document_tokens=max_document_tokens):
                with self.assertRaisesRegex(ValueError, "max_document_tokens"):
                    ChannelBayesFactorEstimator(
                        FakeLanguageModel(), max_document_tokens=max_document_tokens
                    )
