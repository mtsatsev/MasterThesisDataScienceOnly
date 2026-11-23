import math
import unittest

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.expected_verbalized import (
    ExpectedVerbalizedEstimator,
)
from llm_bayesian_reasoning.records.atoms import Atom

FILM = Atom(name="a0", text="{X} is a film", negated_text="{X} is not a film")
AMERICAN = Atom(
    name="a1", text="{X} is an American film.", negated_text="{X} is not American"
)
CANDIDATE = Candidate(
    entity="Blade Runner",
    text="Blade Runner is a 1982 science fiction film.",
    retrieval_score=12.5,
)
QUESTION = (
    "On a scale from 0 to 10, how likely is it that the statement is true?\nAnswer:"
)
SCORE_TOKENS = [f" {score}" for score in range(11)]


def prompt(statement: str, document: str | None = CANDIDATE.text, entity=None) -> str:
    entity = entity or "'Blade Runner'"
    context = f"Context:\n{document}\n\n" if document else ""
    return f"{context}Entity: {entity}\nStatement: {statement}\n\n{QUESTION}"


def only(score: int) -> dict[str, float]:
    """Return log-probabilities that put nearly all mass on one score."""
    return {token: 0.0 if token == f" {score}" else -1000.0 for token in SCORE_TOKENS}


class FakeLanguageModel:
    """Return the same next-token log-probabilities for every prompt."""

    def __init__(self, logprobs: dict[str, float] | None = None):
        self.logprobs = logprobs or {token: -2.0 for token in SCORE_TOKENS}
        self.calls = []

    def next_token_logprobs(
        self, prompts: list[str], tokens: list[str]
    ) -> list[dict[str, float]]:
        self.calls.append((prompts, tokens))
        return [{token: self.logprobs[token] for token in tokens} for _ in prompts]


def candidate(entity: str = "Blade Runner", text: str = CANDIDATE.text) -> Candidate:
    return Candidate(entity=entity, text=text, retrieval_score=1.0)


class TestExpectedVerbalizedEstimator(unittest.TestCase):
    def estimate(
        self,
        atoms: list[Atom],
        scored: Candidate = CANDIDATE,
        logprobs: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], FakeLanguageModel]:
        language_model = FakeLanguageModel(logprobs)
        estimator = ExpectedVerbalizedEstimator(language_model)
        return estimator.estimate_atoms(atoms, scored), language_model

    def prompts(self, atoms: list[Atom], scored: Candidate = CANDIDATE) -> list[str]:
        return self.estimate(atoms, scored)[1].calls[0][0]

    def test_when_document_is_given_then_prompt_asks_for_a_score_from_0_to_10(self):
        self.assertEqual(
            self.prompts([FILM]),
            [
                "Context:\nBlade Runner is a 1982 science fiction film.\n\n"
                "Entity: 'Blade Runner'\n"
                "Statement: 'Blade Runner' is a film\n\n"
                "On a scale from 0 to 10, how likely is it that the statement is "
                "true?\nAnswer:"
            ],
        )

    def test_when_document_is_empty_then_prompt_has_no_context_block(self):
        for text in ("", "  \n "):
            with self.subTest(text=text):
                self.assertEqual(
                    self.prompts([FILM], candidate(text=text)),
                    [prompt("'Blade Runner' is a film", document=None)],
                )

    def test_when_document_has_surrounding_whitespace_then_it_is_stripped(self):
        scored = candidate(text="  \nBlade Runner is a film.\n\n ")

        self.assertEqual(
            self.prompts([FILM], scored),
            [prompt("'Blade Runner' is a film", document="Blade Runner is a film.")],
        )

    def test_when_entity_has_an_apostrophe_then_it_is_quoted_like_python_repr(self):
        scored = candidate(entity="Schindler's List", text="A 1993 film.")

        self.assertEqual(
            self.prompts([FILM], scored),
            [
                prompt(
                    '"Schindler\'s List" is a film',
                    document="A 1993 film.",
                    entity='"Schindler\'s List"',
                )
            ],
        )

    def test_when_content_free_input_is_scored_then_na_fills_document_and_entity(
        self,
    ):
        scored = candidate(entity="N/A", text="N/A")

        self.assertEqual(
            self.prompts([FILM], scored),
            [prompt("'N/A' is a film", document="N/A", entity="'N/A'")],
        )

    def test_when_atoms_are_scored_then_all_prompts_go_in_one_call_with_the_scores(
        self,
    ):
        _, language_model = self.estimate([FILM, AMERICAN])

        self.assertEqual(
            language_model.calls,
            [
                (
                    [
                        prompt("'Blade Runner' is a film"),
                        prompt("'Blade Runner' is an American film."),
                    ],
                    [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " 10"],
                )
            ],
        )

    def test_when_atoms_are_scored_then_each_gets_a_probability_under_its_name(self):
        result, _ = self.estimate([FILM, AMERICAN])

        self.assertEqual(set(result), {"a0", "a1"})

    def test_when_one_score_takes_all_the_mass_then_probability_is_that_score_tenth(
        self,
    ):
        for score in (0, 3, 7, 10):
            with self.subTest(score=score):
                result, _ = self.estimate([FILM], logprobs=only(score))

                self.assertAlmostEqual(result["a0"], score / 10)

    def test_when_scores_are_mixed_then_probability_is_the_expected_score_tenth(self):
        logprobs = {token: -1000.0 for token in SCORE_TOKENS}
        logprobs[" 2"] = math.log(0.25)
        logprobs[" 8"] = math.log(0.75)

        result, _ = self.estimate([FILM], logprobs=logprobs)

        self.assertAlmostEqual(result["a0"], 0.25 * 0.2 + 0.75 * 0.8)

    def test_when_scores_hold_little_mass_then_they_are_renormalized(self):
        logprobs = {token: -1000.0 for token in SCORE_TOKENS}
        logprobs[" 2"] = math.log(0.01)
        logprobs[" 8"] = math.log(0.03)

        result, _ = self.estimate([FILM], logprobs=logprobs)

        self.assertAlmostEqual(result["a0"], 0.25 * 0.2 + 0.75 * 0.8)

    def test_when_every_score_is_equally_likely_then_probability_is_one_half(self):
        result, _ = self.estimate([FILM])

        self.assertAlmostEqual(result["a0"], 0.5)

    def test_when_every_score_is_extremely_unlikely_then_it_does_not_divide_by_zero(
        self,
    ):
        logprobs = {token: -100000.0 for token in SCORE_TOKENS}

        result, _ = self.estimate([FILM], logprobs=logprobs)

        self.assertAlmostEqual(result["a0"], 0.5)
