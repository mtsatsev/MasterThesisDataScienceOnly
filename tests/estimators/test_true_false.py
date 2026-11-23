import math
import unittest

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.true_false import TrueFalseEstimator
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
QUESTION = "Is this phrase true or false?"


class FakeLanguageModel:
    """Return the same next-token log-probabilities for every prompt."""

    def __init__(self, logprobs: dict[str, float] | None = None):
        self.logprobs = logprobs or {" True": -1.0, " False": -2.0}
        self.calls = []

    def next_token_logprobs(
        self, prompts: list[str], tokens: list[str]
    ) -> list[dict[str, float]]:
        self.calls.append((prompts, tokens))
        return [{token: self.logprobs[token] for token in tokens} for _ in prompts]


def candidate(entity: str = "Blade Runner", text: str = CANDIDATE.text) -> Candidate:
    return Candidate(entity=entity, text=text, retrieval_score=1.0)


class TestTrueFalseEstimator(unittest.TestCase):
    def estimate(
        self,
        atoms: list[Atom],
        scored: Candidate = CANDIDATE,
        logprobs: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], FakeLanguageModel]:
        language_model = FakeLanguageModel(logprobs)
        result = TrueFalseEstimator(language_model).estimate_atoms(atoms, scored)
        return result, language_model

    def prompts(self, atoms: list[Atom], scored: Candidate = CANDIDATE) -> list[str]:
        return self.estimate(atoms, scored)[1].calls[0][0]

    def test_when_atom_is_scored_then_prompt_is_document_then_statement_and_question(
        self,
    ):
        self.assertEqual(
            self.prompts([FILM]),
            [
                "Blade Runner is a 1982 science fiction film.\n"
                f"'Blade Runner' is a film. {QUESTION}"
            ],
        )

    def test_when_statement_ends_with_punctuation_then_it_is_replaced_by_a_full_stop(
        self,
    ):
        for ending in (".", "?", "!"):
            with self.subTest(ending=ending):
                atom = Atom(
                    name="a0",
                    text=f"{{X}} is a film{ending}",
                    negated_text="{X} is not a film",
                )

                self.assertEqual(
                    self.prompts([atom]),
                    [f"{CANDIDATE.text}\n'Blade Runner' is a film. {QUESTION}"],
                )

    def test_when_document_has_surrounding_whitespace_then_it_is_stripped(self):
        scored = candidate(text="  \nBlade Runner is a film.\n\n ")

        self.assertEqual(
            self.prompts([FILM], scored),
            [f"Blade Runner is a film.\n'Blade Runner' is a film. {QUESTION}"],
        )

    def test_when_document_is_empty_then_prompt_is_only_statement_and_question(self):
        for text in ("", "  \n "):
            with self.subTest(text=text):
                self.assertEqual(
                    self.prompts([FILM], candidate(text=text)),
                    [f"'Blade Runner' is a film. {QUESTION}"],
                )

    def test_when_entity_has_an_apostrophe_then_it_is_quoted_like_python_repr(self):
        scored = candidate(entity="Schindler's List", text="A 1993 film.")

        self.assertEqual(
            self.prompts([FILM], scored),
            [f'A 1993 film.\n"Schindler\'s List" is a film. {QUESTION}'],
        )

    def test_when_content_free_input_is_scored_then_na_fills_document_and_entity(
        self,
    ):
        scored = candidate(entity="N/A", text="N/A")

        self.assertEqual(
            self.prompts([FILM], scored), [f"N/A\n'N/A' is a film. {QUESTION}"]
        )

    def test_when_atoms_are_scored_then_all_prompts_go_in_one_call_in_atom_order(
        self,
    ):
        _, language_model = self.estimate([FILM, AMERICAN])

        self.assertEqual(
            language_model.calls,
            [
                (
                    [
                        f"{CANDIDATE.text}\n'Blade Runner' is a film. {QUESTION}",
                        f"{CANDIDATE.text}\n'Blade Runner' is an American film. "
                        f"{QUESTION}",
                    ],
                    [" True", " False"],
                )
            ],
        )

    def test_when_atoms_are_scored_then_each_gets_a_probability_under_its_name(self):
        result, _ = self.estimate([FILM, AMERICAN])

        self.assertEqual(set(result), {"a0", "a1"})

    def test_when_true_is_likelier_then_probability_is_softmax_over_true_and_false(
        self,
    ):
        result, _ = self.estimate([FILM], logprobs={" True": -1.0, " False": -2.0})

        self.assertAlmostEqual(
            result["a0"], math.exp(-1) / (math.exp(-1) + math.exp(-2))
        )

    def test_when_true_and_false_are_equally_likely_then_probability_is_one_half(self):
        result, _ = self.estimate([FILM], logprobs={" True": -3.0, " False": -3.0})

        self.assertEqual(result["a0"], 0.5)

    def test_when_log_probabilities_are_far_apart_then_probability_does_not_overflow(
        self,
    ):
        for logprobs, expected in (
            ({" True": 0.0, " False": -1000.0}, 1.0),
            ({" True": -1000.0, " False": 0.0}, 0.0),
        ):
            with self.subTest(logprobs=logprobs):
                result, _ = self.estimate([FILM], logprobs=logprobs)

                self.assertAlmostEqual(result["a0"], expected)
