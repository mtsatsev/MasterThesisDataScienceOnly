import math
import unittest

from llm_bayesian_reasoning.estimators.yes_no import YesNoEstimator

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
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
QUESTION = "Is the statement supported by the context?\nAnswer:"


def prompt(statement: str, document: str | None = CANDIDATE.text, entity=None) -> str:
    entity = entity or "'Blade Runner'"
    context = f"Context:\n{document}\n\n" if document else ""
    return f"{context}Entity: {entity}\nStatement: {statement}\n\n{QUESTION}"


class FakeLanguageModel:
    """Return the same next-token log-probabilities for every prompt."""

    def __init__(self, logprobs: dict[str, float] | None = None):
        self.logprobs = logprobs or {" yes": -1.0, " no": -2.0}
        self.calls = []

    def next_token_logprobs(
        self, prompts: list[str], tokens: list[str]
    ) -> list[dict[str, float]]:
        self.calls.append((prompts, tokens))
        return [{token: self.logprobs[token] for token in tokens} for _ in prompts]


def candidate(entity: str = "Blade Runner", text: str = CANDIDATE.text) -> Candidate:
    return Candidate(entity=entity, text=text, retrieval_score=1.0)


class TestYesNoEstimator(unittest.TestCase):
    def estimate(
        self,
        atoms: list[Atom],
        scored: Candidate = CANDIDATE,
        logprobs: dict[str, float] | None = None,
        temperature: float = 1.0,
    ) -> tuple[dict[str, float], FakeLanguageModel]:
        language_model = FakeLanguageModel(logprobs)
        estimator = YesNoEstimator(language_model, temperature=temperature)
        return estimator.estimate_atoms(atoms, scored), language_model

    def prompts(self, atoms: list[Atom], scored: Candidate = CANDIDATE) -> list[str]:
        return self.estimate(atoms, scored)[1].calls[0][0]

    def test_when_atom_is_scored_then_prompt_has_context_entity_statement_question(
        self,
    ):
        self.assertEqual(
            self.prompts([FILM]),
            [
                "Context:\nBlade Runner is a 1982 science fiction film.\n\n"
                "Entity: 'Blade Runner'\n"
                "Statement: 'Blade Runner' is a film\n\n"
                "Is the statement supported by the context?\nAnswer:"
            ],
        )

    def test_when_statement_ends_with_punctuation_then_it_is_kept(self):
        self.assertEqual(
            self.prompts([AMERICAN]), [prompt("'Blade Runner' is an American film.")]
        )

    def test_when_document_has_surrounding_whitespace_then_it_is_stripped(self):
        scored = candidate(text="  \nBlade Runner is a film.\n\n ")

        self.assertEqual(
            self.prompts([FILM], scored),
            [prompt("'Blade Runner' is a film", document="Blade Runner is a film.")],
        )

    def test_when_document_is_empty_then_prompt_has_no_context_block(self):
        for text in ("", "  \n "):
            with self.subTest(text=text):
                self.assertEqual(
                    self.prompts([FILM], candidate(text=text)),
                    [prompt("'Blade Runner' is a film", document=None)],
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

    def test_when_atoms_are_scored_then_all_prompts_go_in_one_call_in_atom_order(
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
                    [" yes", " no"],
                )
            ],
        )

    def test_when_atoms_are_scored_then_each_gets_a_probability_under_its_name(self):
        result, _ = self.estimate([FILM, AMERICAN])

        self.assertEqual(set(result), {"a0", "a1"})

    def test_when_yes_is_likelier_then_probability_is_sigmoid_of_the_gap(self):
        result, _ = self.estimate([FILM], logprobs={" yes": -1.0, " no": -2.0})

        self.assertAlmostEqual(result["a0"], 1 / (1 + math.exp(-1.0)))

    def test_when_temperature_is_given_then_it_divides_the_gap(self):
        for temperature, expected_logit in ((2.0, 0.5), (0.5, 2.0)):
            with self.subTest(temperature=temperature):
                result, _ = self.estimate(
                    [FILM],
                    logprobs={" yes": -1.0, " no": -2.0},
                    temperature=temperature,
                )

                self.assertAlmostEqual(
                    result["a0"], 1 / (1 + math.exp(-expected_logit))
                )

    def test_when_no_temperature_is_given_then_it_is_one(self):
        language_model = FakeLanguageModel({" yes": -1.0, " no": -2.0})

        result = YesNoEstimator(language_model).estimate_atoms([FILM], CANDIDATE)

        self.assertAlmostEqual(result["a0"], 1 / (1 + math.exp(-1.0)))

    def test_when_yes_and_no_are_equally_likely_then_probability_is_one_half(self):
        result, _ = self.estimate([FILM], logprobs={" yes": -3.0, " no": -3.0})

        self.assertEqual(result["a0"], 0.5)

    def test_when_log_probabilities_are_far_apart_then_probability_does_not_overflow(
        self,
    ):
        for logprobs, expected in (
            ({" yes": 0.0, " no": -1000.0}, 1.0),
            ({" yes": -1000.0, " no": 0.0}, 0.0),
        ):
            with self.subTest(logprobs=logprobs):
                result, _ = self.estimate([FILM], logprobs=logprobs, temperature=0.1)

                self.assertAlmostEqual(result["a0"], expected)

    def test_when_temperature_is_not_positive_and_finite_then_error_names_it(self):
        for temperature in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    YesNoEstimator(FakeLanguageModel(), temperature=temperature)
