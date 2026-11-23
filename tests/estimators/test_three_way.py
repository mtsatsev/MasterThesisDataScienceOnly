import math
import unittest

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.three_way import ThreeWayEstimator
from llm_bayesian_reasoning.records.atoms import Atom
from tests.estimators.fake_language_models import FixedNextTokenModel

FILM = Atom(name="a0", text="{X} is a film", negated_text="{X} is not a film")
AMERICAN = Atom(
    name="a1", text="{X} is an American film.", negated_text="{X} is not American"
)
CANDIDATE = Candidate(
    entity="Blade Runner",
    text="Blade Runner is a 1982 science fiction film.",
    retrieval_score=12.5,
)
WITH_CONTEXT = (
    "Based on the context, the statement is:\n"
    "A) supported\nB) contradicted\nC) not mentioned\nAnswer:"
)
WITHOUT_CONTEXT = "The statement is:\nA) true\nB) false\nC) unknown\nAnswer:"


def prompt(statement: str, document: str | None = CANDIDATE.text, entity=None) -> str:
    entity = entity or "'Blade Runner'"
    if document is None:
        return f"Entity: {entity}\nStatement: {statement}\n\n{WITHOUT_CONTEXT}"
    return (
        f"Context:\n{document}\n\n"
        f"Entity: {entity}\nStatement: {statement}\n\n{WITH_CONTEXT}"
    )


LOGPROBS = {" A": -1.0, " B": -2.0, " C": -3.0}


def candidate(entity: str = "Blade Runner", text: str = CANDIDATE.text) -> Candidate:
    return Candidate(entity=entity, text=text, retrieval_score=1.0)


def softmax(values: list[float]) -> list[float]:
    exponentials = [math.exp(value) for value in values]
    return [value / sum(exponentials) for value in exponentials]


class TestThreeWayEstimator(unittest.TestCase):
    def estimate(
        self,
        atoms: list[Atom],
        scored: Candidate = CANDIDATE,
        logprobs: dict[str, float] | None = None,
        **settings: float,
    ) -> tuple[dict[str, float], FixedNextTokenModel]:
        language_model = FixedNextTokenModel(logprobs or LOGPROBS)
        estimator = ThreeWayEstimator(language_model, **settings)
        return estimator.estimate_atoms(atoms, scored), language_model

    def prompts(self, atoms: list[Atom], scored: Candidate = CANDIDATE) -> list[str]:
        return self.estimate(atoms, scored)[1].calls[0][0]

    def test_when_document_is_given_then_prompt_asks_supported_contradicted_or_not(
        self,
    ):
        self.assertEqual(
            self.prompts([FILM]),
            [
                "Context:\nBlade Runner is a 1982 science fiction film.\n\n"
                "Entity: 'Blade Runner'\n"
                "Statement: 'Blade Runner' is a film\n\n"
                "Based on the context, the statement is:\n"
                "A) supported\nB) contradicted\nC) not mentioned\nAnswer:"
            ],
        )

    def test_when_document_is_empty_then_prompt_asks_true_false_or_unknown(self):
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
                    [" A", " B", " C"],
                )
            ],
        )

    def test_when_atoms_are_scored_then_each_gets_a_probability_under_its_name(self):
        result, _ = self.estimate([FILM, AMERICAN])

        self.assertEqual(set(result), {"a0", "a1"})

    def test_when_no_unknown_weight_is_given_then_probability_is_p_supported(self):
        result, _ = self.estimate([FILM])

        self.assertAlmostEqual(result["a0"], softmax([-1.0, -2.0, -3.0])[0])

    def test_when_unknown_weight_is_given_then_it_adds_that_share_of_p_unknown(self):
        p_supported, _, p_unknown = softmax([-1.0, -2.0, -3.0])
        for weight in (0.5, 1.0):
            with self.subTest(weight=weight):
                result, _ = self.estimate([FILM], unknown_weight=weight)

                self.assertAlmostEqual(result["a0"], p_supported + weight * p_unknown)

    def test_when_temperature_is_given_then_it_divides_the_logprobs(self):
        for temperature in (2.0, 0.5):
            with self.subTest(temperature=temperature):
                result, _ = self.estimate([FILM], temperature=temperature)

                expected = softmax(
                    [-1.0 / temperature, -2.0 / temperature, -3.0 / temperature]
                )
                self.assertAlmostEqual(result["a0"], expected[0])

    def test_when_all_three_are_very_unlikely_then_probability_does_not_divide_by_zero(
        self,
    ):
        result, _ = self.estimate(
            [FILM],
            logprobs={" A": -1000.0, " B": -1000.0, " C": -1000.0},
            temperature=0.1,
        )

        self.assertAlmostEqual(result["a0"], 1 / 3)

    def test_when_one_answer_dominates_then_probability_is_zero_or_one(self):
        for logprobs, expected in (
            ({" A": 0.0, " B": -1000.0, " C": -1000.0}, 1.0),
            ({" A": -1000.0, " B": 0.0, " C": -1000.0}, 0.0),
        ):
            with self.subTest(expected=expected):
                result, _ = self.estimate([FILM], logprobs=logprobs, temperature=0.1)

                self.assertAlmostEqual(result["a0"], expected)

    def test_when_supported_and_unknown_add_up_to_one_then_rounding_stays_at_one(
        self,
    ):
        result, _ = self.estimate(
            [FILM],
            logprobs={" A": 0.0, " B": -40.0, " C": -3.0},
            unknown_weight=1.0,
        )

        self.assertLessEqual(result["a0"], 1.0)

    def test_when_temperature_is_not_positive_and_finite_then_error_names_it(self):
        for temperature in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    ThreeWayEstimator(
                        FixedNextTokenModel(LOGPROBS), temperature=temperature
                    )

    def test_when_unknown_weight_is_outside_zero_and_one_then_error_names_it(self):
        for weight in (-0.1, 1.1, math.nan):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(ValueError, "unknown_weight"):
                    ThreeWayEstimator(
                        FixedNextTokenModel(LOGPROBS), unknown_weight=weight
                    )
