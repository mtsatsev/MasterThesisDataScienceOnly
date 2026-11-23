import math
import unittest

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.pmi_evidence_gain import (
    PMIEvidenceGainEstimator,
)
from llm_bayesian_reasoning.records.atoms import Atom
from tests.estimators.fake_language_models import StoredContinuationModel

FILM = Atom(name="a0", text="{X} is a film", negated_text="{X} is not a film")
AMERICAN = Atom(
    name="a1", text="{X} is an American film.", negated_text="{X} is not American"
)
CANDIDATE = Candidate(
    entity="Blade Runner",
    text="Blade Runner is a 1982 science fiction film.",
    retrieval_score=12.5,
)
HEAD = "Fact: 'Blade Runner'"
WITH_DOCUMENT = f"Context:\n{CANDIDATE.text}\n\n{HEAD}"


def candidate(entity: str = "Blade Runner", text: str = CANDIDATE.text) -> Candidate:
    return Candidate(entity=entity, text=text, retrieval_score=1.0)


def sigmoid(logit: float) -> float:
    return 1 / (1 + math.exp(-logit))


class TestPMIEvidenceGainEstimator(unittest.TestCase):
    def estimate(
        self,
        atoms: list[Atom],
        token_logprobs: dict[tuple[str, str], list[float]] | None = None,
        scored: Candidate = CANDIDATE,
        temperature: float = 1.0,
    ) -> tuple[dict[str, float], StoredContinuationModel]:
        language_model = StoredContinuationModel(token_logprobs)
        estimator = PMIEvidenceGainEstimator(language_model, temperature=temperature)
        return estimator.estimate_atoms(atoms, scored), language_model

    def requests(self, atoms: list[Atom], scored: Candidate = CANDIDATE) -> list:
        return self.estimate(atoms, scored=scored)[1].calls

    def test_when_atoms_are_scored_then_one_call_scores_each_with_and_without_document(
        self,
    ):
        self.assertEqual(
            self.requests([FILM, AMERICAN]),
            [
                [
                    (WITH_DOCUMENT, " is a film"),
                    (HEAD, " is a film"),
                    (WITH_DOCUMENT, " is an American film."),
                    (HEAD, " is an American film."),
                ]
            ],
        )

    def test_when_atom_has_text_before_the_entity_then_it_goes_in_the_prefix(self):
        atom = Atom(
            name="a0",
            text="The director of {X} is American",
            negated_text="The director of {X} is not American",
        )
        head = "Fact: The director of 'Blade Runner'"

        self.assertEqual(
            self.requests([atom]),
            [
                [
                    (f"Context:\n{CANDIDATE.text}\n\n{head}", " is American"),
                    (head, " is American"),
                ]
            ],
        )

    def test_when_document_has_surrounding_whitespace_then_it_is_stripped(self):
        scored = candidate(text="  \nBlade Runner is a film.\n\n ")

        self.assertEqual(
            self.requests([FILM], scored)[0][0],
            (f"Context:\nBlade Runner is a film.\n\n{HEAD}", " is a film"),
        )

    def test_when_document_is_empty_then_both_requests_have_no_context(self):
        for text in ("", "  \n "):
            with self.subTest(text=text):
                self.assertEqual(
                    self.requests([FILM], candidate(text=text)),
                    [[(HEAD, " is a film"), (HEAD, " is a film")]],
                )

    def test_when_entity_has_an_apostrophe_then_it_is_quoted_like_python_repr(self):
        scored = candidate(entity="Schindler's List", text="A 1993 film.")

        self.assertEqual(
            self.requests([FILM], scored)[0][1],
            ('Fact: "Schindler\'s List"', " is a film"),
        )

    def test_when_content_free_input_is_scored_then_na_fills_document_and_entity(
        self,
    ):
        scored = candidate(entity="N/A", text="N/A")

        self.assertEqual(
            self.requests([FILM], scored),
            [
                [
                    ("Context:\nN/A\n\nFact: 'N/A'", " is a film"),
                    ("Fact: 'N/A'", " is a film"),
                ]
            ],
        )

    def test_when_atoms_are_scored_then_each_gets_a_probability_under_its_name(self):
        result, _ = self.estimate([FILM, AMERICAN])

        self.assertEqual(set(result), {"a0", "a1"})

    def test_when_document_raises_the_likelihood_then_probability_is_above_half(self):
        result, _ = self.estimate(
            [FILM],
            {
                (WITH_DOCUMENT, " is a film"): [-0.5, -0.5],
                (HEAD, " is a film"): [-1.0, -1.0],
            },
        )

        self.assertAlmostEqual(result["a0"], sigmoid(-1.0 - -2.0))

    def test_when_continuation_has_several_tokens_then_their_sum_is_used(self):
        result, _ = self.estimate(
            [FILM],
            {
                (WITH_DOCUMENT, " is a film"): [-1.0, -1.0, -1.0],
                (HEAD, " is a film"): [-2.0, -2.0, -2.0],
            },
        )

        self.assertAlmostEqual(result["a0"], sigmoid(3.0))

    def test_when_document_lowers_the_likelihood_then_probability_is_below_half(self):
        result, _ = self.estimate(
            [FILM],
            {(WITH_DOCUMENT, " is a film"): [-4.0], (HEAD, " is a film"): [-1.0]},
        )

        self.assertAlmostEqual(result["a0"], sigmoid(-3.0))

    def test_when_document_is_empty_then_probability_is_one_half(self):
        result, _ = self.estimate(
            [FILM], {(HEAD, " is a film"): [-2.5]}, scored=candidate(text="")
        )

        self.assertEqual(result["a0"], 0.5)

    def test_when_each_atom_has_its_own_scores_then_they_are_not_mixed(self):
        result, _ = self.estimate(
            [FILM, AMERICAN],
            {
                (WITH_DOCUMENT, " is a film"): [-1.0],
                (HEAD, " is a film"): [-3.0],
                (WITH_DOCUMENT, " is an American film."): [-3.0],
                (HEAD, " is an American film."): [-1.0],
            },
        )

        self.assertAlmostEqual(result["a0"], sigmoid(2.0))
        self.assertAlmostEqual(result["a1"], sigmoid(-2.0))

    def test_when_temperature_is_given_then_it_divides_the_gain(self):
        scores = {(WITH_DOCUMENT, " is a film"): [-1.0], (HEAD, " is a film"): [-3.0]}
        for temperature, expected_logit in ((2.0, 1.0), (0.5, 4.0)):
            with self.subTest(temperature=temperature):
                result, _ = self.estimate([FILM], scores, temperature=temperature)

                self.assertAlmostEqual(result["a0"], sigmoid(expected_logit))

    def test_when_no_temperature_is_given_then_it_is_one(self):
        language_model = StoredContinuationModel(
            {(WITH_DOCUMENT, " is a film"): [-1.0], (HEAD, " is a film"): [-3.0]}
        )

        result = PMIEvidenceGainEstimator(language_model).estimate_atoms(
            [FILM], CANDIDATE
        )

        self.assertAlmostEqual(result["a0"], sigmoid(2.0))

    def test_when_the_gain_is_huge_then_probability_does_not_overflow(self):
        for with_document, without, expected in (
            (0.0, -1000.0, 1.0),
            (-1000.0, 0.0, 0.0),
        ):
            with self.subTest(expected=expected):
                result, _ = self.estimate(
                    [FILM],
                    {
                        (WITH_DOCUMENT, " is a film"): [with_document],
                        (HEAD, " is a film"): [without],
                    },
                    temperature=0.1,
                )

                self.assertAlmostEqual(result["a0"], expected)

    def test_when_temperature_is_not_positive_and_finite_then_error_names_it(self):
        for temperature in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    PMIEvidenceGainEstimator(
                        StoredContinuationModel(), temperature=temperature
                    )
