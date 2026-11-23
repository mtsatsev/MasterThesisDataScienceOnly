import math
import unittest

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.entity_likelihood_pmi import (
    EntityLikelihoodPMIEstimator,
)
from llm_bayesian_reasoning.records.atoms import Atom
from tests.estimators.fake_language_models import StoredContinuationModel

FILM = Atom(name="a0", text="{X} is a film", negated_text="{X} is not a film")
AMERICAN = Atom(
    name="a1", text="{X} is an American film", negated_text="{X} is not American"
)
CANDIDATE = Candidate(
    entity="Blade Runner",
    text="Blade Runner is a 1982 science fiction film.",
    retrieval_score=12.5,
)
NAME = " Blade Runner"


def prompt(property_text: str) -> str:
    return f"Name something for which the following is true: {property_text}.\nAnswer:"


BASELINE = (prompt("it exists"), NAME)
FILM_REQUEST = (prompt("it is a film"), NAME)
AMERICAN_REQUEST = (prompt("it is an American film"), NAME)


def candidate(entity: str = "Blade Runner", text: str = CANDIDATE.text) -> Candidate:
    return Candidate(entity=entity, text=text, retrieval_score=1.0)


def sigmoid(logit: float) -> float:
    return 1 / (1 + math.exp(-logit))


class TestEntityLikelihoodPMIEstimator(unittest.TestCase):
    def estimate(
        self,
        atoms: list[Atom],
        token_logprobs: dict[tuple[str, str], list[float]] | None = None,
        scored: Candidate = CANDIDATE,
        temperature: float = 1.0,
    ) -> tuple[dict[str, float], StoredContinuationModel]:
        language_model = StoredContinuationModel(token_logprobs)
        estimator = EntityLikelihoodPMIEstimator(
            language_model, temperature=temperature
        )
        return estimator.estimate_atoms(atoms, scored), language_model

    def requests(self, atoms: list[Atom], scored: Candidate = CANDIDATE) -> list:
        return self.estimate(atoms, scored=scored)[1].calls

    def test_when_atoms_are_scored_then_one_call_scores_the_name_after_each_property(
        self,
    ):
        self.assertEqual(
            self.requests([FILM, AMERICAN]),
            [[BASELINE, FILM_REQUEST, AMERICAN_REQUEST]],
        )

    def test_when_entity_has_a_disambiguation_suffix_then_it_is_removed(self):
        for entity, name in (
            ("Blade Runner (film)", " Blade Runner"),
            ("Blade Runner (1982 film)", " Blade Runner"),
            ("Kill Bill (Volume 1) (film)", " Kill Bill (Volume 1)"),
            ("The (Sound) of Music", " The (Sound) of Music"),
        ):
            with self.subTest(entity=entity):
                calls = self.requests([FILM], candidate(entity=entity))

                self.assertEqual({request[1] for request in calls[0]}, {name})

    def test_when_entity_has_an_apostrophe_then_the_name_is_not_quoted(self):
        calls = self.requests([FILM], candidate(entity="Schindler's List"))

        self.assertEqual(calls[0][0], (prompt("it exists"), " Schindler's List"))

    def test_when_document_changes_then_the_requests_do_not(self):
        for text in ("", "A completely different document."):
            with self.subTest(text=text):
                self.assertEqual(
                    self.requests([FILM], candidate(text=text)),
                    [[BASELINE, FILM_REQUEST]],
                )

    def test_when_content_free_input_is_scored_then_na_is_the_name(self):
        self.assertEqual(
            self.requests([FILM], candidate(entity="N/A", text="N/A")),
            [[(prompt("it exists"), " N/A"), (prompt("it is a film"), " N/A")]],
        )

    def test_when_atoms_are_scored_then_each_gets_a_probability_under_its_name(self):
        result, _ = self.estimate([FILM, AMERICAN])

        self.assertEqual(set(result), {"a0", "a1"})

    def test_when_property_raises_the_name_likelihood_then_probability_is_above_half(
        self,
    ):
        result, _ = self.estimate(
            [FILM], {BASELINE: [-3.0, -2.0], FILM_REQUEST: [-1.0, -1.0]}
        )

        self.assertAlmostEqual(result["a0"], sigmoid(-2.0 - -5.0))

    def test_when_property_lowers_the_name_likelihood_then_probability_is_below_half(
        self,
    ):
        result, _ = self.estimate([FILM], {BASELINE: [-1.0], FILM_REQUEST: [-4.0]})

        self.assertAlmostEqual(result["a0"], sigmoid(-3.0))

    def test_when_several_atoms_are_scored_then_they_share_the_one_baseline(self):
        result, _ = self.estimate(
            [FILM, AMERICAN],
            {BASELINE: [-3.0], FILM_REQUEST: [-1.0], AMERICAN_REQUEST: [-5.0]},
        )

        self.assertAlmostEqual(result["a0"], sigmoid(2.0))
        self.assertAlmostEqual(result["a1"], sigmoid(-2.0))

    def test_when_temperature_is_given_then_it_divides_the_gain(self):
        scores = {BASELINE: [-3.0], FILM_REQUEST: [-1.0]}
        for temperature, expected_logit in ((2.0, 1.0), (0.5, 4.0)):
            with self.subTest(temperature=temperature):
                result, _ = self.estimate([FILM], scores, temperature=temperature)

                self.assertAlmostEqual(result["a0"], sigmoid(expected_logit))

    def test_when_no_temperature_is_given_then_it_is_one(self):
        language_model = StoredContinuationModel(
            {BASELINE: [-3.0], FILM_REQUEST: [-1.0]}
        )

        result = EntityLikelihoodPMIEstimator(language_model).estimate_atoms(
            [FILM], CANDIDATE
        )

        self.assertAlmostEqual(result["a0"], sigmoid(2.0))

    def test_when_the_gain_is_huge_then_probability_does_not_overflow(self):
        for specific, baseline, expected in ((0.0, -1000.0, 1.0), (-1000.0, 0.0, 0.0)):
            with self.subTest(expected=expected):
                result, _ = self.estimate(
                    [FILM],
                    {BASELINE: [baseline], FILM_REQUEST: [specific]},
                    temperature=0.1,
                )

                self.assertAlmostEqual(result["a0"], expected)

    def test_when_temperature_is_not_positive_and_finite_then_error_names_it(self):
        for temperature in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    EntityLikelihoodPMIEstimator(
                        StoredContinuationModel(), temperature=temperature
                    )
