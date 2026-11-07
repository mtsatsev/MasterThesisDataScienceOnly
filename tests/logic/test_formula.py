import unittest

from llm_bayesian_reasoning.logic.formula import evaluate_formula


class TestEvaluateFormula(unittest.TestCase):
    def test_when_two_atoms_are_conjoined_then_probability_is_their_product(self):
        probability = evaluate_formula("a, b", {"a": 0.8, "b": 0.5})

        self.assertAlmostEqual(probability, 0.4)

    def test_when_two_atoms_are_disjoined_then_probability_is_their_noisy_or(self):
        probability = evaluate_formula("a; b", {"a": 0.8, "b": 0.5})

        self.assertAlmostEqual(probability, 0.9)

    def test_when_atom_is_negated_then_probability_is_its_complement(self):
        probability = evaluate_formula("\\+a", {"a": 0.8})

        self.assertAlmostEqual(probability, 0.2)

    def test_when_atom_is_repeated_in_conjunction_then_it_is_counted_once(self):
        probability = evaluate_formula("a, a", {"a": 0.5})

        self.assertAlmostEqual(probability, 0.5)

    def test_when_atom_is_absorbed_by_disjunction_then_probability_equals_that_atom(
        self,
    ):
        probability = evaluate_formula("a; (a, b)", {"a": 0.3, "b": 0.9})

        self.assertAlmostEqual(probability, 0.3)

    def test_when_formula_references_unscored_atom_then_error_is_raised(self):
        with self.assertRaises(ValueError):
            evaluate_formula("a, b", {"a": 0.8})

    def test_when_atom_probability_is_outside_unit_interval_then_error_is_raised(self):
        for invalid_probability in (-0.1, 1.5):
            with self.subTest(probability=invalid_probability):
                with self.assertRaises(ValueError):
                    evaluate_formula("a", {"a": invalid_probability})
