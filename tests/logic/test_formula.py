import re
import unittest

from problog.engine import UnknownClause
from problog.logic import Term
from problog.parser import ParseError
from problog.program import SimpleProgram

from llm_bayesian_reasoning.logic.formula import (
    build_problog_program,
    evaluate_formula,
)


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

    def test_when_atom_probability_is_zero_then_formula_probability_is_zero(self):
        probability = evaluate_formula("a", {"a": 0.0})

        self.assertAlmostEqual(probability, 0.0)

    def test_when_atom_probability_is_one_then_formula_probability_is_one(self):
        probability = evaluate_formula("a", {"a": 1.0})

        self.assertAlmostEqual(probability, 1.0)


class TestEvaluateFormulaErrorReporting(unittest.TestCase):
    def test_when_formula_references_unscored_atom_then_error_names_that_atom(self):
        with self.assertRaisesRegex(ValueError, "unscored atom 'missing_atom'"):
            evaluate_formula("scored_atom, missing_atom", {"scored_atom": 0.5})

    def test_when_formula_references_unscored_atom_then_error_quotes_the_formula(self):
        formula = "scored_atom, missing_atom"

        with self.assertRaisesRegex(ValueError, re.escape(formula)):
            evaluate_formula(formula, {"scored_atom": 0.5})

    def test_when_formula_references_unscored_atom_then_problog_error_is_the_cause(
        self,
    ):
        with self.assertRaises(ValueError) as caught:
            evaluate_formula("scored_atom, missing_atom", {"scored_atom": 0.5})

        self.assertIsInstance(caught.exception.__cause__, UnknownClause)

    def test_when_atom_probability_is_outside_unit_interval_then_error_names_atom_and_value(
        self,
    ):
        for invalid_probability in (-0.1, 1.5):
            with self.subTest(probability=invalid_probability):
                expected = f"'bad_atom'.*{re.escape(str(invalid_probability))}"

                with self.assertRaisesRegex(ValueError, expected):
                    evaluate_formula("bad_atom", {"bad_atom": invalid_probability})

    def test_when_one_atom_probability_is_invalid_then_error_does_not_blame_valid_atoms(
        self,
    ):
        with self.assertRaises(ValueError) as caught:
            evaluate_formula(
                "good_atom, bad_atom",
                {"good_atom": 0.5, "bad_atom": 1.5},
            )

        self.assertNotIn("good_atom", str(caught.exception))

    def test_when_formula_is_malformed_then_value_error_quotes_the_formula(self):
        for formula in ("a, (b", "a b", "a,,b"):
            with self.subTest(formula=formula):
                with self.assertRaisesRegex(ValueError, re.escape(formula)):
                    evaluate_formula(formula, {"a": 0.5, "b": 0.5})

    def test_when_formula_is_malformed_then_problog_error_is_the_cause(self):
        with self.assertRaises(ValueError) as caught:
            evaluate_formula("a, (b", {"a": 0.5, "b": 0.5})

        self.assertIsInstance(caught.exception.__cause__, ParseError)


class TestBuildProblogProgram(unittest.TestCase):
    def test_when_program_is_built_then_it_is_a_problog_simple_program(self):
        program = build_problog_program("a", {"a": 0.8})

        self.assertIsInstance(program, SimpleProgram)

    def test_when_one_atom_is_given_then_program_is_its_fact_rule_and_query(self):
        program = build_problog_program("a", {"a": 0.8})

        self.assertEqual(
            [str(statement) for statement in program],
            ["0.8::a", "formula_holds :- a", "query(formula_holds)"],
        )

    def test_when_several_atoms_are_given_then_each_gets_a_fact_in_input_order(self):
        program = build_problog_program("b, a", {"b": 0.5, "a": 0.8})

        facts = list(program)[:2]

        self.assertEqual([str(fact) for fact in facts], ["0.5::b", "0.8::a"])

    def test_when_formula_has_logical_operators_then_rule_body_has_matching_terms(self):
        a, b = Term("a"), Term("b")
        expected_bodies = {
            "a, b": a & b,
            "a; b": a | b,
            "\\+a": ~a,
            "a; (a, b)": a | (a & b),
            "\\+(a, b)": ~(a & b),
        }

        for formula, expected_body in expected_bodies.items():
            with self.subTest(formula=formula):
                program = build_problog_program(formula, {"a": 0.3, "b": 0.9})

                rule = list(program)[2]

                self.assertEqual(rule, Term("formula_holds") << expected_body)

    def test_when_probability_is_zero_one_or_very_small_then_fact_keeps_the_exact_value(
        self,
    ):
        for probability in (0.0, 1.0, 1e-12):
            with self.subTest(probability=probability):
                program = build_problog_program("a", {"a": probability})

                fact = list(program)[0]

                self.assertEqual(float(fact.probability), probability)
