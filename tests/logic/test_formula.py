import unittest

from problog.program import PrologString

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


class TestBuildProblogProgram(unittest.TestCase):
    def test_when_one_atom_is_given_then_program_is_its_fact_rule_and_query(self):
        program = build_problog_program("a", {"a": 0.8})

        self.assertEqual(program, "0.8::a.\nformula_holds :- a.\nquery(formula_holds).")

    def test_when_several_atoms_are_given_then_each_gets_a_fact_line_in_input_order(
        self,
    ):
        program = build_problog_program("b, a", {"b": 0.5, "a": 0.8})

        self.assertEqual(
            program,
            "0.5::b.\n0.8::a.\nformula_holds :- b, a.\nquery(formula_holds).",
        )

    def test_when_formula_has_logical_operators_then_rule_body_keeps_them_unchanged(
        self,
    ):
        for formula in ("a, b", "a; b", "\\+a", "a; (a, b)", "\\+(a, b)"):
            with self.subTest(formula=formula):
                program = build_problog_program(formula, {"a": 0.3, "b": 0.9})

                self.assertIn(f"formula_holds :- {formula}.", program.splitlines())

    def test_when_formula_has_logical_operators_then_problog_parses_every_statement(
        self,
    ):
        for formula in ("a, b", "a; b", "\\+a", "a; (a, b)", "\\+(a, b)"):
            with self.subTest(formula=formula):
                program = build_problog_program(formula, {"a": 0.3, "b": 0.9})

                statements = list(PrologString(program))

                self.assertEqual(len(statements), 4)

    def test_when_probability_is_zero_one_or_very_small_then_problog_reads_it_back_unchanged(
        self,
    ):
        for probability in (0.0, 1.0, 1e-12):
            with self.subTest(probability=probability):
                program = build_problog_program("a", {"a": probability})

                fact = list(PrologString(program))[0]

                self.assertEqual(float(fact.probability), probability)
