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


class TestEvaluateFormulaDatasetShapes(unittest.TestCase):
    """Logical patterns found in the QUEST test queries, with hand-computed values."""

    def test_when_formula_is_a_single_atom_then_probability_is_that_atom(self):
        probability = evaluate_formula("a", {"a": 0.8})

        self.assertAlmostEqual(probability, 0.8)

    def test_when_three_atoms_are_conjoined_then_probability_is_their_product(self):
        probability = evaluate_formula("a, b, c", {"a": 0.5, "b": 0.8, "c": 0.5})

        self.assertAlmostEqual(probability, 0.2)

    def test_when_disjoined_conjunctions_use_nine_distinct_atoms_then_probability_is_exact(
        self,
    ):
        names = [f"a{index}" for index in range(9)]

        probability = evaluate_formula(
            "(a0, a1, a2, a3); (a0, a4, a5, a6, a7, a8)",
            dict.fromkeys(names, 0.5),
        )

        self.assertAlmostEqual(probability, 0.076171875)

    def test_when_three_atoms_are_disjoined_then_probability_is_their_noisy_or(self):
        probability = evaluate_formula("a; b; c", {"a": 0.5, "b": 0.8, "c": 0.5})

        self.assertAlmostEqual(probability, 0.95)

    def test_when_conjunction_excludes_an_atom_then_its_complement_is_multiplied(self):
        probability = evaluate_formula("a, \\+b", {"a": 0.5, "b": 0.8})

        self.assertAlmostEqual(probability, 0.1)

    def test_when_conjunction_excludes_two_atoms_then_both_complements_are_multiplied(
        self,
    ):
        probability = evaluate_formula(
            "a, \\+b, \\+c", {"a": 0.5, "b": 0.8, "c": 0.5}
        )

        self.assertAlmostEqual(probability, 0.05)

    def test_when_conjunction_excludes_a_conjunction_then_group_complement_is_used(
        self,
    ):
        probability = evaluate_formula(
            "a, \\+(b, c)", {"a": 0.5, "b": 0.8, "c": 0.5}
        )

        self.assertAlmostEqual(probability, 0.3)

    def test_when_conjunction_excludes_a_disjunction_then_group_complement_is_used(
        self,
    ):
        probability = evaluate_formula(
            "a, \\+(b; c)", {"a": 0.5, "b": 0.8, "c": 0.5}
        )

        self.assertAlmostEqual(probability, 0.05)

    def test_when_conjunction_contains_a_disjunction_then_disjunction_is_a_noisy_or(
        self,
    ):
        probability = evaluate_formula("a, (b; c)", {"a": 0.5, "b": 0.8, "c": 0.5})

        self.assertAlmostEqual(probability, 0.45)

    def test_when_disjunction_contains_a_conjunction_then_conjunction_is_a_product(
        self,
    ):
        probability = evaluate_formula("a; (b, c)", {"a": 0.5, "b": 0.8, "c": 0.5})

        self.assertAlmostEqual(probability, 0.7)

    def test_when_disjoined_conjunctions_share_an_atom_then_it_is_counted_once(self):
        probability = evaluate_formula(
            "(a, b); (a, c)", {"a": 0.5, "b": 0.8, "c": 0.5}
        )

        self.assertAlmostEqual(probability, 0.45)

    def test_when_two_disjunctions_are_conjoined_then_probability_is_their_product(
        self,
    ):
        probability = evaluate_formula(
            "(a; b), (c; d)", {"a": 0.5, "b": 0.5, "c": 0.2, "d": 0.5}
        )

        self.assertAlmostEqual(probability, 0.45)

    def test_when_disjunct_contains_a_negation_then_it_applies_only_to_that_disjunct(
        self,
    ):
        probability = evaluate_formula(
            "a; (b, \\+c)", {"a": 0.5, "b": 0.8, "c": 0.5}
        )

        self.assertAlmostEqual(probability, 0.7)

    def test_when_shared_atom_appears_at_two_depths_then_it_is_counted_once(self):
        probability = evaluate_formula(
            "(a, b); (a, c, (d; e))",
            {"a": 0.5, "b": 0.8, "c": 0.5, "d": 0.5, "e": 0.5},
        )

        self.assertAlmostEqual(probability, 0.4375)

    def test_when_formula_is_nested_four_levels_then_probability_is_exact(self):
        probability = evaluate_formula(
            "(a, b); (a, c, (d; (e, f)))",
            dict.fromkeys(["a", "b", "c", "d", "e", "f"], 0.5),
        )

        self.assertAlmostEqual(probability, 0.328125)

    def test_when_conjoined_disjunctions_sit_inside_a_disjunction_then_probability_is_exact(
        self,
    ):
        probability = evaluate_formula(
            "((a; b), (c; d)); (e, f)",
            dict.fromkeys(["a", "b", "c", "d", "e", "f"], 0.5),
        )

        self.assertAlmostEqual(probability, 0.671875)

    def test_when_atom_is_required_and_excluded_then_probability_is_zero(self):
        probability = evaluate_formula("a, \\+a", {"a": 0.5})

        self.assertAlmostEqual(probability, 0.0)

    def test_when_required_atom_also_appears_in_excluded_group_then_it_is_counted_once(
        self,
    ):
        probability = evaluate_formula("a, \\+(b, a)", {"a": 0.5, "b": 0.8})

        self.assertAlmostEqual(probability, 0.1)

    def test_when_atom_is_required_in_one_branch_and_excluded_in_another_then_branches_are_exclusive(
        self,
    ):
        probability = evaluate_formula(
            "(a, b); (\\+a, c)", {"a": 0.5, "b": 0.8, "c": 0.5}
        )

        self.assertAlmostEqual(probability, 0.65)


class TestEvaluateFormulaDatasetPatterns(unittest.TestCase):
    """One real formula per logical pattern in the QUEST test queries.

    Each formula is the shortest of its pattern, as translated from the dataset.
    Expected values are exact truth-table results, computed without ProbLog.
    """

    ATOM_PROBABILITIES = {
        "a0": 0.9,
        "a1": 0.8,
        "a2": 0.7,
        "a3": 0.6,
        "a4": 0.5,
        "a5": 0.4,
        "a6": 0.3,
        "a7": 0.2,
        "a8": 0.1,
    }

    def test_when_formula_is_a_conjunction_of_atoms_then_probability_matches_truth_table(
        self,
    ):
        # Query 213, one of 714 with this pattern.
        probability = evaluate_formula(
            "(a0) , (a1)",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.72, places=12)

    def test_when_formula_is_a_conjunction_excluding_one_atom_then_probability_matches_truth_table(
        self,
    ):
        # Query 971, one of 375 with this pattern.
        probability = evaluate_formula(
            "(\\+(a1)) , (a0)",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.18, places=12)

    def test_when_formula_is_a_disjunction_of_two_conjunctions_then_probability_matches_truth_table(
        self,
    ):
        # Query 905, one of 179 with this pattern.
        probability = evaluate_formula(
            "((a0) , (a1)) ; (a0) , (a2)",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.846, places=12)

    def test_when_formula_is_a_conjunction_containing_one_disjunction_then_probability_matches_truth_table(
        self,
    ):
        # Query 12, one of 157 with this pattern.
        probability = evaluate_formula(
            "((a0) , ((a1) ; (a2)))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.846, places=12)

    def test_when_formula_is_a_disjunction_of_three_conjunctions_then_probability_matches_truth_table(
        self,
    ):
        # Query 1702, one of 117 with this pattern.
        probability = evaluate_formula(
            "((a0) , (a1) ; (a0) , (a2) ; (a0) , (a3))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.8784, places=12)

    def test_when_formula_is_a_disjunction_of_atoms_then_probability_matches_truth_table(
        self,
    ):
        # Query 913, one of 35 with this pattern.
        probability = evaluate_formula(
            "(a0) ; (a1)",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.98, places=12)

    def test_when_formula_is_a_conjunction_excluding_a_conjunction_then_probability_matches_truth_table(
        self,
    ):
        # Query 83, one of 26 with this pattern.
        probability = evaluate_formula(
            "((a0) , (a1) , (\\+((a2) , (a3))))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.4176, places=12)

    def test_when_formula_is_a_disjunction_of_atoms_and_one_conjunction_then_probability_matches_truth_table(
        self,
    ):
        # Query 908, one of 22 with this pattern.
        probability = evaluate_formula(
            "((a0) ; (a1 , a2))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.956, places=12)

    def test_when_formula_is_a_conjunction_excluding_an_atom_and_containing_a_disjunction_then_probability_matches_truth_table(
        self,
    ):
        # Query 586, one of 18 with this pattern.
        probability = evaluate_formula(
            "((a0) , ((a1) ; (a2)) , (\\+(a3)))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.3384, places=12)

    def test_when_formula_is_a_disjunction_of_a_conjunction_and_a_conjunction_containing_a_disjunction_then_probability_matches_truth_table(
        self,
    ):
        # Query 1296, one of 13 with this pattern.
        probability = evaluate_formula(
            "(((a0) ; (a1)) , (a2)) ; ((a1) , (a3))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.83, places=12)

    def test_when_formula_is_a_disjunction_of_atoms_and_two_conjunctions_then_probability_matches_truth_table(
        self,
    ):
        # Query 885, one of 13 with this pattern.
        probability = evaluate_formula(
            "(((a0) , (a1)) ; ((a2) , (a3)) ; (a4))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.9188, places=12)

    def test_when_formula_is_a_conjunction_excluding_a_disjunction_then_probability_matches_truth_table(
        self,
    ):
        # Query 549, one of 9 with this pattern.
        probability = evaluate_formula(
            "((a0) , (a1) , (\\+((a2) ; (a3))))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.0864, places=12)

    def test_when_formula_is_a_conjunction_containing_a_disjunction_with_a_conjunction_then_probability_matches_truth_table(
        self,
    ):
        # Query 1355, one of 8 with this pattern.
        probability = evaluate_formula(
            "((a0) , ((a1) ; ((a2) , (a3))))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.7956, places=12)

    def test_when_formula_is_a_single_atom_then_probability_matches_truth_table(
        self,
    ):
        # Query 1089, one of 7 with this pattern.
        probability = evaluate_formula(
            "(a0)",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.9, places=12)

    def test_when_formula_is_a_conjunction_excluding_two_atoms_then_probability_matches_truth_table(
        self,
    ):
        # Query 89, one of 4 with this pattern.
        probability = evaluate_formula(
            "((a0) , (a3) , (a4) , (\\+(a1)) , (\\+(a2)))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.0162, places=12)

    def test_when_formula_is_a_disjunction_of_atoms_and_a_conjunction_containing_a_disjunction_then_probability_matches_truth_table(
        self,
    ):
        # Query 936, one of 4 with this pattern.
        probability = evaluate_formula(
            "((a0) ; (a1) , ((a2) ; (a3)))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.9704, places=12)

    def test_when_formula_is_a_disjunction_of_four_conjunctions_then_probability_matches_truth_table(
        self,
    ):
        # Query 858, one of 2 with this pattern.
        probability = evaluate_formula(
            "(((a0) , (a1)) ; ((a0) , (a2)) ; ((a0) , (a3) ; (a0) , (a4)))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.8892, places=12)

    def test_when_formula_is_a_conjunction_containing_a_disjunction_of_two_conjunctions_then_probability_matches_truth_table(
        self,
    ):
        # Query 61, one of 1 with this pattern.
        probability = evaluate_formula(
            "((a0) , (((a2) , (a1)) ; ((a4) , (a3))))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.6228, places=12)

    def test_when_formula_is_a_conjunction_of_atoms_and_two_disjunctions_then_probability_matches_truth_table(
        self,
    ):
        # Query 328, one of 1 with this pattern.
        probability = evaluate_formula(
            "((a0) , (a1) , ((a2) ; (a3)) , ((a4) ; (a5)))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.44352, places=12)

    def test_when_formula_is_a_disjunction_of_three_conjunctions_where_one_excludes_two_atoms_then_probability_matches_truth_table(
        self,
    ):
        # Query 429, one of 1 with this pattern.
        probability = evaluate_formula(
            "(((a0) , (a1) , (a2) , (a3)) ; ((a0) , (a4) , (a5) , (a6) , (\\+(a7)) , (\\+(a8))) ; ((a0) , (a4) , (a7) , (a8)))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.33419232, places=12)

    def test_when_formula_is_a_disjunction_with_a_conjunction_nested_four_levels_deep_then_probability_matches_truth_table(
        self,
    ):
        # Query 431, one of 1 with this pattern.
        probability = evaluate_formula(
            "(((a0) , (a1) , (((a2) , (a4)) ; (a3))) ; ((a0) , (a5) , (a6)))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.576864, places=12)

    def test_when_formula_is_a_disjunction_of_atoms_an_excluding_conjunction_and_a_conjunction_containing_a_disjunction_then_probability_matches_truth_table(
        self,
    ):
        # Query 840, one of 1 with this pattern.
        probability = evaluate_formula(
            "(((\\+(a2)) , (a3)) ; ((a0) , (a1) , ((a2) ; (a3))) ; (a4))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.842, places=12)

    def test_when_formula_is_a_conjunction_of_two_disjunctions_then_probability_matches_truth_table(
        self,
    ):
        # Query 1010, one of 1 with this pattern.
        probability = evaluate_formula(
            "(((a0) ; (a1)) , ((a2) ; (a3)))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.8624, places=12)

    def test_when_formula_is_a_disjunction_of_a_conjunction_and_an_excluding_conjunction_then_probability_matches_truth_table(
        self,
    ):
        # Query 1225, one of 1 with this pattern.
        probability = evaluate_formula(
            "((\\+(a1) , (a0)) ; ((a2) , (a3)))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.5244, places=12)

    def test_when_formula_is_a_disjunction_of_atoms_and_an_excluding_conjunction_then_probability_matches_truth_table(
        self,
    ):
        # Query 1612, one of 1 with this pattern.
        probability = evaluate_formula(
            "((a0) ; (a1) , (a2) , (\\+(a3)))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.9224, places=12)

    def test_when_formula_is_a_conjunction_of_two_exclusions_and_a_disjunction_then_probability_matches_truth_table(
        self,
    ):
        # Query 1613, one of 1 with this pattern.
        probability = evaluate_formula(
            "(((a0) ; (a1)) , (\\+(a2)) , (\\+(a3)))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.1176, places=12)

    def test_when_formula_is_a_conjunction_of_an_excluded_conjunction_and_a_disjunction_then_probability_matches_truth_table(
        self,
    ):
        # Query 1626, one of 1 with this pattern.
        probability = evaluate_formula(
            "(((a0) ; (a1)) , (\\+((a2) , (a3))))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.5684, places=12)

    def test_when_formula_is_a_disjunction_of_three_conjunctions_where_one_contains_a_negated_disjunct_then_probability_matches_truth_table(
        self,
    ):
        # Query 1684, one of 1 with this pattern.
        probability = evaluate_formula(
            "(((\\+(a3) ; (a2)) , (a0) , (a4)) ; ((a3) , (a0) , (a4)) ; ((a3) , (a1) , (a5)))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.5556, places=12)

    def test_when_formula_is_a_disjunction_of_a_conjunction_of_disjunctions_and_a_conjunction_then_probability_matches_truth_table(
        self,
    ):
        # Query 1686, one of 1 with this pattern.
        probability = evaluate_formula(
            "(((a0) ; (a1)) , ((a2) ; (a3))) ; ((a4) , (a5))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.88992, places=12)

    def test_when_formula_is_a_conjunction_of_a_disjunction_and_a_disjunction_with_a_conjunction_then_probability_matches_truth_table(
        self,
    ):
        # Query 1708, one of 1 with this pattern.
        probability = evaluate_formula(
            "(((a0) ; (a1)) , ((a2) ; (a3) , (a4)))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.7742, places=12)

    def test_when_formula_is_a_disjunction_of_three_conjunctions_where_one_excludes_an_atom_then_probability_matches_truth_table(
        self,
    ):
        # Query 1718, one of 1 with this pattern.
        probability = evaluate_formula(
            "(((\\+(a1)) , (a2) , (a5)) ; ((a1) , (a3) , (a4) , (a0))) ; ((a1) , (a4) , (a0))",
            self.ATOM_PROBABILITIES,
        )

        self.assertAlmostEqual(probability, 0.416, places=12)


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
