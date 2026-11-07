import re
import unittest

from llm_bayesian_reasoning.logic.formula import evaluate_formula
from llm_bayesian_reasoning.records.translate_formula import translate_logical_query

FILM = "{X} is a film"
BOOK = "{X} is a book"
NAMES = {FILM: "a0", BOOK: "a1"}


class TestTranslateLogicalQuery(unittest.TestCase):
    def test_when_query_is_a_single_atom_then_it_becomes_that_atom_name(self):
        formula = translate_logical_query("({X} is a film)", NAMES)

        self.assertEqual(formula, "(a0)")

    def test_when_atoms_are_joined_by_an_AND_then_they_are_joined_by_a_comma(self):
        formula = translate_logical_query(
            "(({X} is a film) AND ({X} is a book))", NAMES
        )

        self.assertEqual(formula, "((a0) , (a1))")

    def test_when_atoms_are_joined_by_an_OR_then_they_are_joined_by_a_semicolon(self):
        formula = translate_logical_query("(({X} is a film) OR ({X} is a book))", NAMES)

        self.assertEqual(formula, "((a0) ; (a1))")

    def test_when_atom_is_negated_then_NOT_becomes_the_problog_negation(self):
        formula = translate_logical_query(
            "(({X} is a film) AND (NOT({X} is a book)))", NAMES
        )

        self.assertEqual(formula, "((a0) , (\\+(a1)))")

    def test_when_group_is_negated_then_negation_wraps_the_whole_group(self):
        formula = translate_logical_query(
            "(({X} is a film) AND (NOT(({X} is a book) OR ({X} is a film))))", NAMES
        )

        self.assertEqual(formula, "((a0) , (\\+((a1) ; (a0))))")

    def test_when_operators_are_nested_then_grouping_is_preserved(self):
        formula = translate_logical_query(
            "(({X} is an arthropod) AND ({X} is from North America) AND "
            "(({X} is from the Devonian period) OR ({X} is from the Silurian period)))",
            {
                "{X} is an arthropod": "a0",
                "{X} is from North America": "a1",
                "{X} is from the Devonian period": "a2",
                "{X} is from the Silurian period": "a3",
            },
        )

        self.assertEqual(formula, "((a0) , (a1) , ((a2) ; (a3)))")

    def test_when_one_atom_is_a_prefix_of_another_then_each_keeps_its_own_name(self):
        short, long = "{X} is a film", "{X} is a film franchise"
        orders = {
            "short first": {short: "a0", long: "a1"},
            "long first": {long: "a1", short: "a0"},
        }

        for order, names in orders.items():
            with self.subTest(order=order):
                formula = translate_logical_query(
                    "(({X} is a film) AND ({X} is a film franchise))", names
                )

                self.assertEqual(formula, "((a0) , (a1))")

    def test_when_atom_contains_the_word_and_then_it_is_not_treated_as_an_operator(
        self,
    ):
        formula = translate_logical_query(
            "(({X} is in black and white) AND ({X} is a film))",
            {"{X} is in black and white": "a0", FILM: "a1"},
        )

        self.assertEqual(formula, "((a0) , (a1))")

    def test_when_atom_contains_special_characters_then_it_becomes_its_name(self):
        atoms = (
            "{X} is set in New York (state)",
            "{X} is set in New York(state)",
            "{X} is from the Cocos (Keeling) Islands",
            "{X} was shot in LA, California",
            "{X} is a children's film",
            "{X} is from 2012",
            "{X} is a comedy-drama film",
            "{X} is based on an E. B. White novel",
            "{X} is a Karel Jaromír Erben film",
            "{X} is an order",
        )

        for atom in atoms:
            with self.subTest(atom=atom):
                formula = translate_logical_query(
                    f"(({atom}) AND ({FILM}))", {atom: "a0", FILM: "a1"}
                )

                self.assertEqual(formula, "((a0) , (a1))")

    def test_when_closing_bracket_is_missing_then_it_is_added_at_the_end(self):
        formula = translate_logical_query(
            "(({X} is a Pelecaniform) AND ({X} is a Miocene animal)",
            {"{X} is a Pelecaniform": "a0", "{X} is a Miocene animal": "a1"},
        )

        self.assertEqual(formula, "((a0) , (a1))")

    def test_when_query_has_an_unused_atom_then_translation_still_succeeds(self):
        formula = translate_logical_query("({X} is a film)", NAMES)

        self.assertNotIn("a1", formula)

    def test_when_nested_query_is_translated_then_problog_computes_its_probability(
        self,
    ):
        formula = translate_logical_query(
            "(({X} is a film) AND (({X} is a book) OR ({X} is a play)))",
            {FILM: "a0", BOOK: "a1", "{X} is a play": "a2"},
        )

        probability = evaluate_formula(formula, {"a0": 0.5, "a1": 0.8, "a2": 0.5})

        self.assertAlmostEqual(probability, 0.45)

    def test_when_bracket_is_added_then_problog_computes_its_probability(self):
        formula = translate_logical_query("(({X} is a film) AND ({X} is a book)", NAMES)

        probability = evaluate_formula(formula, {"a0": 0.5, "a1": 0.8})

        self.assertAlmostEqual(probability, 0.4)
