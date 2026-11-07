import unittest

from llm_bayesian_reasoning.records.atoms import Atom, build_atoms
from llm_bayesian_reasoning.records.translate_formula import translate_logical_query


class TestBuildAtoms(unittest.TestCase):
    def test_when_one_atom_is_given_then_it_has_the_first_name_and_both_texts(self):
        atoms = build_atoms(["{X} is a film"], ["{X} is not a film"])

        self.assertEqual(
            atoms,
            [Atom(name="a0", text="{X} is a film", negated_text="{X} is not a film")],
        )

    def test_when_several_atoms_are_given_then_names_are_numbered_in_input_order(self):
        atoms = build_atoms(
            ["{X} is a film", "{X} is a book", "{X} is a play"],
            ["{X} is not a film", "{X} is not a book", "{X} is not a play"],
        )

        self.assertEqual(
            [(atom.name, atom.text) for atom in atoms],
            [
                ("a0", "{X} is a film"),
                ("a1", "{X} is a book"),
                ("a2", "{X} is a play"),
            ],
        )

    def test_when_each_atom_has_a_negated_text_then_it_stays_paired_with_that_atom(
        self,
    ):
        atoms = build_atoms(
            ["{X} is a film", "{X} is a book"],
            ["{X} is not a film", "{X} is not a book"],
        )

        self.assertEqual(
            [(atom.text, atom.negated_text) for atom in atoms],
            [
                ("{X} is a film", "{X} is not a film"),
                ("{X} is a book", "{X} is not a book"),
            ],
        )

    def test_when_atom_text_has_a_lowercase_placeholder_then_it_is_normalised(self):
        atoms = build_atoms(["{x} is a film"], ["{X} is not a film"])

        self.assertEqual(atoms[0].text, "{X} is a film")

    def test_when_negated_text_has_a_lowercase_placeholder_then_it_is_normalised(self):
        atoms = build_atoms(["{X} is a film"], ["{x} is not a film"])

        self.assertEqual(atoms[0].negated_text, "{X} is not a film")

    def test_when_atom_text_has_special_characters_then_they_are_unchanged(self):
        texts = (
            "{X} is set in New York (state)",
            "{X} was shot in LA, California",
            "{X} is a children's film",
            "{X} is a comedy-drama film",
            "{X} is a Karel Jaromír Erben film",
        )

        for text in texts:
            with self.subTest(text=text):
                atoms = build_atoms([text], ["{X} is not one"])

                self.assertEqual(atoms[0].text, text)

    def test_when_atom_is_repeated_then_it_gets_one_name_and_later_names_have_no_gap(
        self,
    ):
        atoms = build_atoms(
            ["{x} is from Asia", "{x} is a reptile", "{x} is from Asia", "{x} is old"],
            [
                "{X} is not from Asia",
                "{X} is not a reptile",
                "{X} is not from Asia",
                "{X} is not old",
            ],
        )

        self.assertEqual(
            [(atom.name, atom.text) for atom in atoms],
            [
                ("a0", "{X} is from Asia"),
                ("a1", "{X} is a reptile"),
                ("a2", "{X} is old"),
            ],
        )

    def test_when_atom_is_repeated_with_different_placeholder_case_then_it_gets_one_name(
        self,
    ):
        atoms = build_atoms(
            ["{x} is from Asia", "{X} is from Asia"],
            ["{X} is not from Asia", "{X} is not from Asia"],
        )

        self.assertEqual(len(atoms), 1)

    def test_when_built_atoms_are_given_to_the_translator_then_their_names_are_used(
        self,
    ):
        atoms = build_atoms(
            ["{x} is from Asia", "{x} is a reptile", "{x} is from Asia"],
            ["{X} is not from Asia", "{X} is not a reptile", "{X} is not from Asia"],
        )

        formula = translate_logical_query(
            "(({X} is from Asia) AND ({X} is a reptile) AND ({X} is from Asia))",
            {atom.text: atom.name for atom in atoms},
        )

        self.assertEqual(formula, "((a0) , (a1) , (a0))")

    def test_when_there_are_more_atoms_than_negated_texts_then_error_reports_both_counts(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "2 atoms but 1 negated"):
            build_atoms(["{X} is a film", "{X} is a book"], ["{X} is not a film"])

    def test_when_there_are_more_negated_texts_than_atoms_then_error_reports_both_counts(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "1 atoms but 2 negated"):
            build_atoms(["{X} is a film"], ["{X} is not a film", "{X} is not a book"])

    def test_when_no_atoms_are_given_then_error_is_raised(self):
        with self.assertRaisesRegex(ValueError, "no atoms"):
            build_atoms([], [])
