import unittest
from typing import Any

from pydantic import ValidationError

from llm_bayesian_reasoning.records.atoms import Atom
from llm_bayesian_reasoning.records.prepare import RecordRepair, prepare_record
from llm_bayesian_reasoning.records.query_record import QueryRecord

ORIGINAL_QUERY = "<mark>Books by Robert Nozick</mark>"
RELEVANT_ENTITIES = {ORIGINAL_QUERY: ["Invariances", "Socratic Puzzles"]}


def raw_record() -> dict[str, Any]:
    return {
        "id": 664,
        "query": "Robert Nozick books.",
        "original_query": ORIGINAL_QUERY,
        "parsed": {
            "atoms": ["{x} is a book", "{x} is written by Robert Nozick"],
            "negated_atoms": [
                "{X} is not a book",
                "{X} is not written by Robert Nozick",
            ],
            "logical query": "(({x} is a book) AND ({x} is written by Robert Nozick))",
        },
        "metadata": {
            "template": "_",
            "domain": "books",
            "relevance_ratings": {"Invariances": ["Definitely relevant"]},
        },
    }


class TestRecordRepair(unittest.TestCase):
    def test_when_only_the_query_text_is_given_then_atoms_stay_unset(self):
        repair = RecordRepair(logical_query="({X} is a book)")

        self.assertEqual(
            (repair.logical_query, repair.atoms, repair.negated_atoms),
            ("({X} is a book)", None, None),
        )

    def test_when_only_atoms_and_negated_atoms_are_given_then_query_text_stays_unset(
        self,
    ):
        repair = RecordRepair(
            atoms=["{X} is a book"], negated_atoms=["{X} is not a book"]
        )

        self.assertEqual(
            (repair.logical_query, repair.atoms, repair.negated_atoms),
            (None, ["{X} is a book"], ["{X} is not a book"]),
        )

    def test_when_atoms_are_given_without_negated_atoms_then_error_names_the_field(
        self,
    ):
        with self.assertRaisesRegex(ValidationError, "negated_atoms"):
            RecordRepair(logical_query="({X} is a book)", atoms=["{X} is a book"])

    def test_when_negated_atoms_are_given_without_atoms_then_error_names_the_field(
        self,
    ):
        with self.assertRaisesRegex(ValidationError, "needs atoms"):
            RecordRepair(
                logical_query="({X} is a book)",
                negated_atoms=["{X} is not a book"],
            )

    def test_when_nothing_is_given_then_error_says_the_repair_changes_nothing(self):
        with self.assertRaisesRegex(ValidationError, "changes nothing"):
            RecordRepair()

    def test_when_an_unknown_field_is_given_then_error_names_the_field(self):
        with self.assertRaisesRegex(ValidationError, "reason"):
            RecordRepair(logical_query="({X} is a book)", reason="typo")


class TestPrepareRecord(unittest.TestCase):
    def test_when_raw_record_is_valid_then_every_field_is_filled(self):
        record = prepare_record(raw_record(), RELEVANT_ENTITIES, {})

        self.assertEqual(
            record,
            QueryRecord(
                query_id=664,
                query="Robert Nozick books.",
                atoms=[
                    Atom(
                        name="a0",
                        text="{X} is a book",
                        negated_text="{X} is not a book",
                    ),
                    Atom(
                        name="a1",
                        text="{X} is written by Robert Nozick",
                        negated_text="{X} is not written by Robert Nozick",
                    ),
                ],
                formula="((a0) , (a1))",
                relevant_entities=["Invariances", "Socratic Puzzles"],
                template="_",
                domain="books",
            ),
        )

    def test_when_relevance_ratings_are_missing_then_labels_still_come_from_docs(self):
        raw = raw_record()
        raw["metadata"]["relevance_ratings"] = None

        record = prepare_record(raw, RELEVANT_ENTITIES, {})

        self.assertEqual(record.relevant_entities, ["Invariances", "Socratic Puzzles"])

    def test_when_atom_is_repeated_then_formula_uses_its_one_name_twice(self):
        raw = raw_record()
        raw["parsed"] = {
            "atoms": ["{x} is a book", "{x} is old", "{x} is a book"],
            "negated_atoms": [
                "{X} is not a book",
                "{X} is not old",
                "{X} is not a book",
            ],
            "logical query": "(({x} is a book) OR (({x} is old) AND ({x} is a book)))",
        }

        record = prepare_record(raw, RELEVANT_ENTITIES, {})

        self.assertEqual(record.formula, "((a0) ; ((a1) , (a0)))")

    def test_when_closing_bracket_is_missing_then_record_has_a_balanced_formula(self):
        raw = raw_record()
        raw["parsed"]["logical query"] = (
            "(({x} is a book) AND ({x} is written by Robert Nozick)"
        )

        record = prepare_record(raw, RELEVANT_ENTITIES, {})

        self.assertEqual(record.formula, "((a0) , (a1))")

    def test_when_repair_replaces_the_query_text_then_formula_comes_from_the_repair(
        self,
    ):
        repair = RecordRepair(
            logical_query="(({X} is a book) OR ({X} is written by Robert Nozick))"
        )

        record = prepare_record(raw_record(), RELEVANT_ENTITIES, {664: repair})

        self.assertEqual(record.formula, "((a0) ; (a1))")

    def test_when_repair_replaces_only_the_query_text_then_original_atoms_are_kept(
        self,
    ):
        repair = RecordRepair(logical_query="({X} is a book)")

        record = prepare_record(raw_record(), RELEVANT_ENTITIES, {664: repair})

        self.assertEqual(
            [atom.text for atom in record.atoms],
            ["{X} is a book", "{X} is written by Robert Nozick"],
        )

    def test_when_repair_query_has_a_lowercase_placeholder_then_it_is_normalised(self):
        repair = RecordRepair(logical_query="({x} is a book)")

        record = prepare_record(raw_record(), RELEVANT_ENTITIES, {664: repair})

        self.assertEqual(record.formula, "(a0)")

    def test_when_repair_replaces_the_atoms_then_record_uses_only_the_repaired_atoms(
        self,
    ):
        repair = RecordRepair(
            logical_query="(({X} is a tome) AND (NOT({X} is short)))",
            atoms=["{x} is a tome", "{x} is short"],
            negated_atoms=["{X} is not a tome", "{X} is not short"],
        )

        record = prepare_record(raw_record(), RELEVANT_ENTITIES, {664: repair})

        self.assertEqual(
            [(atom.text, atom.negated_text) for atom in record.atoms],
            [
                ("{X} is a tome", "{X} is not a tome"),
                ("{X} is short", "{X} is not short"),
            ],
        )

    def test_when_repair_replaces_only_the_atoms_then_formula_comes_from_the_raw_query(
        self,
    ):
        repair = RecordRepair(
            atoms=["{x} is a book", "{x} is written by Robert Nozick"],
            negated_atoms=["{X} is no book", "{X} is not by Robert Nozick"],
        )

        record = prepare_record(raw_record(), RELEVANT_ENTITIES, {664: repair})

        self.assertEqual(
            (record.formula, record.atoms[0].negated_text),
            ("((a0) , (a1))", "{X} is no book"),
        )

    def test_when_repair_belongs_to_another_query_then_record_is_unchanged(self):
        repair = RecordRepair(logical_query="({X} is a book)")

        record = prepare_record(raw_record(), RELEVANT_ENTITIES, {999: repair})

        self.assertEqual(record.formula, "((a0) , (a1))")

    def test_when_repair_fixes_text_that_is_not_an_atom_then_record_is_valid(self):
        raw = raw_record()
        raw["parsed"]["logical query"] = "(({x} is a book) AND ({x} is from 2012))"
        repair = RecordRepair(logical_query="({X} is a book)")

        record = prepare_record(raw, RELEVANT_ENTITIES, {664: repair})

        self.assertEqual(record.formula, "(a0)")

    def test_when_query_uses_text_that_is_not_an_atom_then_error_names_query_and_text(
        self,
    ):
        raw = raw_record()
        raw["parsed"]["logical query"] = "(({x} is a book) AND ({x} is from 2012))"

        with self.assertRaisesRegex(ValueError, r"Query 664.*\{X\} is from 2012"):
            prepare_record(raw, RELEVANT_ENTITIES, {})

    def test_when_query_has_no_quest_entry_then_error_names_the_query(self):
        with self.assertRaisesRegex(ValueError, "Query 664.*no QUEST entry"):
            prepare_record(raw_record(), {}, {})

    def test_when_quest_entry_has_no_relevant_entities_then_error_names_the_query(self):
        with self.assertRaisesRegex(ValueError, "(?s)Query 664.*relevant_entities"):
            prepare_record(raw_record(), {ORIGINAL_QUERY: []}, {})

    def test_when_a_raw_field_is_missing_then_error_names_the_query_and_field(self):
        locations = {
            "query": (),
            "original_query": (),
            "parsed": (),
            "metadata": (),
            "atoms": ("parsed",),
            "negated_atoms": ("parsed",),
            "logical query": ("parsed",),
            "template": ("metadata",),
            "domain": ("metadata",),
        }

        for field, parents in locations.items():
            with self.subTest(field=field):
                raw = raw_record()
                container = raw
                for parent in parents:
                    container = container[parent]
                del container[field]

                with self.assertRaisesRegex(ValueError, f"Query 664.*{field}"):
                    prepare_record(raw, RELEVANT_ENTITIES, {})

    def test_when_preparation_fails_then_original_error_is_the_cause(self):
        with self.assertRaises(ValueError) as caught:
            prepare_record(raw_record(), {}, {})

        self.assertIsInstance(caught.exception.__cause__, ValueError)
