import unittest
from typing import Any

from pydantic import ValidationError

from llm_bayesian_reasoning.records.atoms import Atom
from llm_bayesian_reasoning.records.query_record import QueryRecord

FILM = Atom(name="a0", text="{X} is a film", negated_text="{X} is not a film")
BOOK = Atom(name="a1", text="{X} is a book", negated_text="{X} is not a book")


def record_fields(**changes: Any) -> dict[str, Any]:
    return {
        "query_id": 7,
        "query": "Films that are books",
        "atoms": [FILM, BOOK],
        "formula": "((a0) , (a1))",
        "relevant_entities": ["Entity One", "Entity Two"],
        "template": "_ that are also _",
        "domain": "films",
        **changes,
    }


class TestQueryRecord(unittest.TestCase):
    def test_when_all_fields_are_valid_then_record_keeps_their_values(self):
        fields = record_fields()

        record = QueryRecord(**fields)

        self.assertEqual(
            {name: getattr(record, name) for name in fields},
            fields,
        )

    def test_when_record_is_written_to_json_then_reading_it_back_gives_an_equal_record(
        self,
    ):
        record = QueryRecord(**record_fields())

        self.assertEqual(
            QueryRecord.model_validate_json(record.model_dump_json()), record
        )

    def test_when_formula_uses_each_logical_operator_then_record_is_accepted(self):
        for formula in (
            "(a0)",
            "((a0) ; (a1))",
            "((a0) , (\\+(a1)))",
            "\\+((a0) ; (a1))",
        ):
            with self.subTest(formula=formula):
                record = QueryRecord(**record_fields(formula=formula))

                self.assertEqual(record.formula, formula)

    def test_when_query_has_no_relevant_entities_then_error_names_the_field(self):
        with self.assertRaisesRegex(ValidationError, "relevant_entities"):
            QueryRecord(**record_fields(relevant_entities=[]))

    def test_when_a_relevant_entity_is_empty_then_error_names_the_field(self):
        with self.assertRaisesRegex(ValidationError, "relevant_entities"):
            QueryRecord(**record_fields(relevant_entities=["Entity One", ""]))

    def test_when_a_relevant_entity_is_repeated_then_error_names_that_entity(self):
        with self.assertRaisesRegex(ValidationError, "Entity One"):
            QueryRecord(**record_fields(relevant_entities=["Entity One", "Entity One"]))

    def test_when_query_has_no_atoms_then_error_names_the_field(self):
        with self.assertRaisesRegex(ValidationError, "atoms"):
            QueryRecord(**record_fields(atoms=[], formula="(a0)"))

    def test_when_two_atoms_share_a_name_then_error_names_that_atom(self):
        same_name = Atom(name="a0", text="{X} is a book", negated_text="{X} is not one")

        with self.assertRaisesRegex(ValidationError, "name 'a0'"):
            QueryRecord(**record_fields(atoms=[FILM, same_name], formula="(a0)"))

    def test_when_two_atoms_share_a_text_then_error_names_that_text(self):
        same_text = Atom(name="a1", text="{X} is a film", negated_text="{X} is not one")

        with self.assertRaisesRegex(ValidationError, "text '{X} is a film'"):
            QueryRecord(**record_fields(atoms=[FILM, same_text]))

    def test_when_an_atom_is_invalid_then_error_names_the_atom_field(self):
        invalid = {"name": "bad", "text": "{X} is a film", "negated_text": "{X} no"}

        with self.assertRaisesRegex(ValidationError, "atoms.0.name"):
            QueryRecord(**record_fields(atoms=[invalid], formula="(a0)"))

    def test_when_formula_uses_an_unknown_atom_then_error_names_that_atom(self):
        with self.assertRaisesRegex(ValidationError, "unscored atom 'a5'"):
            QueryRecord(**record_fields(formula="((a0) , (a5))"))

    def test_when_formula_is_malformed_then_error_reports_the_formula(self):
        for formula in ("((a0) , (a1)", "(a0) (a1)", "(a0) ,, (a1)"):
            with self.subTest(formula=formula):
                with self.assertRaisesRegex(ValidationError, "Malformed formula"):
                    QueryRecord(**record_fields(formula=formula))

    def test_when_a_text_field_is_empty_then_error_names_the_field(self):
        for field in ("query", "formula", "template", "domain"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValidationError, field):
                    QueryRecord(**record_fields(**{field: ""}))

    def test_when_query_id_is_not_an_integer_then_error_names_the_field(self):
        with self.assertRaisesRegex(ValidationError, "query_id"):
            QueryRecord(**record_fields(query_id="not a number"))

    def test_when_a_field_is_missing_then_error_names_the_field(self):
        for field in record_fields():
            with self.subTest(field=field):
                fields = record_fields()
                del fields[field]

                with self.assertRaisesRegex(ValidationError, field):
                    QueryRecord(**fields)

    def test_when_an_unknown_field_is_given_then_error_names_the_field(self):
        with self.assertRaisesRegex(ValidationError, "metadata"):
            QueryRecord(**record_fields(metadata={}))
