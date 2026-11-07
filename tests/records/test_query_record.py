import tempfile
import unittest
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from llm_bayesian_reasoning.records.atoms import Atom
from llm_bayesian_reasoning.records.query_record import (
    QueryRecord,
    load_query_records,
)

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


class TestLoadQueryRecords(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "queries.jsonl"

    def write_lines(self, *lines: str) -> None:
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def record_line(self, query_id: int, **changes: Any) -> str:
        record = QueryRecord(**record_fields(query_id=query_id, **changes))
        return record.model_dump_json()

    def test_when_file_has_valid_records_then_all_are_loaded_in_file_order(self):
        self.write_lines(self.record_line(3), self.record_line(1), self.record_line(2))

        records = load_query_records(self.path)

        self.assertEqual([record.query_id for record in records], [3, 1, 2])

    def test_when_records_are_written_and_loaded_then_they_are_equal(self):
        written = [
            QueryRecord(**record_fields(query_id=1)),
            QueryRecord(**record_fields(query_id=2, query="Other films")),
        ]
        self.write_lines(*(record.model_dump_json() for record in written))

        self.assertEqual(load_query_records(self.path), written)

    def test_when_last_line_has_no_newline_then_it_is_still_loaded(self):
        self.path.write_text(self.record_line(1), encoding="utf-8")

        records = load_query_records(self.path)

        self.assertEqual([record.query_id for record in records], [1])

    def test_when_record_has_non_ascii_text_then_it_is_read_unchanged(self):
        query = "Karel Jaromír Erben films"
        record = QueryRecord(**record_fields(query_id=1, query=query))
        self.path.write_bytes(
            (record.model_dump_json() + "\n").encode("utf-8"),
        )

        self.assertEqual(load_query_records(self.path)[0].query, query)

    def test_when_a_line_is_not_json_then_error_reports_its_line_number(self):
        self.write_lines(self.record_line(1), "{not json", self.record_line(3))

        with self.assertRaisesRegex(ValueError, "line 2"):
            load_query_records(self.path)

    def test_when_a_line_is_blank_then_error_reports_its_line_number(self):
        self.write_lines(self.record_line(1), "", self.record_line(3))

        with self.assertRaisesRegex(ValueError, "line 2"):
            load_query_records(self.path)

    def test_when_a_record_is_invalid_then_error_reports_line_number_and_field(self):
        invalid = self.record_line(2).replace(
            '"relevant_entities":["Entity One","Entity Two"]',
            '"relevant_entities":[]',
        )
        self.write_lines(self.record_line(1), invalid)

        with self.assertRaisesRegex(ValueError, "(?s)line 2.*relevant_entities"):
            load_query_records(self.path)

    def test_when_a_line_is_invalid_then_error_reports_the_file(self):
        self.write_lines("{not json")

        with self.assertRaisesRegex(ValueError, "queries.jsonl"):
            load_query_records(self.path)

    def test_when_a_line_is_invalid_then_validation_error_is_the_cause(self):
        self.write_lines("{not json")

        with self.assertRaises(ValueError) as caught:
            load_query_records(self.path)

        self.assertIsInstance(caught.exception.__cause__, ValidationError)

    def test_when_query_id_repeats_on_a_later_line_then_error_reports_id_and_line(self):
        self.write_lines(self.record_line(7), self.record_line(8), self.record_line(7))

        with self.assertRaisesRegex(ValueError, "line 3.*query id 7"):
            load_query_records(self.path)

    def test_when_file_has_no_records_then_error_reports_the_file(self):
        self.path.write_text("", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "queries.jsonl.*no records"):
            load_query_records(self.path)

    def test_when_file_does_not_exist_then_file_not_found_error_is_raised(self):
        with self.assertRaises(FileNotFoundError):
            load_query_records(self.path)
