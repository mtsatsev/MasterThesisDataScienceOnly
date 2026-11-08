import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from llm_bayesian_reasoning.records.atoms import Atom
from llm_bayesian_reasoning.records.prepare import (
    RecordRepair,
    prepare_dataset,
    prepare_record,
)
from llm_bayesian_reasoning.records.query_record import (
    QueryRecord,
    load_query_records,
)
from llm_bayesian_reasoning.records.record_repairs import load_record_repairs

DATA_DIRECTORY = Path(__file__).parents[2] / "llm_bayesian_reasoning" / "data"
RAW_RECORDS_PATH = DATA_DIRECTORY / "preprocessed_data/parsed_test_with_negs.jsonl"
QUEST_PATH = DATA_DIRECTORY / "quest_data/test.jsonl"

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


class TestPrepareDataset(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.raw_path = root / "raw.jsonl"
        self.quest_path = root / "quest.jsonl"
        self.output_path = root / "queries.jsonl"

    def raw_line(self, query_id: int, **changes: Any) -> str:
        return json.dumps(
            {
                **raw_record(),
                "id": query_id,
                "original_query": f"Query {query_id}",
                **changes,
            },
            ensure_ascii=False,
        )

    def quest_line(self, query_id: int, *docs: str) -> str:
        return json.dumps(
            {
                "original_query": f"Query {query_id}",
                "docs": list(docs or [f"Entity of {query_id}"]),
                "scores": None,
            },
            ensure_ascii=False,
        )

    def write_files(self, raw_lines: list[str], quest_lines: list[str]) -> None:
        self.raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
        self.quest_path.write_text("\n".join(quest_lines) + "\n", encoding="utf-8")

    def prepare(self, repairs: dict[int, RecordRepair] | None = None) -> None:
        prepare_dataset(self.raw_path, self.quest_path, self.output_path, repairs or {})

    def test_when_raw_file_has_records_then_output_has_one_per_input_in_order(self):
        self.write_files(
            [self.raw_line(3), self.raw_line(1), self.raw_line(2)],
            [self.quest_line(1), self.quest_line(2), self.quest_line(3)],
        )

        self.prepare()

        self.assertEqual(
            [record.query_id for record in load_query_records(self.output_path)],
            [3, 1, 2],
        )

    def test_when_quest_file_has_docs_then_each_record_gets_its_own_labels(self):
        self.write_files(
            [self.raw_line(1), self.raw_line(2)],
            [self.quest_line(2, "Second"), self.quest_line(1, "First", "Other")],
        )

        self.prepare()

        self.assertEqual(
            [
                record.relevant_entities
                for record in load_query_records(self.output_path)
            ],
            [["First", "Other"], ["Second"]],
        )

    def test_when_a_repair_is_given_then_only_its_query_is_repaired(self):
        self.write_files(
            [self.raw_line(1), self.raw_line(2)],
            [self.quest_line(1), self.quest_line(2)],
        )

        self.prepare({2: RecordRepair(logical_query="({X} is a book)")})

        self.assertEqual(
            [record.formula for record in load_query_records(self.output_path)],
            ["((a0) , (a1))", "(a0)"],
        )

    def test_when_record_has_non_ascii_text_then_output_keeps_it_unchanged(self):
        query = "Karel Jaromír Erben films"
        self.write_files(
            [self.raw_line(1, query=query)], [self.quest_line(1, "Kytice")]
        )

        self.prepare()

        self.assertEqual(load_query_records(self.output_path)[0].query, query)
        self.assertIn(query, self.output_path.read_text(encoding="utf-8"))

    def test_when_output_already_exists_then_it_is_replaced(self):
        self.output_path.write_text("old content\n", encoding="utf-8")
        self.write_files([self.raw_line(1)], [self.quest_line(1)])

        self.prepare()

        self.assertEqual(len(load_query_records(self.output_path)), 1)

    def test_when_a_record_is_invalid_then_error_names_the_query(self):
        self.write_files([self.raw_line(1), self.raw_line(2)], [self.quest_line(1)])

        with self.assertRaisesRegex(ValueError, "Query 2.*no QUEST entry"):
            self.prepare()

    def test_when_a_raw_line_is_not_json_then_error_names_the_file_and_line(self):
        self.write_files([self.raw_line(1), "{not json"], [self.quest_line(1)])

        with self.assertRaisesRegex(ValueError, "raw.jsonl.*line 2"):
            self.prepare()

    def test_when_a_quest_line_is_not_json_then_error_names_the_file_and_line(self):
        self.write_files([self.raw_line(1)], [self.quest_line(1), "{not json"])

        with self.assertRaisesRegex(ValueError, "quest.jsonl.*line 2"):
            self.prepare()

    def test_when_query_id_repeats_then_error_names_the_id(self):
        self.write_files([self.raw_line(7), self.raw_line(7)], [self.quest_line(7)])

        with self.assertRaisesRegex(ValueError, "query id 7"):
            self.prepare()

    def test_when_a_repair_names_an_unknown_query_then_error_names_the_id(self):
        self.write_files([self.raw_line(1)], [self.quest_line(1)])

        with self.assertRaisesRegex(ValueError, "repair.*query id 99"):
            self.prepare({99: RecordRepair(logical_query="({X} is a book)")})

    def test_when_raw_file_has_no_records_then_error_names_the_file(self):
        self.raw_path.write_text("", encoding="utf-8")
        self.quest_path.write_text(self.quest_line(1) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "raw.jsonl.*no records"):
            self.prepare()

    def test_when_preparation_fails_then_no_output_file_is_created(self):
        failures = {
            "invalid record": (
                [self.raw_line(1), self.raw_line(2)],
                [self.quest_line(1)],
            ),
            "repeated id": ([self.raw_line(7), self.raw_line(7)], [self.quest_line(7)]),
            "bad raw line": ([self.raw_line(1), "{not json"], [self.quest_line(1)]),
        }

        for failure, (raw_lines, quest_lines) in failures.items():
            with self.subTest(failure=failure):
                self.write_files(raw_lines, quest_lines)

                with self.assertRaises(ValueError):
                    self.prepare()

                self.assertFalse(self.output_path.exists())

    def test_when_preparation_fails_then_existing_output_is_left_unchanged(self):
        self.output_path.write_text("previous run\n", encoding="utf-8")
        self.write_files([self.raw_line(1), self.raw_line(2)], [self.quest_line(1)])

        with self.assertRaises(ValueError):
            self.prepare()

        self.assertEqual(self.output_path.read_text(encoding="utf-8"), "previous run\n")

    def test_when_raw_file_does_not_exist_then_file_not_found_error_is_raised(self):
        self.quest_path.write_text(self.quest_line(1) + "\n", encoding="utf-8")

        with self.assertRaises(FileNotFoundError):
            self.prepare()

    def test_when_quest_file_does_not_exist_then_file_not_found_error_is_raised(self):
        self.raw_path.write_text(self.raw_line(1) + "\n", encoding="utf-8")

        with self.assertRaises(FileNotFoundError):
            self.prepare()


@unittest.skipUnless(
    RAW_RECORDS_PATH.exists() and QUEST_PATH.exists(), "The QUEST data is not present"
)
class TestRealDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        output_path = Path(directory.name) / "records.jsonl"
        cls.repairs = load_record_repairs()
        prepare_dataset(RAW_RECORDS_PATH, QUEST_PATH, output_path, cls.repairs)
        cls.records = load_query_records(output_path)

        with RAW_RECORDS_PATH.open(encoding="utf-8") as file:
            cls.raw_records = [json.loads(line) for line in file]
        with QUEST_PATH.open(encoding="utf-8") as file:
            cls.quest_records = [json.loads(line) for line in file]

    def test_when_real_data_is_prepared_then_every_query_is_written_in_order(self):
        self.assertEqual(
            [record.query_id for record in self.records],
            [raw["id"] for raw in self.raw_records],
        )

    def test_when_real_data_is_prepared_then_labels_are_the_quest_docs(self):
        self.assertEqual(
            {record.query: record.relevant_entities for record in self.records},
            {quest["query"]: quest["docs"] for quest in self.quest_records},
        )

    def test_when_real_data_is_prepared_then_every_domain_keeps_its_queries(self):
        self.assertEqual(
            Counter(record.domain for record in self.records),
            {"films": 459, "books": 441, "plants": 414, "animals": 413},
        )

    def test_when_written_file_is_loaded_then_it_equals_the_prepared_records(self):
        relevant_entities = {
            quest["original_query"]: quest["docs"] for quest in self.quest_records
        }

        self.assertEqual(
            self.records,
            [
                prepare_record(raw, relevant_entities, self.repairs)
                for raw in self.raw_records
            ],
        )
