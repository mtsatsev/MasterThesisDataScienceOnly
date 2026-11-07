import json
import re
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from llm_bayesian_reasoning.logic.formula import evaluate_formula
from llm_bayesian_reasoning.records.prepare import RecordRepair, prepare_record
from llm_bayesian_reasoning.records.query_record import QueryRecord
from llm_bayesian_reasoning.records.record_repairs import load_record_repairs

DATA_DIRECTORY = Path(__file__).parents[2] / "llm_bayesian_reasoning" / "data"
RAW_RECORDS_PATH = DATA_DIRECTORY / "preprocessed_data/parsed_test_with_negs.jsonl"
QUEST_PATH = DATA_DIRECTORY / "quest_data/test.jsonl"


def readable_formula(record: QueryRecord) -> str:
    """Return the formula with each atom name replaced by its text."""
    texts = {atom.name: atom.text.removeprefix("{X} ") for atom in record.atoms}
    return re.sub(r"\ba\d+\b", lambda name: texts[name.group()], record.formula)


class TestLoadRecordRepairs(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "repairs.json"

    def test_when_file_is_valid_then_repairs_are_keyed_by_integer_query_id(self):
        self.path.write_text(
            json.dumps({"7": {"logical_query": "({X} is a book)"}}), encoding="utf-8"
        )

        repairs = load_record_repairs(self.path)

        self.assertEqual(repairs, {7: RecordRepair(logical_query="({X} is a book)")})

    def test_when_repair_has_non_ascii_text_then_it_is_read_unchanged(self):
        query = "({X} is found in Annobón)"
        self.path.write_bytes(
            json.dumps({"7": {"logical_query": query}}, ensure_ascii=False).encode(
                "utf-8"
            )
        )

        self.assertEqual(load_record_repairs(self.path)[7].logical_query, query)

    def test_when_a_repair_is_invalid_then_error_names_its_query_id(self):
        self.path.write_text(json.dumps({"341": {}}), encoding="utf-8")

        with self.assertRaisesRegex(ValidationError, "341"):
            load_record_repairs(self.path)

    def test_when_a_query_id_is_not_an_integer_then_validation_error_is_raised(self):
        self.path.write_text(
            json.dumps({"abc": {"logical_query": "({X} is a book)"}}), encoding="utf-8"
        )

        with self.assertRaises(ValidationError):
            load_record_repairs(self.path)

    def test_when_file_does_not_exist_then_file_not_found_error_is_raised(self):
        with self.assertRaises(FileNotFoundError):
            load_record_repairs(self.path)

    def test_when_no_path_is_given_then_the_reviewed_table_is_loaded(self):
        self.assertEqual(len(load_record_repairs()), 180)


@unittest.skipUnless(
    RAW_RECORDS_PATH.exists() and QUEST_PATH.exists(), "The QUEST data is not present"
)
class TestFormulaRepairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repairs = load_record_repairs()
        with RAW_RECORDS_PATH.open(encoding="utf-8") as file:
            cls.raw_records = {
                raw["id"]: raw for raw in (json.loads(line) for line in file)
            }
        with QUEST_PATH.open(encoding="utf-8") as file:
            relevant_entities = {
                quest["original_query"]: quest["docs"]
                for quest in (json.loads(line) for line in file)
            }

        cls.records = {}
        cls.failures = {}
        for query_id, raw in cls.raw_records.items():
            try:
                cls.records[query_id] = prepare_record(
                    raw, relevant_entities, cls.repairs
                )
            except ValueError as error:
                cls.failures[query_id] = str(error)

    def assert_formula(self, query_id: int, expected: str) -> None:
        self.assertEqual(readable_formula(self.records[query_id]), expected)

    def negated_text(self, query_id: int, atom_text: str) -> str:
        atoms = {atom.text: atom for atom in self.records[query_id].atoms}
        return atoms[atom_text].negated_text

    def test_when_table_is_loaded_then_every_repair_belongs_to_a_dataset_query(self):
        self.assertEqual(set(self.repairs) - set(self.raw_records), set())

    def test_when_table_is_applied_then_every_query_builds(self):
        self.assertEqual(self.failures, {})

    def test_when_table_is_applied_then_every_formula_can_be_satisfied(self):
        impossible = [
            query_id
            for query_id, record in self.records.items()
            if evaluate_formula(
                record.formula, {atom.name: 0.5 for atom in record.atoms}
            )
            == 0.0
        ]

        self.assertEqual(impossible, [])

    def test_when_table_is_applied_then_no_negated_text_equals_its_atom(self):
        identical = [
            (query_id, atom.text)
            for query_id, record in self.records.items()
            for atom in record.atoms
            if atom.text == atom.negated_text
        ]

        self.assertEqual(identical, [])

    def test_when_table_is_applied_then_every_atom_is_used_by_its_formula(self):
        unused = [
            (query_id, atom.text)
            for query_id, record in self.records.items()
            for atom in record.atoms
            if atom.name not in re.findall(r"\ba\d+\b", record.formula)
        ]

        self.assertEqual(unused, [])

    def test_when_query_excludes_folklore_then_formula_excludes_it(self):
        self.assert_formula(
            108,
            "((is a film) , (is a Hong Kong film) , (is a supernatural film) , "
            "(\\+(is based on folklore)))",
        )

    def test_when_query_excludes_judaism_then_formula_excludes_it(self):
        self.assert_formula(
            565,
            "((is a book) , (is written by Isaac Asimov) , (\\+(is about Judaism)))",
        )

    def test_when_query_excludes_american_prison_films_then_they_stay_one_atom(self):
        self.assert_formula(
            341,
            "((is a film) , (is an American comedy film) , (is a prison film) , "
            "(\\+(is an American prison film)))",
        )

    def test_when_query_409_is_repaired_then_only_drama_film_is_removed(self):
        self.assert_formula(
            409,
            "(((is a film) , (is a Hungarian film) , (is a WWII film)) ; "
            "((is a film) , (is a Biographical drama film)) ; "
            "((is a film) , (is a Romanian film) , (is a historical drama film)))",
        )

    def test_when_query_576_is_repaired_then_it_excludes_american_novels(self):
        self.assert_formula(
            576,
            "((is a novel) , (is a science fiction novel) , (is from 2012) , "
            "(\\+(is an American novel)))",
        )

    def test_when_query_1372_is_repaired_then_its_alternatives_are_explicit(self):
        self.assert_formula(
            1372,
            "((is an animal) , (is from Antarctica) , "
            "((is prehistoric) ; (is from the Paleogene)))",
        )

    def test_when_query_says_or_then_formula_has_alternatives(self):
        self.assert_formula(
            487,
            "((is a novel) , ((is set in the Zhou dynasty) ; (is set in Henan)))",
        )

    def test_when_query_asks_for_an_intersection_then_formula_requires_both(self):
        self.assert_formula(
            1048,
            "((is a flora) , ((is from Yemen) , (is from Egypt)))",
        )

    def test_when_a_query_was_kept_then_the_table_has_no_entry_for_it(self):
        kept = (104, 283, 346, 544, 561, 724, 948, 1233, 1620, 1725)

        for query_id in kept:
            with self.subTest(query_id=query_id):
                self.assertNotIn(query_id, self.repairs)

    def test_when_negated_text_was_identical_then_it_now_has_a_negation(self):
        self.assertEqual(
            self.negated_text(186, "{X} is from 1987"), "{X} is not from 1987"
        )

    def test_when_negated_text_named_an_alternative_then_it_now_has_a_negation(self):
        self.assertEqual(
            self.negated_text(136, "{X} is from 2016"), "{X} is not from 2016"
        )

    def test_when_a_negated_text_was_kept_then_it_is_unchanged(self):
        self.assertEqual(
            self.negated_text(104, "{X} is in a non-English language"),
            "{X} is in an English language",
        )

    def test_when_an_unused_atom_was_dropped_then_the_record_does_not_have_it(self):
        atom_texts = [atom.text for atom in self.records[1712].atoms]

        self.assertNotIn("{X} is a film", atom_texts)
