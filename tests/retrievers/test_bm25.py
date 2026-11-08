import tempfile
import unittest
from pathlib import Path

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.retrievers.bm25 import BM25Retriever

TITLES = ["Zettel", "Invariances", "Socratic Puzzles", "Anarchy", "Casablanca"]
TEXTS = [
    "Zettel is a book by Ludwig Wittgenstein about philosophy.",
    "Invariances is a book by Robert Nozick about objectivity.",
    "Socratic Puzzles is a collection of essays by Robert Nozick.",
    "Anarchy, State, and Utopia is a book by Robert Nozick about politics.",
    "Casablanca is a film.",
]


def entities(candidates: list[Candidate]) -> list[str]:
    return [candidate.entity for candidate in candidates]


class TestBM25Retriever(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.index_path = Path(directory.name) / "bm25"
        BM25Retriever.build(TITLES, TEXTS, self.index_path)

    def retrieve(self, query: str, top_n: int) -> list[Candidate]:
        return BM25Retriever(self.index_path).retrieve(query, top_n)

    def test_when_one_document_has_the_query_word_then_it_comes_first(self):
        self.assertEqual(entities(self.retrieve("Wittgenstein", 1)), ["Zettel"])

    def test_when_query_is_retrieved_then_candidates_have_title_text_and_score(self):
        candidate = self.retrieve("Wittgenstein", 1)[0]

        self.assertEqual((candidate.entity, candidate.text), ("Zettel", TEXTS[0]))
        self.assertGreater(candidate.retrieval_score, 0.0)

    def test_when_query_is_retrieved_then_scores_never_increase(self):
        scores = [c.retrieval_score for c in self.retrieve("Robert Nozick book", 5)]

        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_when_query_differs_only_in_case_then_results_are_the_same(self):
        self.assertEqual(
            self.retrieve("WITTGENSTEIN Book", 5), self.retrieve("wittgenstein book", 5)
        )

    def test_when_top_n_is_smaller_than_the_index_then_only_top_n_are_returned(self):
        self.assertEqual(len(self.retrieve("Nozick", 2)), 2)

    def test_when_top_n_exceeds_the_index_then_every_document_is_returned(self):
        self.assertEqual(len(self.retrieve("Nozick", 10)), len(TITLES))

    def test_when_scores_tie_then_index_order_is_kept(self):
        candidates = self.retrieve("Wittgenstein", 5)

        self.assertEqual(
            entities(candidates),
            ["Zettel", "Invariances", "Socratic Puzzles", "Anarchy", "Casablanca"],
        )
        self.assertEqual(
            [c.retrieval_score for c in candidates[1:]], [0.0, 0.0, 0.0, 0.0]
        )

    def test_when_no_query_word_is_indexed_then_every_score_is_zero(self):
        candidates = self.retrieve("zeppelin", 3)

        self.assertEqual(
            [(c.entity, c.retrieval_score) for c in candidates],
            [("Zettel", 0.0), ("Invariances", 0.0), ("Socratic Puzzles", 0.0)],
        )

    def test_when_query_has_only_stopwords_then_error_names_the_query(self):
        with self.assertRaisesRegex(ValueError, "'is a the'"):
            self.retrieve("is a the", 3)

    def test_when_top_n_is_below_one_then_error_names_it(self):
        for top_n in (0, -1):
            with self.subTest(top_n=top_n):
                with self.assertRaisesRegex(ValueError, f"top_n.*{top_n}"):
                    self.retrieve("Nozick", top_n)

    def test_when_index_files_are_removed_after_loading_then_retrieval_still_works(
        self,
    ):
        retriever = BM25Retriever(self.index_path)
        for path in self.index_path.iterdir():
            path.unlink()

        self.assertEqual(entities(retriever.retrieve("Wittgenstein", 1)), ["Zettel"])

    def test_when_index_does_not_exist_then_file_not_found_error_is_raised(self):
        with self.assertRaises(FileNotFoundError):
            BM25Retriever(self.index_path.with_name("missing"))

    def test_when_titles_and_texts_differ_in_length_then_error_names_both_counts(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "5 titles.*4 texts"):
            BM25Retriever.build(TITLES, TEXTS[:4], self.index_path)

    def test_when_no_document_is_given_then_error_is_raised(self):
        with self.assertRaisesRegex(ValueError, "no documents"):
            BM25Retriever.build([], [], self.index_path)

    def test_when_index_has_non_ascii_text_then_it_is_read_unchanged(self):
        BM25Retriever.build(["Annobón"], ["Annobón is an island."], self.index_path)

        candidate = self.retrieve("Annobón", 1)[0]

        self.assertEqual(
            (candidate.entity, candidate.text), ("Annobón", "Annobón is an island.")
        )
        self.assertGreater(candidate.retrieval_score, 0.0)
