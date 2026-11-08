import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.retrievers.e5 import E5Retriever

# Each document: (title, text, embedding). The old index layout stores the
# embeddings in embeddings.npy, texts in entities.jsonl and titles in titles.jsonl.
DOCUMENTS = [
    ("Zettel", "Zettel is a book by Wittgenstein.", [0.0, 1.0]),
    ("Invariances", "Invariances is a book by Nozick.", [1.0, 0.0]),
    ("Socratic Puzzles", "Socratic Puzzles is a book by Nozick.", [3.0, 3.0]),
    ("Anarchy", "Anarchy, State, and Utopia is a book by Nozick.", [0.2, 0.1]),
]
QUERY_VECTORS = {
    "query: Nozick books": [2.0, 0.0],
    "query: tied": [1.0, 1.0],
}


class FakeQueryEncoder:
    """Return stored vectors for known texts and remember every call."""

    def __init__(self):
        self.calls = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return np.array([QUERY_VECTORS[text] for text in texts])


def write_index(directory: Path, documents=DOCUMENTS) -> None:
    np.save(directory / "embeddings.npy", np.array([doc[2] for doc in documents]))
    (directory / "entities.jsonl").write_text(
        "".join(
            json.dumps({"entity": doc[1]}, ensure_ascii=False) + "\n"
            for doc in documents
        ),
        encoding="utf-8",
    )
    (directory / "titles.jsonl").write_text(
        "".join(
            json.dumps({"title": doc[0]}, ensure_ascii=False) + "\n"
            for doc in documents
        ),
        encoding="utf-8",
    )


def cosine(first: list[float], second: list[float]) -> float:
    return float(
        np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    )


class TestE5Retriever(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.index_path = Path(directory.name)
        write_index(self.index_path)
        self.encoder = FakeQueryEncoder()

    def retrieve(self, query: str, top_n: int) -> list[Candidate]:
        return E5Retriever(self.index_path, self.encoder).retrieve(query, top_n)

    def test_when_query_is_retrieved_then_documents_are_ordered_by_cosine(self):
        candidates = self.retrieve("Nozick books", 4)

        self.assertEqual(
            [candidate.entity for candidate in candidates],
            ["Invariances", "Anarchy", "Socratic Puzzles", "Zettel"],
        )

    def test_when_a_document_vector_is_long_then_its_length_does_not_raise_its_rank(
        self,
    ):
        candidates = self.retrieve("Nozick books", 2)

        self.assertNotIn("Socratic Puzzles", [c.entity for c in candidates])

    def test_when_query_is_retrieved_then_candidates_have_title_text_and_cosine(self):
        candidate = self.retrieve("Nozick books", 1)[0]

        self.assertEqual(
            (candidate.entity, candidate.text),
            ("Invariances", "Invariances is a book by Nozick."),
        )
        self.assertAlmostEqual(candidate.retrieval_score, 1.0)

    def test_when_query_is_retrieved_then_every_score_is_the_cosine(self):
        candidates = self.retrieve("Nozick books", 4)
        vectors = {doc[0]: doc[2] for doc in DOCUMENTS}

        for candidate in candidates:
            with self.subTest(entity=candidate.entity):
                self.assertAlmostEqual(
                    candidate.retrieval_score,
                    cosine(vectors[candidate.entity], [2.0, 0.0]),
                )

    def test_when_top_n_is_smaller_than_the_index_then_only_top_n_are_returned(self):
        self.assertEqual(len(self.retrieve("Nozick books", 2)), 2)

    def test_when_top_n_exceeds_the_index_then_every_document_is_returned(self):
        self.assertEqual(len(self.retrieve("Nozick books", 10)), len(DOCUMENTS))

    def test_when_scores_tie_then_index_order_is_kept(self):
        documents = [
            ("Zettel", "Text one.", [1.0, 0.0]),
            ("Invariances", "Text two.", [0.0, 1.0]),
            ("Anarchy", "Text three.", [2.0, 0.0]),
        ]
        write_index(self.index_path, documents)

        candidates = self.retrieve("tied", 3)

        self.assertEqual(
            [candidate.entity for candidate in candidates],
            ["Zettel", "Invariances", "Anarchy"],
        )

    def test_when_query_is_retrieved_then_encoder_gets_it_once_with_query_prefix(self):
        self.retrieve("Nozick books", 2)

        self.assertEqual(self.encoder.calls, [["query: Nozick books"]])

    def test_when_index_files_are_removed_after_loading_then_retrieval_still_works(
        self,
    ):
        retriever = E5Retriever(self.index_path, self.encoder)
        for path in self.index_path.iterdir():
            path.unlink()

        self.assertEqual(len(retriever.retrieve("Nozick books", 2)), 2)

    def test_when_top_n_is_below_one_then_error_names_it(self):
        for top_n in (0, -1):
            with self.subTest(top_n=top_n):
                with self.assertRaisesRegex(ValueError, f"top_n.*{top_n}"):
                    self.retrieve("Nozick books", top_n)

    def test_when_index_files_have_different_lengths_then_error_names_the_files(self):
        for name in ("entities.jsonl", "titles.jsonl"):
            with self.subTest(name=name):
                write_index(self.index_path)
                path = self.index_path / name
                path.write_text(
                    "".join(path.read_text(encoding="utf-8").splitlines(True)[:-1]),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ValueError, name):
                    E5Retriever(self.index_path, self.encoder)

    def test_when_an_index_file_is_missing_then_file_not_found_error_is_raised(self):
        for name in ("embeddings.npy", "entities.jsonl", "titles.jsonl"):
            with self.subTest(name=name):
                write_index(self.index_path)
                (self.index_path / name).unlink()

                with self.assertRaises(FileNotFoundError):
                    E5Retriever(self.index_path, self.encoder)

    def test_when_index_has_non_ascii_text_then_it_is_read_unchanged(self):
        documents = [("Annobón", "Annobón is an island.", [1.0, 0.0])]
        write_index(self.index_path, documents)

        candidate = self.retrieve("Nozick books", 1)[0]

        self.assertEqual(
            (candidate.entity, candidate.text), ("Annobón", "Annobón is an island.")
        )
