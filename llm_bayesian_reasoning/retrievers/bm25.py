import json
from pathlib import Path
from typing import cast

import bm25s
import numpy as np

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate


def _tokenize_documents(texts: list[str]) -> bm25s.tokenization.Tokenized:
    """
    Lowercase the texts and split them into word ids, without English stopwords.
    
    Args:
        texts (list[str]): The documents to tokenize.
    
    Returns:
        bm25s.tokenization.Tokenized: The tokenized documents.
    """
    tokens = bm25s.tokenize(texts, stopwords="en", return_ids=True, show_progress=False)
    return cast(bm25s.tokenization.Tokenized, tokens)


def _tokenize_query(query: str) -> list[str]:
    """
    Lowercase the query and split it into words, without English stopwords.

    Args:
        query (str): The query text.

    Returns:
        list[str]: The tokenized query.
    """
    tokens = bm25s.tokenize(
        [query], stopwords="en", return_ids=False, show_progress=False
    )
    return cast(list[list[str]], tokens)[0]


class BM25Retriever:
    """Rank indexed documents by their BM25 score for a query."""

    @staticmethod
    def build(titles: list[str], texts: list[str], index_path: Path) -> None:
        """
        Index the texts with BM25 and save the index with the titles and texts.

        Args:
            titles (list[str]):
                The entity name of each document.
            texts (list[str]):
                The text of each document, in the same order.
            index_path (Path):
                The folder to save the index to.
        """
        if len(titles) != len(texts):
            raise ValueError(f"Got {len(titles)} titles but {len(texts)} texts")
        if not titles:
            raise ValueError("There are no documents to index")

        model = bm25s.BM25()
        model.index(_tokenize_documents(texts), show_progress=False)
        model.save(index_path, show_progress=False)
        with (index_path / "documents.jsonl").open("w", encoding="utf-8") as file:
            for title, text in zip(titles, texts):
                document = {"title": title, "text": text}
                file.write(json.dumps(document, ensure_ascii=False) + "\n")

    def __init__(self, index_path: Path):
        """
        Load an index saved by build.

        Args:
            index_path (Path):
                The index folder.
        """
        with (index_path / "documents.jsonl").open(encoding="utf-8") as file:
            documents = [json.loads(line) for line in file]
        self.titles = [document["title"] for document in documents]
        self.texts = [document["text"] for document in documents]
        self.model = bm25s.BM25.load(index_path, show_progress=False)

    def retrieve(self, query: str, top_n: int) -> list[Candidate]:
        """
        Return the top_n documents with the highest BM25 score, best first.

        Args:
            query (str):
                The query text.
            top_n (int):
                The number of documents to return, at least 1.

        Returns:
            list[Candidate]: The documents, with their BM25 score.
        """
        if top_n < 1:
            raise ValueError(f"top_n must be at least 1, got {top_n}")
        tokens = _tokenize_query(query)
        if not tokens:
            raise ValueError(f"Query {query!r} has no searchable words")

        scores = self.model.get_scores(tokens)
        best = np.argsort(-scores, kind="stable")[:top_n]
        return [
            Candidate(
                entity=self.titles[index],
                text=self.texts[index],
                retrieval_score=float(scores[index]),
            )
            for index in best
        ]
