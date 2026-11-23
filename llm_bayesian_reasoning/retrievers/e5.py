from pathlib import Path
from typing import Protocol

import numpy as np

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.retrievers.index_documents import (
    read_index_documents,
)
from llm_bayesian_reasoning.retrievers.top_candidates import (
    check_top_n,
    top_candidates,
)


class QueryEncoder(Protocol):
    """Anything that turns texts into embedding vectors."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one embedding row per text."""
        ...


class E5Retriever:
    """Rank indexed documents by cosine similarity to an encoded query."""

    def __init__(self, index_path: Path, query_encoder: QueryEncoder):
        """
        Load an index folder with embeddings.npy, entities.jsonl and titles.jsonl.

        Args:
            index_path (Path):
                The index folder, one row per document in each file.
            query_encoder (QueryEncoder):
                The encoder that embeds queries.
        """
        embeddings = np.load(index_path / "embeddings.npy")
        self.titles, self.texts = read_index_documents(index_path)
        if len(self.titles) != len(embeddings):
            raise ValueError(
                f"titles.jsonl and entities.jsonl have {len(self.titles)} rows but "
                f"embeddings.npy has {len(embeddings)}"
            )
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.query_encoder = query_encoder

    def retrieve(self, query: str, top_n: int) -> list[Candidate]:
        """
        Return the top_n documents most similar to the query, best first.

        Args:
            query (str):
                The query text, without the E5 prefix.
            top_n (int):
                The number of documents to return, at least 1.

        Returns:
            list[Candidate]: The documents, with their cosine similarity as score.
        """
        check_top_n(top_n)
        query_vector = self.query_encoder.encode([f"query: {query}"])[0]
        scores = self.embeddings @ (query_vector / np.linalg.norm(query_vector))
        return top_candidates(scores, self.titles, self.texts, top_n)
