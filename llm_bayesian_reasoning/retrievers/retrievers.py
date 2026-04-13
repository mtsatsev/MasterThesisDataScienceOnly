import json
import logging
import pickle
from pathlib import Path

import nltk
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from llm_bayesian_reasoning.retrievers.base_retriever import BaseRetriever
from llm_bayesian_reasoning.retrievers.document import RetrievalResult, ScoredDocument

logger = logging.getLogger(__name__)


def _materialize_scored_documents(
    entities: list[str] | None,
    titles: list[str] | None,
    results: list[tuple[int, float]],
) -> list[ScoredDocument]:
    if entities is None:
        raise ValueError("Index not loaded. Call build_index() or load_index() first.")

    scored_documents: list[ScoredDocument] = []
    for idx, score in results:
        title = titles[idx] if titles is not None else entities[idx]
        scored_documents.append(
            ScoredDocument(title=title, text=entities[idx], score=score)
        )
    return scored_documents


def _build_retrieval_result(
    entities: list[str] | None,
    titles: list[str] | None,
    results: list[tuple[int, float]],
) -> RetrievalResult:
    return RetrievalResult(
        results=results,
        documents=_materialize_scored_documents(entities, titles, results),
    )


class BM25Retriever(BaseRetriever):
    """BM25-based retriever for ranking entities by relevance to a query."""

    def __init__(self):
        """Initialize BM25 retriever."""
        self.bm25 = None
        self.entities = None
        self.titles = None
        self.title_to_indices = None
        self.index_path = None

    def build_index(
        self,
        entities: list[str],
        index_path: str | Path,
        titles: list[str] | None = None,
    ) -> None:
        """
        Build BM25 index from entities. Saves to directory structure for incremental updates.

        Args:
            entities (list[str]): List of entity strings to index
            index_path (str | Path): Path to directory to save the index
            titles (list[str] | None): Optional list of titles/ids parallel to `entities`
        """
        index_dir = Path(index_path)
        index_dir.mkdir(parents=True, exist_ok=True)

        # Tokenize and build BM25
        tokenized_entities = [
            nltk.tokenize.word_tokenize(entity) for entity in entities
        ]
        self.bm25 = BM25Okapi(tokenized_entities)
        self.entities = entities

        # Save entities to JSONL for incremental appending
        with open(index_dir / "entities.jsonl", "w", encoding="utf-8") as f:
            for e in entities:
                f.write(json.dumps({"entity": e}, ensure_ascii=False) + "\n")

        # Handle titles
        self.titles = titles
        if titles is not None:
            with open(index_dir / "titles.jsonl", "w", encoding="utf-8") as f:
                for t in titles:
                    f.write(json.dumps({"title": t}, ensure_ascii=False) + "\n")

            mapping: dict[str, list[int]] = {}
            for i, t in enumerate(titles):
                mapping.setdefault(t, []).append(i)
            self.title_to_indices = mapping
        else:
            self.title_to_indices = None

        # Pickle BM25
        with open(index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

        # Save metadata
        with open(index_dir / "meta.pkl", "wb") as f:
            pickle.dump(
                {"titles": self.titles, "title_to_indices": self.title_to_indices},
                f,
            )

        self.index_path = index_path
        logger.debug(
            "Built and saved BM25 index to %s with %d entities",
            index_path,
            len(entities),
        )

    def load_index(self, index_path: str | Path) -> None:
        """
        Load a previously built BM25 index from directory.

        Args:
            index_path (str | Path): Path to the index directory
        """
        index_dir = Path(index_path)

        # Load BM25
        with open(index_dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        # Load entities
        entities = []
        with open(index_dir / "entities.jsonl", encoding="utf-8") as f:
            for line in f:
                entities.append(json.loads(line)["entity"])
        self.entities = entities

        # Load metadata
        with open(index_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
            self.titles = meta.get("titles")
            self.title_to_indices = meta.get("title_to_indices")

        self.index_path = index_path
        logger.debug(
            "Loaded BM25 index from %s with %d entities", index_path, len(self.entities)
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Retrieve top-k entities together with both scores and resolved documents.

        Args:
            query (str): Query string
            top_k (int): Number of top entities to return

        Returns:
            RetrievalResult with raw (index, score) pairs and resolved documents
        """

        if self.bm25 is None or self.entities is None:
            raise ValueError(
                "Index not loaded. Call build_index() or load_index() first."
            )
        tokenized_query = nltk.tokenize.word_tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        results = [(int(i), float(scores[i])) for i in top_indices]
        logger.debug("Retrieved %d entities for query: '%s'", len(results), query)

        return _build_retrieval_result(
            self.entities,
            self.titles,
            results,
        )

    def append_batch(
        self,
        entities: list[str],
        index_path: str | Path,
        titles: list[str] | None = None,
    ) -> None:
        """
        Append entities to an existing BM25 index. Requires finalize_index() to rebuild BM25.

        Args:
            entities (list[str]): New entity strings to append
            index_path (str): Path to the index directory
            titles (list[str] | None): Optional list of titles parallel to new entities
        """
        index_dir = Path(index_path)

        # Append entities to JSONL
        with open(index_dir / "entities.jsonl", "a", encoding="utf-8") as f:
            for e in entities:
                f.write(json.dumps({"entity": e}, ensure_ascii=False) + "\n")

        # Append titles if provided
        if titles:
            with open(index_dir / "titles.jsonl", "a", encoding="utf-8") as f:
                for t in titles:
                    f.write(json.dumps({"title": t}, ensure_ascii=False) + "\n")

        logger.debug(
            "Appended %d entities to %s. Call finalize_index() to rebuild BM25.",
            len(entities),
            index_path,
        )

    def finalize_index(self, index_path: str | Path) -> None:
        """
        Rebuild BM25 index from all accumulated entities (after append_batch calls).

        Args:
            index_path (str): Path to the index directory
        """
        index_dir = Path(index_path)

        # Load all entities
        entities = []
        with open(index_dir / "entities.jsonl", encoding="utf-8") as f:
            for line in f:
                entities.append(json.loads(line)["entity"])

        # Load all titles if they exist
        titles = None
        if (index_dir / "titles.jsonl").exists():
            titles = []
            with open(index_dir / "titles.jsonl", encoding="utf-8") as f:
                for line in f:
                    titles.append(json.loads(line)["title"])

        # Rebuild BM25
        tokenized_entities = [
            nltk.tokenize.word_tokenize(entity) for entity in entities
        ]
        self.bm25 = BM25Okapi(tokenized_entities)
        self.entities = entities
        self.titles = titles

        if titles is not None:
            mapping: dict[str, list[int]] = {}
            for i, t in enumerate(titles):
                mapping.setdefault(t, []).append(i)
            self.title_to_indices = mapping
        else:
            self.title_to_indices = None

        # Pickle BM25
        with open(index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

        # Save metadata
        with open(index_dir / "meta.pkl", "wb") as f:
            pickle.dump(
                {"titles": self.titles, "title_to_indices": self.title_to_indices},
                f,
            )

        logger.debug("Finalized BM25 index with %d entities", len(entities))


class E5Retriever(BaseRetriever):
    """E5-base-v2 dense retriever for ranking entities by semantic similarity to a query."""

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        device: str | None = None,
    ):
        """
        Initialize E5 retriever.

        Args:
            model_name: HuggingFace model identifier for E5
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.entity_embeddings = None
        self.entities = None
        self.titles = None
        self.title_to_indices = None
        self.index_path = None

        logger.debug("Loaded E5 model: %s (device=%s)", model_name, device or "auto")

    def build_index(
        self,
        entities: list[str],
        index_path: str | Path,
        titles: list[str] | None = None,
    ) -> None:
        """
        Build embeddings index from entities. Saves to directory for incremental appending.

        Args:
            entities (list[str]): List of entity strings to index
            index_path (str): Path to directory to save the index
            titles (list[str] | None): Optional list of titles/ids parallel to `entities`
        """
        index_dir = Path(index_path)
        index_dir.mkdir(parents=True, exist_ok=True)

        # Encode and save embeddings
        passages = [f"passage: {entity}" for entity in entities]
        embeddings = self.model.encode(passages, convert_to_numpy=True)
        np.save(index_dir / "embeddings.npy", embeddings)
        self.entity_embeddings = embeddings

        # Save entities to JSONL
        self.entities = entities
        with open(index_dir / "entities.jsonl", "w", encoding="utf-8") as f:
            for e in entities:
                f.write(json.dumps({"entity": e}, ensure_ascii=False) + "\n")

        # Handle titles
        self.titles = titles
        if titles is not None:
            with open(index_dir / "titles.jsonl", "w", encoding="utf-8") as f:
                for t in titles:
                    f.write(json.dumps({"title": t}, ensure_ascii=False) + "\n")

            mapping: dict[str, list[int]] = {}
            for i, t in enumerate(titles):
                mapping.setdefault(t, []).append(i)
            self.title_to_indices = mapping
        else:
            self.title_to_indices = None

        # Save metadata
        with open(index_dir / "meta.pkl", "wb") as f:
            pickle.dump(
                {"titles": self.titles, "title_to_indices": self.title_to_indices},
                f,
            )

        self.index_path = index_path
        logger.debug(
            f"Built and saved E5 index to {index_path} with {len(entities)} entities"
        )

    def load_index(self, index_path: str | Path) -> None:
        """
        Load a previously built E5 index from directory.

        Args:
            index_path: Path to the index directory
        """
        index_dir = Path(index_path)

        # Load embeddings
        self.entity_embeddings = np.load(index_dir / "embeddings.npy")

        # Load entities
        entities = []
        with open(index_dir / "entities.jsonl", encoding="utf-8") as f:
            for line in f:
                entities.append(json.loads(line)["entity"])
        self.entities = entities

        # Load metadata
        with open(index_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
            self.titles = meta.get("titles")
            self.title_to_indices = meta.get("title_to_indices")

        self.index_path = index_path
        logger.debug(
            f"Loaded E5 index from {index_path} with {len(self.entities)} entities"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Retrieve top-k entities together with both scores and resolved documents.

        Args:
            query (str): Query string
            top_k (int): Number of top entities to return

        Returns:
            RetrievalResult with raw (index, score) pairs and resolved documents
        """
        # Rebuild index if entities changed
        if self.entity_embeddings is None or self.entities is None:
            raise ValueError(
                "Index not loaded. Call build_index() or load_index() first."
            )

        # Encode query with "query: " prefix
        query_embedding = self.model.encode(f"query: {query}", convert_to_numpy=True)

        # Compute cosine similarity
        scores = np.dot(self.entity_embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [(int(i), float(scores[i])) for i in top_indices]
        logger.debug(f"Retrieved {len(results)} entities for query: '{query}'")

        return _build_retrieval_result(
            self.entities,
            self.titles,
            results,
        )

    def append_batch(
        self,
        entities: list[str],
        index_path: str | Path,
        titles: list[str] | None = None,
    ) -> None:
        """
        Append and encode new entities to an existing E5 index.

        Args:
            entities (list[str]): New entity strings to append and encode
            index_path (str): Path to the index directory
            titles (list[str] | None): Optional list of titles parallel to new entities
        """
        index_dir = Path(index_path)

        # Load existing embeddings
        old_emb = np.load(index_dir / "embeddings.npy")

        # Encode new embeddings
        passages = [f"passage: {entity}" for entity in entities]
        new_emb = self.model.encode(passages, convert_to_numpy=True)

        # Concatenate and save
        combined = np.vstack([old_emb, new_emb])
        np.save(index_dir / "embeddings.npy", combined)

        # Append entities to JSONL
        with open(index_dir / "entities.jsonl", "a", encoding="utf-8") as f:
            for e in entities:
                f.write(json.dumps({"entity": e}, ensure_ascii=False) + "\n")

        # Append titles and rebuild mapping
        if titles:
            with open(index_dir / "titles.jsonl", "a", encoding="utf-8") as f:
                for t in titles:
                    f.write(json.dumps({"title": t}, ensure_ascii=False) + "\n")

            # Reload and rebuild title_to_indices mapping
            all_titles = []
            with open(index_dir / "titles.jsonl", encoding="utf-8") as f:
                for line in f:
                    all_titles.append(json.loads(line)["title"])

            title_to_indices = {}
            for i, t in enumerate(all_titles):
                title_to_indices.setdefault(t, []).append(i)

            # Save updated metadata
            with open(index_dir / "meta.pkl", "wb") as f:
                pickle.dump(
                    {"titles": all_titles, "title_to_indices": title_to_indices},
                    f,
                )
        else:
            # No titles, just save metadata
            with open(index_dir / "meta.pkl", "wb") as f:
                pickle.dump({"titles": None, "title_to_indices": None}, f)

        logger.debug(
            f"Appended {len(entities)} entities to E5 index at {index_path}. Embeddings and metadata updated."
        )

    def finalize_index(self, index_path: Path) -> None:
        raise NotImplementedError(
            "E5 retriever does not require finalize_index() as it updates embeddings incrementally."
        )
