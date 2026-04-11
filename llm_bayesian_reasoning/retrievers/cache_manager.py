import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_bayesian_reasoning.retrievers.document import RetrievalResult, ScoredDocument

logger = logging.getLogger(__name__)


class CachedDocumentRef(BaseModel):
    """Compact on-disk reference to a retrieved document."""

    idx: int = Field(ge=0)
    title: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class CachedRetrievalEntry(BaseModel):
    """Serialized retrieval candidate pool stored on disk."""

    query: str = Field(min_length=1)
    retriever_key: str = Field(min_length=1)
    pool_size: int = Field(ge=1)
    results: list[tuple[int, float]]
    documents: list[CachedDocumentRef]
    metadata: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


def build_retriever_cache_key(
    retriever_type: str,
    index_path: str | Path,
    retriever_model_name: str | None = None,
) -> str:
    payload = {
        "retriever_type": retriever_type,
        "index_path": str(index_path),
        "retriever_model_name": retriever_model_name or "",
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"{retriever_type.lower()}_{digest[:16]}"


class RetrievalCacheManager:
    """Disk-backed cache for reusable retrieval candidate pools."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._entries_by_retriever: dict[str, dict[str, CachedRetrievalEntry]] = {}

    def _cache_file_path(self, retriever_key: str) -> Path:
        return self.cache_dir / f"{retriever_key}.jsonl"

    def _load_entries(self, retriever_key: str) -> dict[str, CachedRetrievalEntry]:
        if retriever_key in self._entries_by_retriever:
            return self._entries_by_retriever[retriever_key]

        cache_file = self._cache_file_path(retriever_key)
        entries: dict[str, CachedRetrievalEntry] = {}
        if cache_file.exists():
            with cache_file.open(encoding="utf-8") as file_handle:
                for line in file_handle:
                    line = line.strip()
                    if not line:
                        continue
                    entry = CachedRetrievalEntry.model_validate_json(line)
                    current = entries.get(entry.query)
                    if current is None or entry.pool_size >= current.pool_size:
                        entries[entry.query] = entry

        self._entries_by_retriever[retriever_key] = entries
        return entries

    @staticmethod
    def _materialize_result(
        retriever: Any,
        entry: CachedRetrievalEntry,
        requested_pool_size: int,
    ) -> RetrievalResult:
        entities = getattr(retriever, "entities", None)
        titles = getattr(retriever, "titles", None)
        if entities is None:
            raise ValueError(
                "Retriever must expose loaded entities to resolve cache hits"
            )

        sliced_results = entry.results[:requested_pool_size]
        documents: list[ScoredDocument] = []
        for idx, score in sliced_results:
            title = titles[idx] if titles is not None else entities[idx]
            documents.append(
                ScoredDocument(title=title, text=entities[idx], score=score)
            )
        return RetrievalResult(results=sliced_results, documents=documents)

    @staticmethod
    def _slice_result(
        result: RetrievalResult, requested_pool_size: int
    ) -> RetrievalResult:
        return RetrievalResult(
            results=result.results[:requested_pool_size],
            documents=result.documents[:requested_pool_size],
        )

    def get(
        self,
        query: str,
        retriever_key: str,
        requested_pool_size: int,
        retriever: Any,
    ) -> RetrievalResult | None:
        entries = self._load_entries(retriever_key)
        entry = entries.get(query)
        if entry is None:
            return None
        if entry.pool_size < requested_pool_size:
            logger.debug(
                "Cached retrieval pool too small for key=%s query=%r: cached=%d requested=%d",
                retriever_key,
                query,
                entry.pool_size,
                requested_pool_size,
            )
            return None

        logger.debug(
            "Loaded cached retrieval pool for key=%s query=%r size=%d",
            retriever_key,
            query,
            requested_pool_size,
        )
        return self._materialize_result(retriever, entry, requested_pool_size)

    def set(
        self,
        query: str,
        retriever_key: str,
        retrieval_result: RetrievalResult,
        metadata: dict[str, str] | None = None,
    ) -> RetrievalResult:
        entry = CachedRetrievalEntry(
            query=query,
            retriever_key=retriever_key,
            pool_size=len(retrieval_result.documents),
            results=retrieval_result.results,
            documents=[
                CachedDocumentRef(idx=idx, title=document.title)
                for (idx, _score), document in zip(
                    retrieval_result.results,
                    retrieval_result.documents,
                    strict=True,
                )
            ],
            metadata=metadata or {},
        )
        entries = self._load_entries(retriever_key)
        current = entries.get(query)
        if current is None or entry.pool_size >= current.pool_size:
            entries[query] = entry

        cache_file = self._cache_file_path(retriever_key)
        with cache_file.open("a", encoding="utf-8") as file_handle:
            file_handle.write(entry.model_dump_json() + "\n")
        return retrieval_result
