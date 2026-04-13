import logging
from pathlib import Path

from llm_bayesian_reasoning.pipeline.config import RetrieverConfig
from llm_bayesian_reasoning.pipeline.results_io import safe_name
from llm_bayesian_reasoning.retrievers.cache_manager import (
    RetrievalCacheManager,
    build_retriever_cache_key,
)
from llm_bayesian_reasoning.retrievers.document import RetrievalResult
from llm_bayesian_reasoning.retrievers.factory import build_or_load_retriever

logger = logging.getLogger(__name__)


class CachedRetrieverService:
    """Load one retriever and serve cached candidate pools for a fixed pool size."""

    def __init__(
        self,
        retriever_config: RetrieverConfig,
        cache_root: Path,
        retrieval_pool_size: int,
    ):
        self.retriever_config = retriever_config
        self.retrieval_pool_size = retrieval_pool_size
        self.retriever = build_or_load_retriever(
            documents_path=retriever_config.documents_path,
            index_path=retriever_config.index_path,
            retriever_type=retriever_config.retriever_type,
            batch_size=retriever_config.batch_size,
            limit=retriever_config.index_limit,
            retriever_model_name=retriever_config.retriever_model_name,
        )
        self.cache_manager = RetrievalCacheManager(cache_root / safe_name(retriever_config.name))
        self.retriever_cache_key = build_retriever_cache_key(
            retriever_config.retriever_type.value,
            retriever_config.index_path,
            retriever_config.retriever_model_name,
        )
        self._retrieval_pool_cache: dict[int | str, RetrievalResult] = {}

    def get_candidate_pool(
        self,
        record_id: int | str,
        query: str,
    ) -> RetrievalResult:
        if record_id in self._retrieval_pool_cache:
            return self._retrieval_pool_cache[record_id]

        cached = self.cache_manager.get(
            query=query,
            retriever_key=self.retriever_cache_key,
            requested_pool_size=self.retrieval_pool_size,
            retriever=self.retriever,
        )
        if cached is not None:
            self._retrieval_pool_cache[record_id] = cached
            return cached

        retrieval_result = self.retriever.retrieve(query, top_k=self.retrieval_pool_size)
        stored = self.cache_manager.set(
            query=query,
            retriever_key=self.retriever_cache_key,
            retrieval_result=retrieval_result,
            metadata={
                "retriever_name": self.retriever_config.name,
                "retriever_type": self.retriever_config.retriever_type.value,
                "index_path": str(self.retriever_config.index_path),
                "retriever_model_name": self.retriever_config.retriever_model_name,
            },
        )
        self._retrieval_pool_cache[record_id] = stored
        return stored