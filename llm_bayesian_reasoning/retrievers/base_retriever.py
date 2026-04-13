import logging
from abc import ABC, abstractmethod
from pathlib import Path

from llm_bayesian_reasoning.retrievers.document import RetrievalResult

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def build_index(
        self, entities: list[str], index_path: Path, titles: list[str] | None = None
    ) -> None:
        """
        Build index from entities.

        Args:
            entities: List of entity strings to index
            index_path: Path where the index should be saved
            titles: Optional list of titles/ids parallel to `entities`
        """

    @abstractmethod
    def load_index(self, index_path: Path) -> None:
        """
        Load a previously built index.

        Args:
            index_path (Path): Path to the saved index
        """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """
        Retrieve top-k entities together with both raw scores and resolved documents.

        Args:
            query (str): Query string
            top_k (int): Number of top entities to return

        Returns:
            RetrievalResult: Containing raw (index, score) pairs and resolved documents
        """

    @abstractmethod
    def append_batch(
        self, entities: list[str], index_path: Path, titles: list[str] | None = None
    ) -> None:
        """
        Append a batch of entities to the existing index.

        Args:
            entities (list[str]): List of entity strings to append
            index_path (Path): Path to the existing index
            titles (list[str] | None): Optional list of titles/ids parallel to `entities`
        """

    @abstractmethod
    def finalize_index(self, index_path: Path) -> None:
        """
        Finalize the index after all batches have been appended.

        Args:
            index_path (Path): Path to the existing index
        """
