import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def build_index(
        self, entities: list[str], index_path: str, titles: list[str] | None = None
    ) -> None:
        """
        Build index from entities.

        Args:
            entities: List of entity strings to index
            index_path: Path where the index should be saved
            titles: Optional list of titles/ids parallel to `entities`
        """

    @abstractmethod
    def load_index(self, index_path: str) -> None:
        """
        Load a previously built index.

        Args:
            index_path: Path to the saved index
        """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Retrieve top-k entities most relevant to the query.

        Args:
            query: Query string
            top_k: Number of top entities to return

        Returns:
            List of tuples (index, score) for top-k entities sorted by relevance
        """

    @abstractmethod
    def append_batch(
        self, entities: list[str], index_path: str, titles: list[str] | None = None
    ) -> None:
        """
        Append a batch of entities to the existing index.

        Args:
            entities: List of entity strings to append
            titles: Optional list of titles/ids parallel to `entities`
        """
