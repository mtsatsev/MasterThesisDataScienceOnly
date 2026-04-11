import json
import logging
from pathlib import Path

from llm_bayesian_reasoning.pipeline.config import RetrieverType
from llm_bayesian_reasoning.retrievers.base_retriever import BaseRetriever
from llm_bayesian_reasoning.retrievers.retrievers import BM25Retriever, E5Retriever

logger = logging.getLogger(__name__)

DEFAULT_E5_MODEL_NAME = "intfloat/e5-base-v2"


def create_retriever(
    retriever_type: RetrieverType,
    retriever_model_name: str = DEFAULT_E5_MODEL_NAME,
) -> BaseRetriever:
    """Instantiate a retriever from configuration."""
    if retriever_type == RetrieverType.BM25:
        return BM25Retriever()
    if retriever_type == RetrieverType.E5:
        return E5Retriever(model_name=retriever_model_name)
    raise ValueError(f"Unsupported retriever type: {retriever_type}")


def _iter_document_batches(
    documents_path: str | Path,
    batch_size: int,
    limit: int | None = None,
):
    entities_batch: list[str] = []
    titles_batch: list[str] = []
    total = 0

    with open(documents_path, encoding="utf-8") as file_handle:
        for line in file_handle:
            if limit is not None and total >= limit:
                break

            document = json.loads(line)
            entities_batch.append(document["text"])
            titles_batch.append(document["title"])
            total += 1

            if len(entities_batch) >= batch_size:
                yield entities_batch, titles_batch
                entities_batch = []
                titles_batch = []

    if entities_batch:
        yield entities_batch, titles_batch


def build_or_load_retriever(
    documents_path: str | Path,
    index_path: str | Path,
    retriever_type: RetrieverType = RetrieverType.BM25,
    batch_size: int = 1000,
    limit: int | None = None,
    retriever_model_name: str = DEFAULT_E5_MODEL_NAME,
) -> BaseRetriever:
    """Build or load a configured retriever index."""
    index_dir = Path(index_path)
    retriever = create_retriever(retriever_type, retriever_model_name)

    index_artifact = {
        RetrieverType.BM25: index_dir / "bm25.pkl",
        RetrieverType.E5: index_dir / "embeddings.npy",
    }[retriever_type]

    if index_artifact.exists():
        logger.info(
            "Loading existing %s index from %s", retriever_type.value, index_dir
        )
        retriever.load_index(index_dir)
        return retriever

    logger.info(
        "Building %s index from %s into %s (batch_size=%d)",
        retriever_type.value,
        documents_path,
        index_dir,
        batch_size,
    )
    index_dir.mkdir(parents=True, exist_ok=True)

    if retriever_type == RetrieverType.BM25:
        total = 0
        for entities_batch, titles_batch in _iter_document_batches(
            documents_path,
            batch_size,
            limit,
        ):
            retriever.append_batch(
                entities=entities_batch,
                index_path=index_dir,
                titles=titles_batch,
            )
            total += len(entities_batch)
            logger.debug("Appended BM25 batch; total so far: %d", total)

        logger.info("Finalizing BM25 index with %d documents", total)
        retriever.finalize_index(index_dir)
        retriever.load_index(index_dir)
        return retriever

    built = False
    total = 0
    for entities_batch, titles_batch in _iter_document_batches(
        documents_path,
        batch_size,
        limit,
    ):
        if not built:
            retriever.build_index(
                entities=entities_batch,
                index_path=index_dir,
                titles=titles_batch,
            )
            built = True
        else:
            retriever.append_batch(
                entities=entities_batch,
                index_path=index_dir,
                titles=titles_batch,
            )
        total += len(entities_batch)
        logger.debug("Built E5 batch; total so far: %d", total)

    if not built:
        raise ValueError(f"No documents were loaded from {documents_path}")

    retriever.load_index(index_dir)
    logger.info("Built E5 index with %d documents", total)
    return retriever
