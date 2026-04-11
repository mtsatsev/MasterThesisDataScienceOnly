import argparse
import logging
import shutil
from pathlib import Path

import nltk

from llm_bayesian_reasoning.pipeline.config import RetrieverType
from llm_bayesian_reasoning.retrievers.base_retriever import BaseRetriever
from llm_bayesian_reasoning.retrievers.factory import (
    DEFAULT_E5_MODEL_NAME,
    build_or_load_retriever,
)

logger = logging.getLogger(__name__)

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab/english",
}


def add_common_index_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--documents-path",
        type=Path,
        required=True,
        help="JSONL corpus path with title/text fields",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        required=True,
        help="Directory where the index artifacts will be written",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of documents processed per batch while indexing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of documents to index; omit it to index the full corpus",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing target directory before rebuilding the index",
    )
    parser.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default="INFO",
        help="Logging verbosity",
    )


def add_e5_index_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_E5_MODEL_NAME,
        help="SentenceTransformer model identifier used to build the E5 index",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="SentenceTransformer device such as auto, cpu, cuda, cuda:0, or mps",
    )


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def validate_common_arguments(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1 when provided")
    if not args.documents_path.exists():
        raise FileNotFoundError(f"Documents path does not exist: {args.documents_path}")
    if not args.documents_path.is_file():
        raise ValueError(f"Documents path must be a file: {args.documents_path}")
    if args.index_path.exists() and args.index_path.is_file():
        raise ValueError(f"Index path must be a directory: {args.index_path}")


def prepare_index_directory(index_path: Path, overwrite: bool) -> None:
    if not index_path.exists():
        return

    if not any(index_path.iterdir()):
        return

    if not overwrite:
        raise FileExistsError(
            f"Index directory already exists and is not empty: {index_path}. "
            "Pass --overwrite to rebuild it."
        )

    shutil.rmtree(index_path)


def ensure_bm25_tokenizer_resources() -> None:
    for package_name, resource_path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)
            nltk.data.find(resource_path)


def _normalize_device(device: str | None) -> str | None:
    if device is None:
        return None
    normalized = device.strip()
    if not normalized or normalized.lower() == "auto":
        return None
    return normalized


def build_index(
    retriever_type: RetrieverType,
    documents_path: Path,
    index_path: Path,
    batch_size: int,
    limit: int | None,
    overwrite: bool,
    model_name: str = DEFAULT_E5_MODEL_NAME,
    device: str | None = None,
) -> BaseRetriever:
    prepare_index_directory(index_path, overwrite)

    if retriever_type == RetrieverType.BM25:
        ensure_bm25_tokenizer_resources()

    logger.info(
        "Building %s index from %s into %s",
        retriever_type.value,
        documents_path,
        index_path,
    )
    retriever = build_or_load_retriever(
        documents_path=documents_path,
        index_path=index_path,
        retriever_type=retriever_type,
        batch_size=batch_size,
        limit=limit,
        retriever_model_name=model_name,
        retriever_device=_normalize_device(device),
        show_progress=True,
    )
    entity_count = len(getattr(retriever, "entities", []) or [])
    logger.info(
        "%s index is ready at %s with %d documents",
        retriever_type.value,
        index_path,
        entity_count,
    )
    return retriever
