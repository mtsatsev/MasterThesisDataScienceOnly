#!/usr/bin/env python3
"""Build a BM25 index from a JSONL corpus with title/text fields."""

import argparse

from llm_bayesian_reasoning.pipeline.config import RetrieverType
from llm_bayesian_reasoning.retrievers.index_cli import (
    add_common_index_arguments,
    build_index,
    configure_logging,
    validate_common_arguments,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a BM25 index in the specified folder"
    )
    add_common_index_arguments(parser)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)
    validate_common_arguments(args)
    build_index(
        retriever_type=RetrieverType.BM25,
        documents_path=args.documents_path,
        index_path=args.index_path,
        batch_size=args.batch_size,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
