#!/usr/bin/env python3
"""Build an E5 dense index from a JSONL corpus with title/text fields."""

import argparse

from llm_bayesian_reasoning.pipeline.config import RetrieverType
from llm_bayesian_reasoning.retrievers.index_cli import (
    add_common_index_arguments,
    add_e5_index_arguments,
    build_index,
    configure_logging,
    validate_common_arguments,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an E5 index in the specified folder"
    )
    add_common_index_arguments(parser)
    add_e5_index_arguments(parser)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)
    validate_common_arguments(args)
    build_index(
        retriever_type=RetrieverType.E5,
        documents_path=args.documents_path,
        index_path=args.index_path,
        batch_size=args.batch_size,
        limit=args.limit,
        overwrite=args.overwrite,
        model_name=args.model_name,
        device=args.device,
    )


if __name__ == "__main__":
    main()