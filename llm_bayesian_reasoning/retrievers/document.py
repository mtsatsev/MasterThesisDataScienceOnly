import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__file__)


class Document(BaseModel):
    """Represents a document with its title and text."""

    title: str
    text: str


class ExampleMetadata(BaseModel):
    """Optional metadata used for analysis."""

    template: str | None = None
    domain: str | None = None
    fluency: Sequence[str] | None = None
    meaning: Sequence[str] | None = None
    naturalness: Sequence[str] | None = None
    relevance_ratings: dict[str, Sequence[str]] | None = None
    evidence_ratings: dict[str, Sequence[str]] | None = None
    attributions: dict[str, Sequence[dict[str, Any]]] | None = None

    @field_validator("attributions", mode="before")
    @classmethod
    def _clean_attributions(cls, v: Any):
        if v is None:
            return None
        if not isinstance(v, dict):
            raise TypeError("attributions must be a dict[str, list[dict]] | None")

        cleaned: dict[str, list[dict[str, Any]]] = {}
        for key, items in v.items():
            if items is None:
                continue
            if not isinstance(items, (list, tuple)):
                items = [items]

            only_dicts = [x for x in items if isinstance(x, dict)]
            if only_dicts:
                cleaned[str(key)] = only_dicts

        return cleaned or None


class Example(BaseModel):
    """Represents a query paired with a set of documents."""

    query: str
    docs: list[str]
    original_query: str | None = None
    scores: list[float] | None = None
    metadata: ExampleMetadata | None = None


def read(
    filepath: str | Path, limit: int | None = None, verbose: bool = False
) -> list[Any]:
    """Read jsonl file to a List of Dicts."""
    data: list[Any] = []
    with open(filepath, encoding="utf-8") as jsonl_file:
        for idx, line in enumerate(jsonl_file):
            if limit is not None and idx >= limit:
                break
            if verbose and idx % 100 == 0:
                logger.debug("Processing line %d.", idx)
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error("Failed to parse line: `%s`", line)
                raise e
    logger.debug("Loaded %d lines from %s.", len(data), filepath)
    return data


def write(filepath: str | Path, rows: Sequence[BaseModel]) -> None:
    """
    Write a list of Pydantic BaseModel instances to a jsonl file, one JSON object per
    line.

    Args:
        filepath (str | Path): The path to the output jsonl file.
        rows (Sequence[BaseModel]): The list of Pydantic BaseModel instances to write.
    """
    with open(filepath, "w", encoding="utf-8") as jsonl_file:
        for row in rows:
            line = f"{json.dumps(row.model_dump())}\n"
            jsonl_file.write(line)
    logger.debug("Wrote %d lines to %s.", len(rows), filepath)


def read_documents(filepath: str | Path, limit: int | None = None) -> list[Document]:
    """
    Read a jsonl file containing documents and return a list of Document instances.

    Args:
        filepath (str | Path): The path to the input jsonl file.
        limit (int | None, optional): The maximum number of documents to read.
        Defaults to None.


    Returns:
        list[Document]: The list of Document instances.
    """
    documents_json = read(filepath, limit=limit, verbose=True)
    return [Document.model_validate(document) for document in documents_json]


def write_documents(filepath: str | Path, documents: list[Document]) -> None:
    """
    Write a list of Document instances to a jsonl file.

    Args:
        filepath (str | Path): The path to the output jsonl file.
        documents (list[Document]): The list of Document instances to write.
    """
    write(filepath, documents)


def read_examples(
    filepath: str | Path, limit: int | None = None, verbose: bool = False
) -> list[Example]:
    """
    Read a jsonl file containing examples and return a list of Example instances.

    Args:
        filepath (str | Path): The path to the input jsonl file.
        limit (int | None, optional): The maximum number of examples to read.
        Defaults to None.

    Returns:
        list[Example]: The list of Example instances.
    """
    examples_json = read(filepath, limit=limit, verbose=verbose)
    examples: list[Example] = []
    for example in examples_json:
        metadata = ExampleMetadata.model_validate(example["metadata"])
        parsed = Example(
            query=example["query"],
            docs=example["docs"],
            original_query=example.get("original_query"),
            scores=example.get("scores"),
            metadata=metadata,
        )
        examples.append(parsed)
    return examples


def write_examples(filepath: str | Path, examples: list[Example]) -> None:
    """
    Write a list of Example instances to a jsonl file.

    Args:
        filepath (str | Path): The path to the output jsonl file.
        examples (list[Example]): The list of Example instances to write.
    """
    write(filepath, examples)
