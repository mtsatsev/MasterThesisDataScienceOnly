import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from llm_bayesian_reasoning.retrievers.factory import build_or_load_retriever

logger = logging.getLogger("build_dpp_dataset")


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            rows.append(json.loads(line))
    return rows


def _load_documents(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    documents = _read_jsonl(path)
    by_title = {document["title"]: document for document in documents}
    return documents, by_title


def _base_row(record: dict[str, Any]) -> dict[str, Any]:
    parsed = record.get("parsed") or {}
    metadata = record.get("metadata") or {}
    return {
        "id": record.get("id"),
        "query": record.get("query", ""),
        "original_query": record.get("original_query"),
        "atoms": parsed.get("atoms", []),
        "negated_atoms": parsed.get("negated_atoms", []),
        "logical_query": parsed.get("logical query") or parsed.get("logical_query"),
        "domain": metadata.get("domain"),
        "template": metadata.get("template"),
    }


def _make_row(
    record: dict[str, Any],
    entity: str,
    text: str,
    relevance: int,
    weight: float,
    source: str,
) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    evidence_ratings = metadata.get("evidence_ratings") or {}
    attributions = metadata.get("attributions") or {}

    row = _base_row(record)
    row.update(
        {
            "entity": entity,
            "text": text,
            "relevance": relevance,
            "weight": weight,
            "source": source,
            "evidence_ratings": evidence_ratings.get(entity),
            "attributions": attributions.get(entity),
        }
    )
    return row


def _positive_entities(record: dict[str, Any]) -> list[str]:
    metadata = record.get("metadata") or {}
    relevance_ratings = metadata.get("relevance_ratings") or {}
    return [entity for entity in relevance_ratings.keys() if isinstance(entity, str)]


def _collect_positive_rows(
    records: list[dict[str, Any]],
    documents_by_title: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int | str, set[str]], list[dict[str, Any]]]:
    dataset: list[dict[str, Any]] = []
    positives_by_id: dict[int | str, set[str]] = {}
    usable_records: list[dict[str, Any]] = []

    for record in records:
        record_id = record.get("id")
        if record_id is None:
            continue

        positive_entities = _positive_entities(record)
        if not positive_entities:
            continue

        usable_records.append(record)
        positives_by_id[record_id] = set(positive_entities)

        for entity in positive_entities:
            document = documents_by_title.get(entity)
            dataset.append(
                _make_row(
                    record=record,
                    entity=entity,
                    text=document["text"] if document is not None else "",
                    relevance=1,
                    weight=1.0,
                    source="positive",
                )
            )

    return dataset, positives_by_id, usable_records


def _append_hard_negatives(
    dataset: list[dict[str, Any]],
    usable_records: list[dict[str, Any]],
    positives_by_id: dict[int | str, set[str]],
    retriever,
    negatives_per_positive: int,
    negative_weight: float,
) -> None:
    if negatives_per_positive <= 0:
        return

    for record in usable_records:
        record_id = record["id"]
        positive_entities = positives_by_id[record_id]
        if not positive_entities:
            continue

        target_count = len(positive_entities) * negatives_per_positive
        retrieval_depth = max(target_count * 3, len(positive_entities) + target_count)
        retrieval = retriever.retrieve(record["query"], top_k=retrieval_depth)

        added_entities: set[str] = set()
        for document in retrieval.documents:
            if document.title in positive_entities or document.title in added_entities:
                continue

            dataset.append(
                _make_row(
                    record=record,
                    entity=document.title,
                    text=document.text,
                    relevance=0,
                    weight=negative_weight,
                    source="hard_negative",
                )
            )
            added_entities.add(document.title)

            if len(added_entities) >= target_count:
                break


def _append_random_negatives(
    dataset: list[dict[str, Any]],
    usable_records: list[dict[str, Any]],
    positives_by_id: dict[int | str, set[str]],
    documents: list[dict[str, Any]],
    negatives_per_positive: int,
    negative_weight: float,
    rng: random.Random,
) -> None:
    if negatives_per_positive <= 0:
        return

    for record in usable_records:
        record_id = record["id"]
        positive_entities = positives_by_id[record_id]
        if not positive_entities:
            continue

        target_count = len(positive_entities) * negatives_per_positive
        candidate_pool = [
            document for document in documents if document["title"] not in positive_entities
        ]
        if not candidate_pool:
            continue

        sample_size = min(target_count, len(candidate_pool))
        for document in rng.sample(candidate_pool, k=sample_size):
            dataset.append(
                _make_row(
                    record=record,
                    entity=document["title"],
                    text=document["text"],
                    relevance=0,
                    weight=negative_weight,
                    source="random_negative",
                )
            )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _default_output_path(parsed_data_path: Path) -> Path:
    return parsed_data_path.with_name(parsed_data_path.stem + "_dpp_dataset.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DeepProbLog training dataset")
    parser.add_argument(
        "--parsed-data-path",
        type=Path,
        default=Path(
            "llm_bayesian_reasoning/data/preprocessed_data/parsed_test_with_negs.jsonl"
        ),
        help="QUEST-style parsed JSONL with metadata.relevance_ratings",
    )
    parser.add_argument(
        "--documents-path",
        type=Path,
        default=Path("llm_bayesian_reasoning/data/index_data/documents.jsonl"),
        help="Corpus JSONL with title/text fields",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("index/bm25_dataset"),
        help="BM25 index path used for hard-negative retrieval; built on first run",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output JSONL path; defaults next to the parsed input",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of parsed query records for smoke tests",
    )
    parser.add_argument(
        "--hard-negatives-per-positive",
        type=int,
        default=2,
        help="Number of BM25 negatives to add per positive entity",
    )
    parser.add_argument(
        "--random-negatives-per-positive",
        type=int,
        default=1,
        help="Number of random negatives to add per positive entity",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used for random-negative sampling",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_path = args.output_path or _default_output_path(args.parsed_data_path)
    logger.info("Loading parsed records from %s", args.parsed_data_path)
    records = _read_jsonl(args.parsed_data_path, limit=args.limit)

    logger.info("Loading documents from %s", args.documents_path)
    documents, documents_by_title = _load_documents(args.documents_path)

    dataset, positives_by_id, usable_records = _collect_positive_rows(
        records,
        documents_by_title,
    )
    logger.info(
        "Collected %d positive rows from %d query records",
        len(dataset),
        len(usable_records),
    )

    if args.hard_negatives_per_positive > 0 and usable_records:
        logger.info("Building/loading retriever from %s", args.index_path)
        retriever = build_or_load_retriever(
            documents_path=args.documents_path,
            index_path=args.index_path,
            show_progress=True,
        )
        _append_hard_negatives(
            dataset=dataset,
            usable_records=usable_records,
            positives_by_id=positives_by_id,
            retriever=retriever,
            negatives_per_positive=args.hard_negatives_per_positive,
            negative_weight=0.5,
        )

    _append_random_negatives(
        dataset=dataset,
        usable_records=usable_records,
        positives_by_id=positives_by_id,
        documents=documents,
        negatives_per_positive=args.random_negatives_per_positive,
        negative_weight=0.2,
        rng=random.Random(args.seed),
    )

    _write_jsonl(output_path, dataset)

    positive_count = sum(1 for row in dataset if row["relevance"] == 1)
    negative_count = len(dataset) - positive_count
    logger.info("Wrote %d rows to %s", len(dataset), output_path)
    logger.info("Positives: %d | Negatives: %d", positive_count, negative_count)


if __name__ == "__main__":
    main()
