import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def _normalize_text(value: str) -> str:
    normalized = value.replace("{x}", "{X}")
    normalized = normalized.replace("{X}", "")
    normalized = normalized.replace("'", "")
    normalized = normalized.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


class WeakLabelConfig(BaseModel):
    negative_target: float = Field(default=0.05, ge=0.0, le=1.0)
    positive_without_evidence_target: float = Field(default=0.55, ge=0.0, le=1.0)
    partial_evidence_target: float = Field(default=0.7, ge=0.0, le=1.0)
    complete_evidence_target: float = Field(default=0.85, ge=0.0, le=1.0)
    attributed_target: float = Field(default=0.95, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class DeepProbLogRow(BaseModel):
    id: int | str
    query: str = Field(min_length=1)
    original_query: str | None = None
    atoms: list[str] = Field(default_factory=list)
    negated_atoms: list[str] = Field(default_factory=list)
    logical_query: str = Field(
        min_length=1,
        validation_alias=AliasChoices("logical_query", "logical query"),
    )
    entity: str = Field(min_length=1)
    text: str = Field(default="")
    relevance: int = Field(ge=0, le=1)
    weight: float = Field(default=1.0, ge=0.0)
    source: str = Field(default="unknown", min_length=1)
    domain: str | None = None
    template: str | None = None
    evidence_ratings: list[str] | None = None
    attributions: list[dict[str, Any]] | None = None

    model_config = ConfigDict(extra="ignore")

    def atom_targets(
        self,
        weak_label_config: WeakLabelConfig | None = None,
    ) -> list[float]:
        config = weak_label_config or WeakLabelConfig()
        if self.relevance == 0:
            return [config.negative_target for _ in self.atoms]

        evidence_strings = [
            _normalize_text(evidence)
            for evidence in (self.evidence_ratings or [])
            if isinstance(evidence, str)
        ]
        evidence_level = config.positive_without_evidence_target
        if any("complete" in evidence for evidence in evidence_strings):
            evidence_level = config.complete_evidence_target
        elif any("partial" in evidence for evidence in evidence_strings):
            evidence_level = config.partial_evidence_target

        attribution_keys: list[str] = []
        for attribution in self.attributions or []:
            for key in attribution:
                if isinstance(key, str):
                    attribution_keys.append(_normalize_text(key))

        targets: list[float] = []
        for atom in self.atoms:
            atom_norm = _normalize_text(atom)
            is_directly_attributed = any(
                atom_norm == key or atom_norm in key or key in atom_norm
                for key in attribution_keys
            )
            if is_directly_attributed:
                targets.append(config.attributed_target)
            else:
                targets.append(evidence_level)
        return targets


class DeepProbLogCandidate(BaseModel):
    entity: str = Field(min_length=1)
    text: str = Field(default="")
    relevance: int = Field(ge=0, le=1)
    weight: float = Field(default=1.0, ge=0.0)
    source: str = Field(min_length=1)
    evidence_ratings: list[str] | None = None
    attributions: list[dict[str, Any]] | None = None
    atom_targets: list[float] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class DeepProbLogGroupedExample(BaseModel):
    id: int | str
    query: str = Field(min_length=1)
    original_query: str | None = None
    logical_query: str = Field(min_length=1)
    atoms: list[str] = Field(default_factory=list)
    negated_atoms: list[str] = Field(default_factory=list)
    domain: str | None = None
    template: str | None = None
    candidates: list[DeepProbLogCandidate] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class AtomClassificationExample(BaseModel):
    feature_id: int = Field(ge=0)
    query_id: int = Field(ge=0)
    candidate_id: int = Field(ge=0)
    atom_index: int = Field(ge=0)
    query: str = Field(min_length=1)
    entity: str = Field(min_length=1)
    text: str = Field(default="")
    atom: str = Field(min_length=1)
    target: float = Field(ge=0.0, le=1.0)
    label: int = Field(ge=0, le=1)
    weight: float = Field(default=1.0, ge=0.0)

    model_config = ConfigDict(extra="forbid")


def read_deepproblog_rows(
    path: str | Path,
    limit: int | None = None,
) -> list[DeepProbLogRow]:
    rows: list[DeepProbLogRow] = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            payload = json.loads(line)
            attributions = payload.get("attributions")
            if isinstance(attributions, list):
                payload["attributions"] = [
                    attribution
                    for attribution in attributions
                    if isinstance(attribution, dict)
                ]
            rows.append(DeepProbLogRow.model_validate(payload))
    return rows


def group_deepproblog_rows(
    rows: list[DeepProbLogRow],
    weak_label_config: WeakLabelConfig | None = None,
) -> list[DeepProbLogGroupedExample]:
    grouped: dict[int | str, list[DeepProbLogRow]] = defaultdict(list)
    for row in rows:
        grouped[row.id].append(row)

    grouped_examples: list[DeepProbLogGroupedExample] = []
    for record_id, record_rows in grouped.items():
        first = record_rows[0]
        candidates = [
            DeepProbLogCandidate(
                entity=row.entity,
                text=row.text,
                relevance=row.relevance,
                weight=row.weight,
                source=row.source,
                evidence_ratings=row.evidence_ratings,
                attributions=row.attributions,
                atom_targets=row.atom_targets(weak_label_config),
            )
            for row in record_rows
        ]
        grouped_examples.append(
            DeepProbLogGroupedExample(
                id=record_id,
                query=first.query,
                original_query=first.original_query,
                logical_query=first.logical_query,
                atoms=first.atoms,
                negated_atoms=first.negated_atoms,
                domain=first.domain,
                template=first.template,
                candidates=candidates,
            )
        )

    grouped_examples.sort(key=lambda example: str(example.id))
    return grouped_examples


def _query_stratum(example: DeepProbLogGroupedExample) -> str:
    template = (example.template or "").strip().lower()
    domain = (example.domain or "").strip().lower()
    atom_bucket = str(min(len(example.atoms), 4))
    candidate_bucket = str(min(len(example.candidates), 4))
    positive_bucket = str(
        min(sum(int(candidate.relevance > 0) for candidate in example.candidates), 3)
    )

    if template:
        return f"template:{template}|atoms:{atom_bucket}|positives:{positive_bucket}"
    if domain:
        return f"domain:{domain}|atoms:{atom_bucket}|positives:{positive_bucket}"
    return (
        f"atoms:{atom_bucket}|candidates:{candidate_bucket}|positives:{positive_bucket}"
    )


def _stratify_labels(
    grouped_examples: list[DeepProbLogGroupedExample],
) -> list[str] | None:
    if len(grouped_examples) < 2:
        return None

    labels = [_query_stratum(example) for example in grouped_examples]
    counts = {label: labels.count(label) for label in set(labels)}
    if len(counts) < 2:
        return None
    if min(counts.values()) < 2:
        return None
    return labels


def _split_grouped_indices(
    indices: list[int],
    train_size: float | int,
    seed: int,
    grouped_examples: list[DeepProbLogGroupedExample],
) -> tuple[list[int], list[int]]:
    stratify_labels = _stratify_labels(grouped_examples)
    stratify = None
    if stratify_labels is not None:
        stratify = [stratify_labels[index] for index in indices]

    try:
        train_indices, holdout_indices = train_test_split(
            indices,
            train_size=train_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        train_indices, holdout_indices = train_test_split(
            indices,
            train_size=train_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
    return list(train_indices), list(holdout_indices)


def select_grouped_example_subset(
    grouped_examples: list[DeepProbLogGroupedExample],
    limit: int | None,
    seed: int,
) -> list[DeepProbLogGroupedExample]:
    if limit is None or limit >= len(grouped_examples):
        return grouped_examples
    if limit < 1:
        raise ValueError("limit must be >= 1 when provided")

    selected_indices, _ = _split_grouped_indices(
        indices=list(range(len(grouped_examples))),
        train_size=limit,
        seed=seed,
        grouped_examples=grouped_examples,
    )
    selected_examples = [grouped_examples[index] for index in selected_indices]
    selected_examples.sort(key=lambda example: str(example.id))
    return selected_examples


def split_grouped_examples(
    grouped_examples: list[DeepProbLogGroupedExample],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[
    list[DeepProbLogGroupedExample],
    list[DeepProbLogGroupedExample],
    list[DeepProbLogGroupedExample],
]:
    total = train_fraction + val_fraction + test_fraction
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1")
    if any(value < 0.0 for value in (train_fraction, val_fraction, test_fraction)):
        raise ValueError("split fractions must be >= 0")
    if not grouped_examples:
        return [], [], []

    all_indices = list(range(len(grouped_examples)))
    if math.isclose(train_fraction, 1.0, rel_tol=0.0, abs_tol=1e-9):
        return list(grouped_examples), [], []
    if math.isclose(train_fraction, 0.0, rel_tol=0.0, abs_tol=1e-9):
        train_indices = []
        holdout_indices = all_indices
    else:
        train_indices, holdout_indices = _split_grouped_indices(
            indices=all_indices,
            train_size=train_fraction,
            seed=seed,
            grouped_examples=grouped_examples,
        )

    holdout_total = val_fraction + test_fraction
    if math.isclose(holdout_total, 0.0, rel_tol=0.0, abs_tol=1e-9):
        val_indices = []
        test_indices = []
    elif math.isclose(val_fraction, 0.0, rel_tol=0.0, abs_tol=1e-9):
        val_indices = []
        test_indices = holdout_indices
    elif math.isclose(test_fraction, 0.0, rel_tol=0.0, abs_tol=1e-9):
        val_indices = holdout_indices
        test_indices = []
    else:
        val_share = val_fraction / holdout_total
        val_indices, test_indices = _split_grouped_indices(
            indices=holdout_indices,
            train_size=val_share,
            seed=seed,
            grouped_examples=grouped_examples,
        )

    return (
        [grouped_examples[index] for index in train_indices],
        [grouped_examples[index] for index in val_indices],
        [grouped_examples[index] for index in test_indices],
    )


def rows_from_grouped_examples(
    grouped_examples: list[DeepProbLogGroupedExample],
) -> list[DeepProbLogRow]:
    rows: list[DeepProbLogRow] = []
    for example in grouped_examples:
        for candidate in example.candidates:
            rows.append(
                DeepProbLogRow(
                    id=example.id,
                    query=example.query,
                    original_query=example.original_query,
                    atoms=list(example.atoms),
                    negated_atoms=list(example.negated_atoms),
                    logical_query=example.logical_query,
                    entity=candidate.entity,
                    text=candidate.text,
                    relevance=candidate.relevance,
                    weight=candidate.weight,
                    source=candidate.source,
                    domain=example.domain,
                    template=example.template,
                    evidence_ratings=candidate.evidence_ratings,
                    attributions=candidate.attributions,
                )
            )
    return rows


def flatten_atom_supervision_examples(
    grouped_examples: list[DeepProbLogGroupedExample],
) -> list[AtomClassificationExample]:
    flattened: list[AtomClassificationExample] = []
    feature_id = 0
    for query_id, example in enumerate(grouped_examples):
        for candidate_id, candidate in enumerate(example.candidates):
            for atom_index, atom in enumerate(example.atoms):
                target = candidate.atom_targets[atom_index]
                flattened.append(
                    AtomClassificationExample(
                        feature_id=feature_id,
                        query_id=query_id,
                        candidate_id=candidate_id,
                        atom_index=atom_index,
                        query=example.query,
                        entity=candidate.entity,
                        text=candidate.text,
                        atom=atom,
                        target=target,
                        label=candidate.relevance,
                        weight=candidate.weight,
                    )
                )
                feature_id += 1
    return flattened


class AtomClassificationDataset(Dataset):
    def __init__(self, examples: list[AtomClassificationExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> AtomClassificationExample:
        return self.examples[index]


class GroupedQueryDataset(Dataset):
    def __init__(self, examples: list[DeepProbLogGroupedExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DeepProbLogGroupedExample:
        return self.examples[index]
