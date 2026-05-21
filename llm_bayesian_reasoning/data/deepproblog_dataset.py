from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
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
