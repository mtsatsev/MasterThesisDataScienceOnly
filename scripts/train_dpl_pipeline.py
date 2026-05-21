#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import re
import time
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from hashlib import blake2b
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from deepproblog.dataset import DataLoader as DeepProbLogDataLoader
from deepproblog.dataset import Dataset as DeepProbLogDataset
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.query import Query
from deepproblog.train import TrainObject
from problog.logic import Constant, Term
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import DistilBertConfig, DistilBertModel

logger = logging.getLogger("train_dpl_pipeline")

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)


@dataclass(slots=True)
class TrainingConfig:
    seed: int
    data_path: Path
    output_dir: Path
    max_examples: int | None
    compute_baseline_metrics: bool
    train_fraction: float
    val_fraction: float
    test_fraction: float
    tensor_source_name: str
    max_length: int
    vocab_size: int
    hidden_size: int
    n_layers: int
    n_heads: int
    learning_rate: float
    batch_size: int
    epochs: int
    device: str
    problog_log_level: str
    use_mlflow: bool
    mlflow_tracking_uri: str | None
    mlflow_experiment_name: str
    mlflow_run_name: str | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainingConfig":
        return cls(
            seed=int(raw.get("seed", 7)),
            data_path=Path(raw["data_path"]),
            output_dir=Path(raw["output_dir"]),
            max_examples=(
                None if raw.get("max_examples") is None else int(raw["max_examples"])
            ),
            compute_baseline_metrics=parse_bool(
                raw.get("compute_baseline_metrics", True)
            ),
            train_fraction=float(raw.get("train_fraction", 0.70)),
            val_fraction=float(raw.get("val_fraction", 0.15)),
            test_fraction=float(raw.get("test_fraction", 0.15)),
            tensor_source_name=str(raw.get("tensor_source_name", "atom_inputs")),
            max_length=int(raw.get("max_length", 256)),
            vocab_size=int(raw.get("vocab_size", 32768)),
            hidden_size=int(raw.get("hidden_size", 32)),
            n_layers=int(raw.get("n_layers", 3)),
            n_heads=int(raw.get("n_heads", 4)),
            learning_rate=float(raw.get("learning_rate", 5e-4)),
            batch_size=int(raw.get("batch_size", 8)),
            epochs=int(raw.get("epochs", 20)),
            device=str(raw.get("device", "cuda")),
            problog_log_level=str(raw.get("problog_log_level", "ERROR")),
            use_mlflow=parse_bool(raw.get("use_mlflow", True)),
            mlflow_tracking_uri=(
                None
                if raw.get("mlflow_tracking_uri") in (None, "")
                else str(raw.get("mlflow_tracking_uri"))
            ),
            mlflow_experiment_name=str(
                raw.get("mlflow_experiment_name", "deepproblog-dpl-pipeline")
            ),
            mlflow_run_name=(
                None
                if raw.get("mlflow_run_name") in (None, "")
                else str(raw.get("mlflow_run_name"))
            ),
        )

    def validate(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "train_fraction + val_fraction + test_fraction must equal 1.0"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not hasattr(logging, self.problog_log_level.upper()):
            raise ValueError(
                "problog_log_level must be a valid logging level name, such as ERROR or CRITICAL"
            )

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_path"] = str(self.data_path)
        payload["output_dir"] = str(self.output_dir)
        return payload


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


@dataclass(frozen=True)
class ExampleRow:
    example_id: int
    query_id: int
    query_text: str
    original_query: str
    atoms: tuple[str, ...]
    logical_query: str
    entity: str
    candidate_text: str
    relevance: float
    weight: float
    source: str

    @property
    def target_probability(self) -> float:
        return float(self.relevance)


@dataclass(frozen=True)
class AtomRecord:
    atom_id: int
    example_id: int
    query_id: int
    atom_index: int
    atom_text: str
    query_text: str
    candidate_text: str
    entity: str

    def model_input_segments(self) -> tuple[str, ...]:
        return (self.query_text, self.atom_text, self.candidate_text)


@dataclass(frozen=True)
class AtomNode:
    text: str


@dataclass(frozen=True)
class AndNode:
    children: tuple[object, ...]


@dataclass(frozen=True)
class OrNode:
    children: tuple[object, ...]


class AtomTensorSource(Mapping):
    def __init__(self, tensors_by_atom_id: dict[int, torch.Tensor]):
        self.tensors_by_atom_id = dict(tensors_by_atom_id)

    def __getitem__(self, key: object) -> torch.Tensor:
        if isinstance(key, tuple):
            if len(key) != 1:
                raise KeyError(f"Expected a one-element key tuple, got {key!r}")
            key = key[0]
        if hasattr(key, "value"):
            key = key.value
        return self.tensors_by_atom_id[int(key)]

    def __iter__(self):
        for atom_id in self.tensors_by_atom_id:
            yield (atom_id,)

    def __len__(self) -> int:
        return len(self.tensors_by_atom_id)


class HashingTextTensorizer:
    def __init__(self, vocab_size: int, max_length: int):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.unk_token_id = 100

    def _hash_token(self, token: str) -> int:
        if not token:
            return self.unk_token_id
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        available = max(self.vocab_size - 103, 1)
        return 103 + (value % available)

    def _tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def encode_segments(self, segments: tuple[str, ...]) -> torch.Tensor:
        token_ids = [self.cls_token_id]
        for segment in segments:
            token_ids.extend(
                self._hash_token(token) for token in self._tokenize(segment)
            )
            token_ids.append(self.sep_token_id)

        token_ids = token_ids[: self.max_length]
        attention_mask = [1] * len(token_ids)

        if len(token_ids) < self.max_length:
            padding = [self.pad_token_id] * (self.max_length - len(token_ids))
            token_ids = token_ids + padding
            attention_mask = attention_mask + [0] * len(padding)

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        return torch.stack((input_ids, attention_mask_tensor), dim=0)

    def encode_atom_record(self, atom_record: AtomRecord) -> torch.Tensor:
        return self.encode_segments(atom_record.model_input_segments())

    def build_tensor_source(self, atom_records: list[AtomRecord]) -> AtomTensorSource:
        tensors = {
            atom_record.atom_id: self.encode_atom_record(atom_record)
            for atom_record in atom_records
        }
        return AtomTensorSource(tensors)


class AtomInputDataset(TorchDataset):
    def __init__(
        self, atom_records: list[AtomRecord], tensorizer: HashingTextTensorizer
    ):
        self.atom_records = list(atom_records)
        self.tensorizer = tensorizer

    def __len__(self) -> int:
        return len(self.atom_records)

    def __getitem__(self, index: int) -> dict[str, object]:
        atom_record = self.atom_records[index]
        return {
            "packed_input": self.tensorizer.encode_atom_record(atom_record),
            "atom_text": atom_record.atom_text,
            "query_text": atom_record.query_text,
            "entity": atom_record.entity,
        }


class QuerySubset(DeepProbLogDataset):
    def __init__(
        self,
        examples: list[ExampleRow],
        indices: list[int],
        predicate_name: str = "query_relevant",
    ):
        self.examples = examples
        self.indices = list(indices)
        self.predicate_name = predicate_name

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> ExampleRow:
        return self.examples[self.indices[index]]

    def to_query(self, i: int) -> Query:
        example = self.examples[self.indices[i]]
        return Query(
            Term(
                self.predicate_name,
                Constant(example.query_id),
                Constant(example.example_id),
            ),
            p=example.target_probability,
        )


class TransformerAtomScorer(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    @classmethod
    def from_random_distilbert(
        cls,
        vocab_size: int,
        max_length: int,
        hidden_size: int,
        n_layers: int,
        n_heads: int,
    ) -> "TransformerAtomScorer":
        config = DistilBertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_length,
            sinusoidal_pos_embds=False,
            n_layers=n_layers,
            n_heads=n_heads,
            dim=hidden_size,
            hidden_dim=hidden_size * 4,
            dropout=0.1,
            attention_dropout=0.1,
        )
        return cls(encoder=DistilBertModel(config), hidden_size=config.dim)

    def forward(self, packed_inputs: torch.Tensor) -> torch.Tensor:
        if packed_inputs.dim() == 2:
            packed_inputs = packed_inputs.unsqueeze(0)
        if packed_inputs.dim() != 3 or packed_inputs.size(1) != 2:
            raise ValueError(
                f"Expected packed inputs of shape [batch, 2, seq_len], got {tuple(packed_inputs.shape)}"
            )

        input_ids = packed_inputs[:, 0, :].long()
        attention_mask = packed_inputs[:, 1, :].long()
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding)).squeeze(-1)
        positive_probability = torch.sigmoid(logits)
        negative_probability = 1.0 - positive_probability
        return torch.stack((negative_probability, positive_probability), dim=1)


def collapse_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def load_training_config(path: Path) -> TrainingConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = TrainingConfig.from_dict(raw)
    config.validate()
    return config


def configure_library_logging(problog_log_level: str) -> None:
    level_name = problog_log_level.upper()
    level_value = getattr(logging, level_name)
    for logger_name in ("problog", "deepproblog"):
        library_logger = logging.getLogger(logger_name)
        library_logger.setLevel(level_value)


@contextmanager
def log_major_operation(message: str):
    logger.info("%s", message)
    started_at = time.perf_counter()
    try:
        yield
    finally:
        logger.info(
            "Completed %s in %.2fs", message.lower(), time.perf_counter() - started_at
        )


def load_examples(path: Path) -> list[ExampleRow]:
    examples: list[ExampleRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            atoms = tuple(
                collapse_whitespace(atom)
                for atom in (raw.get("atoms") or [])
                if collapse_whitespace(atom)
            )
            logical_query = collapse_whitespace(raw.get("logical_query", ""))
            candidate_text = collapse_whitespace(raw.get("text", ""))
            query_text = collapse_whitespace(raw.get("query", ""))
            entity = collapse_whitespace(raw.get("entity", ""))

            if (
                not atoms
                or not logical_query
                or not candidate_text
                or not query_text
                or not entity
            ):
                continue

            examples.append(
                ExampleRow(
                    example_id=len(examples),
                    query_id=int(raw["id"]),
                    query_text=query_text,
                    original_query=collapse_whitespace(raw.get("original_query", "")),
                    atoms=atoms,
                    logical_query=logical_query,
                    entity=entity,
                    candidate_text=candidate_text,
                    relevance=float(raw["relevance"]),
                    weight=float(raw.get("weight", 1.0)),
                    source=str(raw.get("source", "unknown")),
                )
            )

    if not examples:
        raise ValueError("No usable examples were loaded from the JSONL file")

    return examples


def select_example_subset(
    examples: list[ExampleRow],
    limit: int | None,
    seed: int,
) -> list[ExampleRow]:
    if limit is None or limit >= len(examples):
        return examples
    if limit < 1:
        raise ValueError("max_examples must be >= 1 when provided")

    split_frame = pd.DataFrame(
        {
            "index": list(range(len(examples))),
            "target": [example.relevance for example in examples],
        }
    )

    try:
        subset_frame, _ = train_test_split(
            split_frame,
            train_size=limit,
            random_state=seed,
            shuffle=True,
            stratify=split_frame["target"],
        )
    except ValueError as exc:
        logger.warning(
            "Falling back to unstratified max_examples sampling for %d examples: %s",
            limit,
            exc,
        )
        subset_frame, _ = train_test_split(
            split_frame,
            train_size=limit,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    selected_examples = [
        examples[int(index)] for index in subset_frame["index"].tolist()
    ]
    return [
        replace(example, example_id=new_id)
        for new_id, example in enumerate(selected_examples)
    ]


def split_indices(
    examples: list[ExampleRow],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    split_frame = pd.DataFrame(
        {
            "index": list(range(len(examples))),
            "target": [example.relevance for example in examples],
        }
    )

    train_frame, holdout_frame = train_test_split(
        split_frame,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True,
        stratify=split_frame["target"],
    )
    val_share = val_fraction / (val_fraction + test_fraction)
    val_frame, test_frame = train_test_split(
        holdout_frame,
        train_size=val_share,
        random_state=seed,
        shuffle=True,
        stratify=holdout_frame["target"],
    )

    return (
        train_frame["index"].tolist(),
        val_frame["index"].tolist(),
        test_frame["index"].tolist(),
    )


def build_atom_records(
    examples: list[ExampleRow],
) -> tuple[list[AtomRecord], dict[int, list[AtomRecord]]]:
    atom_records: list[AtomRecord] = []
    atom_records_by_example: dict[int, list[AtomRecord]] = {}

    for example in examples:
        example_atom_records: list[AtomRecord] = []
        for atom_index, atom_text in enumerate(example.atoms):
            atom_record = AtomRecord(
                atom_id=len(atom_records),
                example_id=example.example_id,
                query_id=example.query_id,
                atom_index=atom_index,
                atom_text=atom_text,
                query_text=example.query_text,
                candidate_text=example.candidate_text,
                entity=example.entity,
            )
            atom_records.append(atom_record)
            example_atom_records.append(atom_record)
        atom_records_by_example[example.example_id] = example_atom_records

    return atom_records, atom_records_by_example


def strip_outer_parentheses(expression: str) -> str:
    candidate = expression.strip()
    while candidate.startswith("(") and candidate.endswith(")"):
        depth = 0
        wraps_entire_expression = True
        for index, char in enumerate(candidate):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            if depth == 0 and index != len(candidate) - 1:
                wraps_entire_expression = False
                break
        if not wraps_entire_expression:
            break
        candidate = candidate[1:-1].strip()
    return candidate


def strip_unmatched_edge_parentheses(expression: str) -> str:
    candidate = expression.strip()
    while candidate.startswith("(") and candidate.count("(") > candidate.count(")"):
        candidate = candidate[1:].strip()
    while candidate.endswith(")") and candidate.count(")") > candidate.count("("):
        candidate = candidate[:-1].strip()
    return candidate


def split_top_level(expression: str, operator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    index = 0

    while index < len(expression):
        char = expression[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and expression.startswith(operator, index):
            parts.append(expression[start:index].strip())
            index += len(operator)
            start = index
            continue
        index += 1

    parts.append(expression[start:].strip())
    return parts


def parse_formula(expression: str) -> object:
    normalized_expression = " ".join(expression.split())
    normalized_expression = strip_outer_parentheses(normalized_expression)

    or_parts = split_top_level(normalized_expression, " OR ")
    if len(or_parts) > 1:
        return OrNode(children=tuple(parse_formula(part) for part in or_parts))

    and_parts = split_top_level(normalized_expression, " AND ")
    if len(and_parts) > 1:
        return AndNode(children=tuple(parse_formula(part) for part in and_parts))

    atom_text = strip_unmatched_edge_parentheses(
        strip_outer_parentheses(normalized_expression)
    )
    if not atom_text:
        raise ValueError(f"Could not parse formula expression: {expression!r}")
    return AtomNode(text=atom_text)


def literal_parts(literal: str) -> tuple[str, bool]:
    normalized_literal = strip_unmatched_edge_parentheses(
        strip_outer_parentheses(" ".join(literal.split()))
    )
    if normalized_literal.startswith("NOT(") and normalized_literal.endswith(")"):
        atom_text = strip_unmatched_edge_parentheses(
            strip_outer_parentheses(normalized_literal[4:-1])
        )
        return atom_text, True
    return normalized_literal, False


def node_to_dnf(node: object) -> tuple[tuple[str, ...], ...]:
    if isinstance(node, AtomNode):
        return ((node.text,),)

    if isinstance(node, OrNode):
        branches: list[tuple[str, ...]] = []
        for child in node.children:
            branches.extend(node_to_dnf(child))
        return tuple(branches)

    if isinstance(node, AndNode):
        child_branches = [node_to_dnf(child) for child in node.children]
        combined_branches: list[tuple[str, ...]] = []
        for branch_product in product(*child_branches):
            merged_branch: list[str] = []
            for branch in branch_product:
                merged_branch.extend(branch)
            combined_branches.append(tuple(merged_branch))
        return tuple(combined_branches)

    raise TypeError(f"Unsupported node type: {type(node)!r}")


def formula_to_dnf(expression: str) -> tuple[tuple[str, ...], ...]:
    return node_to_dnf(parse_formula(expression))


def compile_formula_rules(
    example: ExampleRow,
    atom_predicate: str = "atom_holds",
    negative_atom_predicate: str = "atom_not_holds",
    query_predicate: str = "query_relevant",
) -> list[str]:
    atom_index = {atom_text: index for index, atom_text in enumerate(example.atoms)}
    rules: list[str] = []

    for branch in formula_to_dnf(example.logical_query):
        branch_literals: list[str] = []
        skip_branch = False
        for literal in branch:
            atom_text, is_negated = literal_parts(literal)
            if atom_text not in atom_index:
                skip_branch = True
                break

            atom_id = atom_index[atom_text]
            if is_negated:
                branch_literals.append(
                    f"{negative_atom_predicate}({example.query_id}, {example.example_id}, {atom_id})"
                )
            else:
                branch_literals.append(
                    f"{atom_predicate}({example.query_id}, {example.example_id}, {atom_id})"
                )

        if skip_branch:
            continue

        body = ", ".join(branch_literals) if branch_literals else "true"
        rules.append(
            f"{query_predicate}({example.query_id}, {example.example_id}) :- {body}."
        )

    return rules


def compile_training_program(
    examples: list[ExampleRow],
    atom_records_by_example: dict[int, list[AtomRecord]],
    indices: list[int],
    tensor_source_name: str,
    network_name: str = "atom_classifier",
    neural_predicate: str = "atom_truth",
    atom_predicate: str = "atom_holds",
    negative_atom_predicate: str = "atom_not_holds",
    query_predicate: str = "query_relevant",
) -> str:
    lines = [
        f"nn({network_name}, [tensor({tensor_source_name}(AtomId))], Label, [false, true]) :: {neural_predicate}(AtomId, Label)."
    ]

    for example_index in indices:
        example = examples[example_index]
        for atom_record in atom_records_by_example[example.example_id]:
            lines.append(
                f"{atom_predicate}({example.query_id}, {example.example_id}, {atom_record.atom_index}) :- {neural_predicate}({atom_record.atom_id}, true)."
            )
            lines.append(
                f"{negative_atom_predicate}({example.query_id}, {example.example_id}, {atom_record.atom_index}) :- {neural_predicate}({atom_record.atom_id}, false)."
            )
        lines.extend(
            compile_formula_rules(
                example,
                atom_predicate=atom_predicate,
                negative_atom_predicate=negative_atom_predicate,
                query_predicate=query_predicate,
            )
        )
        lines.append(
            f"query({query_predicate}({example.query_id}, {example.example_id}))."
        )

    return "\n".join(lines)


def as_float(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def build_training_runtime(
    atom_records: list[AtomRecord],
    program: str,
    config: TrainingConfig,
) -> dict[str, object]:
    with log_major_operation(
        f"Initializing tensorizer (vocab_size={config.vocab_size}, max_length={config.max_length})"
    ):
        tensorizer = HashingTextTensorizer(
            vocab_size=config.vocab_size,
            max_length=config.max_length,
        )

    with log_major_operation(f"Encoding {len(atom_records)} atom records into tensors"):
        tensor_source = tensorizer.build_tensor_source(atom_records)

    with log_major_operation("Initializing transformer atom scorer"):
        scorer = TransformerAtomScorer.from_random_distilbert(
            vocab_size=tensorizer.vocab_size,
            max_length=tensorizer.max_length,
            hidden_size=config.hidden_size,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
        )

    with log_major_operation("Creating optimizer and DeepProbLog network wrapper"):
        optimizer = torch.optim.Adam(scorer.parameters(), lr=config.learning_rate)
        network = Network(scorer, "atom_classifier", optimizer=optimizer, batching=True)

    with log_major_operation("Constructing DeepProbLog model from compiled program"):
        model = Model(program, [network], load=False)

    with log_major_operation(
        f"Registering tensor source '{config.tensor_source_name}'"
    ):
        model.add_tensor_source(config.tensor_source_name, tensor_source)

    with log_major_operation("Attaching ExactEngine to DeepProbLog model"):
        model.set_engine(ExactEngine(model))

    return {
        "tensorizer": tensorizer,
        "tensor_source": tensor_source,
        "scorer": scorer,
        "optimizer": optimizer,
        "network": network,
        "model": model,
    }


def count_model_parameters(module: nn.Module) -> tuple[int, int]:
    total_parameters = sum(parameter.numel() for parameter in module.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return total_parameters, trainable_parameters


def get_module_device(module: nn.Module) -> str:
    try:
        return str(next(module.parameters()).device)
    except StopIteration:
        return "no-parameters"


def evaluate_query_dataset(
    model: Model, query_dataset: QuerySubset
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    binary_cross_entropy_terms: list[float] = []
    correct_predictions = 0

    for local_index in range(len(query_dataset)):
        query = query_dataset.to_query(local_index)
        solve_result = model.solve([query])[0]
        result_term = next(iter(solve_result.result))
        prediction = as_float(solve_result.result[result_term])
        prediction = min(max(prediction, 1e-6), 1.0 - 1e-6)
        target = as_float(query.p)
        source_example = query_dataset[local_index]

        absolute_errors.append(abs(prediction - target))
        squared_errors.append((prediction - target) ** 2)
        binary_cross_entropy_terms.append(
            -(
                target * math.log(prediction)
                + (1.0 - target) * math.log(1.0 - prediction)
            )
        )
        correct_predictions += int((prediction >= 0.5) == (target >= 0.5))

        rows.append(
            {
                "example_id": source_example.example_id,
                "query_id": source_example.query_id,
                "term": str(result_term),
                "target": target,
                "prediction": prediction,
            }
        )

    return {
        "loss": sum(binary_cross_entropy_terms) / len(binary_cross_entropy_terms),
        "mae": sum(absolute_errors) / len(absolute_errors),
        "brier": sum(squared_errors) / len(squared_errors),
        "accuracy": correct_predictions / len(query_dataset),
        "rows": rows,
    }


def summarize_metrics(metrics: dict[str, object]) -> dict[str, float]:
    return {
        key: round(float(value), 4) for key, value in metrics.items() if key != "rows"
    }


def flatten_metrics(prefix: str, metrics: dict[str, object]) -> dict[str, float]:
    return {
        f"{prefix}_{key}": float(value)
        for key, value in metrics.items()
        if key != "rows"
    }


def import_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow tracking is enabled, but the mlflow package is not available in the active environment"
        ) from exc
    return mlflow


def configure_mlflow(config: TrainingConfig) -> Any | None:
    if not config.use_mlflow:
        return None
    mlflow = import_mlflow()
    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)
    return mlflow


def log_mlflow_params(mlflow: Any, config: TrainingConfig) -> None:
    mlflow.log_params(
        {
            key: ("null" if value is None else value)
            for key, value in config.to_json_dict().items()
        }
    )
    mlflow.set_tags(
        {
            "trainer": "train_dpl_pipeline",
            "reasoner": "deepproblog",
            "artifact_output_dir": str(config.output_dir),
        }
    )


def log_mlflow_epoch_metrics(mlflow: Any, epoch_record: dict[str, float]) -> None:
    epoch_step = int(epoch_record["epoch"])
    metrics = {
        key: float(value) for key, value in epoch_record.items() if key != "epoch"
    }
    mlflow.log_metrics(metrics, step=epoch_step)


def build_mlflow_summary_metrics(result: dict[str, Any]) -> dict[str, float]:
    summary_metrics: dict[str, float] = {}
    for section in ("baseline", "final"):
        section_metrics = result.get(section) or {}
        for split_name, split_metrics in section_metrics.items():
            summary_metrics.update(
                flatten_metrics(f"{section}_{split_name}", split_metrics)
            )

    summary_metrics["best_epoch"] = float(result["best"]["epoch"])
    summary_metrics["best_validation_accuracy"] = float(
        result["best"]["validation_accuracy"]
    )
    summary_metrics["split_train_count"] = float(len(result["train_indices"]))
    summary_metrics["split_validation_count"] = float(len(result["val_indices"]))
    summary_metrics["split_test_count"] = float(len(result["test_indices"]))
    for split_name in ("train", "validation", "test"):
        summary_metrics.update(
            flatten_metrics(f"best_{split_name}", result["best"][split_name])
        )

    return summary_metrics


def allocate_run_output_dir(base_output_dir: Path) -> Path:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = base_output_dir / timestamp
    suffix = 1
    while candidate.exists():
        candidate = base_output_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def save_training_curves(
    training_history: list[dict[str, float]],
    baseline_validation_metrics: dict[str, object] | None,
    output_dir: Path,
) -> dict[str, Path]:
    history_frame = pd.DataFrame(training_history)
    history_csv_path = output_dir / "training_history.csv"
    history_frame.to_csv(history_csv_path, index=False)

    epochs = history_frame["epoch"].astype(int).tolist()

    loss_plot_path = output_dir / "loss_curves.png"
    loss_figure, loss_axis = plt.subplots(figsize=(8, 4.5))
    loss_axis.plot(
        epochs,
        history_frame["train_loss"],
        marker="o",
        linewidth=2,
        label="Train batch loss",
    )
    loss_axis.plot(
        epochs,
        history_frame["train_query_loss"],
        marker="s",
        linewidth=2,
        label="Train query loss",
    )
    loss_axis.plot(
        epochs,
        history_frame["val_loss"],
        marker="^",
        linewidth=2,
        label="Validation query loss",
    )
    if baseline_validation_metrics is not None:
        loss_axis.axhline(
            float(baseline_validation_metrics["loss"]),
            linestyle="--",
            alpha=0.5,
            color="#ff7f0e",
            label="Baseline validation loss",
        )
    loss_axis.set_title("Training and Validation Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(alpha=0.3)
    loss_axis.legend()
    loss_figure.tight_layout()
    loss_figure.savefig(loss_plot_path, dpi=160)
    plt.close(loss_figure)

    validation_plot_path = output_dir / "validation_metrics.png"
    validation_figure, validation_axis = plt.subplots(figsize=(8, 4.5))
    validation_axis.plot(
        epochs,
        history_frame["train_accuracy"],
        marker="o",
        linewidth=2,
        label="Train accuracy",
    )
    validation_axis.plot(
        epochs,
        history_frame["val_accuracy"],
        marker="s",
        linewidth=2,
        label="Validation accuracy",
    )
    validation_axis.plot(
        epochs,
        history_frame["val_mae"],
        marker="^",
        linewidth=2,
        label="Validation MAE",
    )
    validation_axis.plot(
        epochs,
        history_frame["val_brier"],
        marker="d",
        linewidth=2,
        label="Validation Brier",
    )
    if baseline_validation_metrics is not None:
        validation_axis.axhline(
            float(baseline_validation_metrics["accuracy"]),
            linestyle="--",
            alpha=0.5,
            color="#1f77b4",
            label="Baseline validation accuracy",
        )
        validation_axis.axhline(
            float(baseline_validation_metrics["brier"]),
            linestyle=":",
            alpha=0.6,
            color="#ff7f0e",
            label="Baseline validation Brier",
        )
    validation_axis.set_title("Per-Epoch Validation Metrics")
    validation_axis.set_xlabel("Epoch")
    validation_axis.set_ylabel("Metric value")
    validation_axis.grid(alpha=0.3)
    validation_axis.legend()
    validation_figure.tight_layout()
    validation_figure.savefig(validation_plot_path, dpi=160)
    plt.close(validation_figure)

    return {
        "training_history_csv": history_csv_path,
        "loss_plot": loss_plot_path,
        "validation_plot": validation_plot_path,
    }


def log_mlflow_artifacts(
    mlflow: Any,
    config: TrainingConfig,
    result: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> None:
    mlflow.log_metrics(build_mlflow_summary_metrics(result))
    mlflow.log_dict(
        {"training_history": result["training_history"]},
        "metrics/training_history.json",
    )
    mlflow.log_dict(config.to_json_dict(), "config/run_config.json")

    for artifact_name, artifact_path in artifact_paths.items():
        if artifact_name.endswith("_plot"):
            artifact_subdir = "plots"
        elif artifact_name.endswith("_csv"):
            artifact_subdir = "tables"
        else:
            artifact_subdir = "outputs"
        mlflow.log_artifact(str(artifact_path), artifact_path=artifact_subdir)

    logger.info(
        "Logged MLflow run artifacts to experiment '%s'", config.mlflow_experiment_name
    )


def train_model_from_config(
    config: TrainingConfig,
    mlflow: Any | None = None,
) -> dict[str, Any]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_printoptions(edgeitems=3, linewidth=120)

    with log_major_operation(f"Loading examples from {config.data_path}"):
        examples = load_examples(config.data_path)
        examples = select_example_subset(examples, config.max_examples, config.seed)

    with log_major_operation("Splitting examples into train/validation/test"):
        train_indices, val_indices, test_indices = split_indices(
            examples,
            config.train_fraction,
            config.val_fraction,
            config.test_fraction,
            config.seed,
        )

    with log_major_operation("Expanding examples into atom records"):
        atom_records, atom_records_by_example = build_atom_records(examples)

    with log_major_operation("Creating query datasets"):
        query_train_dataset = QuerySubset(examples, train_indices)
        query_val_dataset = QuerySubset(examples, val_indices)
        query_test_dataset = QuerySubset(examples, test_indices)

    with log_major_operation("Compiling DeepProbLog training program"):
        program = compile_training_program(
            examples=examples,
            atom_records_by_example=atom_records_by_example,
            indices=train_indices + val_indices + test_indices,
            tensor_source_name=config.tensor_source_name,
        )

    with log_major_operation("Building DeepProbLog runtime"):
        runtime = build_training_runtime(
            atom_records=atom_records, program=program, config=config
        )
    scorer = runtime["scorer"]
    optimizer = runtime["optimizer"]
    model = runtime["model"]
    total_parameters, trainable_parameters = count_model_parameters(scorer)
    effective_device = get_module_device(scorer)

    with log_major_operation("Creating tensor and query data loaders"):
        atom_input_dataset = AtomInputDataset(atom_records, runtime["tensorizer"])
        atom_batch_loader = TorchDataLoader(
            atom_input_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )
        first_atom_batch = next(iter(atom_batch_loader))
        query_train_loader = DeepProbLogDataLoader(
            query_train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )

    baseline_metrics: dict[str, dict[str, object]] = {}
    if config.compute_baseline_metrics:
        logger.info("Starting baseline metric sweep")
        baseline_metrics = {}
        for split_name, dataset in (
            ("train", query_train_dataset),
            ("validation", query_val_dataset),
            ("test", query_test_dataset),
        ):
            split_started_at = time.perf_counter()
            logger.info(
                "Evaluating baseline metrics on %s split (%d queries)",
                split_name,
                len(dataset),
            )
            baseline_metrics[split_name] = evaluate_query_dataset(model, dataset)
            logger.info(
                "Finished baseline %s evaluation in %.2fs",
                split_name,
                time.perf_counter() - split_started_at,
            )

    logger.info("Loaded examples after cleaning: %d", len(examples))
    logger.info("Expanded atom records: %d", len(atom_records))
    logger.info(
        "Train/validation/test sizes: %d/%d/%d",
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )
    logger.info("Program lines: %d", len(program.splitlines()))
    logger.info(
        "Model parameters: total=%d trainable=%d",
        total_parameters,
        trainable_parameters,
    )
    logger.info(
        "Device configuration: requested=%s effective_model_device=%s cuda_available=%s",
        config.device,
        effective_device,
        torch.cuda.is_available(),
    )
    logger.info(
        "Atom batch tensor shape: %s",
        tuple(first_atom_batch["packed_input"].shape),
    )
    logger.info("Train batches per epoch: %d", len(query_train_loader))
    if baseline_metrics:
        logger.info(
            "Baseline metrics: %s",
            {
                split: summarize_metrics(metrics)
                for split, metrics in baseline_metrics.items()
            },
        )
    else:
        logger.info("Baseline metric pass skipped")

    train_object = TrainObject(model)
    loss_function = getattr(model.solver.semiring, "cross_entropy")
    training_history: list[dict[str, float]] = []
    best_val_accuracy = float("-inf")
    best_epoch = 0
    best_model_state = copy.deepcopy(scorer.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())

    for epoch_index in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        logger.info("Starting epoch %d/%d", epoch_index, config.epochs)
        train_object.model.optimizer.step_epoch()
        train_object.timing = [0.0, 0.0, 0.0]
        epoch_loss_total = 0.0

        batch_training_start = time.perf_counter()
        for batch in query_train_loader:
            train_object.i += 1
            train_object.model.train()
            train_object.model.optimizer.zero_grad()
            batch_loss = train_object.get_loss(batch, loss_function)
            epoch_loss_total += as_float(batch_loss)
            train_object.model.optimizer.step()
        batch_training_duration = time.perf_counter() - batch_training_start

        train_eval_start = time.perf_counter()
        train_metrics = evaluate_query_dataset(model, query_train_dataset)
        train_eval_duration = time.perf_counter() - train_eval_start

        val_eval_start = time.perf_counter()
        val_metrics = evaluate_query_dataset(model, query_val_dataset)
        val_eval_duration = time.perf_counter() - val_eval_start
        epoch_duration = time.perf_counter() - epoch_start

        epoch_record = {
            "epoch": float(epoch_index),
            "train_loss": epoch_loss_total / len(query_train_loader),
            "train_query_loss": float(train_metrics["loss"]),
            "train_mae": float(train_metrics["mae"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_loss": float(val_metrics["loss"]),
            "val_mae": float(val_metrics["mae"]),
            "val_brier": float(val_metrics["brier"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "duration_seconds": epoch_duration,
            "ground_time": float(train_object.timing[0]),
            "compile_time": float(train_object.timing[1]),
            "eval_time": float(train_object.timing[2]),
            "batch_training_seconds": batch_training_duration,
            "train_eval_seconds": train_eval_duration,
            "val_eval_seconds": val_eval_duration,
        }
        training_history.append(epoch_record)
        if epoch_record["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = epoch_record["val_accuracy"]
            best_epoch = epoch_index
            best_model_state = copy.deepcopy(scorer.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        if mlflow is not None:
            log_mlflow_epoch_metrics(mlflow, epoch_record)
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_mae=%.4f | val_accuracy=%.4f | epoch_seconds=%.1f | batch_train=%.1fs | train_eval=%.1fs | val_eval=%.1fs",
            epoch_index,
            config.epochs,
            epoch_record["train_loss"],
            epoch_record["val_loss"],
            epoch_record["val_mae"],
            epoch_record["val_accuracy"],
            epoch_record["duration_seconds"],
            epoch_record["batch_training_seconds"],
            epoch_record["train_eval_seconds"],
            epoch_record["val_eval_seconds"],
        )

    logger.info("Evaluating final metrics with last-epoch weights")
    final_metrics_start = time.perf_counter()
    final_train_metrics = evaluate_query_dataset(model, query_train_dataset)
    final_val_metrics = evaluate_query_dataset(model, query_val_dataset)
    final_test_metrics = evaluate_query_dataset(model, query_test_dataset)
    logger.info(
        "Finished final metric evaluation in %.2fs",
        time.perf_counter() - final_metrics_start,
    )

    logger.info("Restoring best checkpoint from epoch %d", best_epoch)
    scorer.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)

    logger.info("Evaluating metrics with best checkpoint weights")
    best_metrics_start = time.perf_counter()
    best_train_metrics = evaluate_query_dataset(model, query_train_dataset)
    best_val_metrics = evaluate_query_dataset(model, query_val_dataset)
    best_test_metrics = evaluate_query_dataset(model, query_test_dataset)
    logger.info(
        "Finished best-checkpoint metric evaluation in %.2fs",
        time.perf_counter() - best_metrics_start,
    )

    return {
        "examples": examples,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "program": program,
        "runtime": runtime,
        "training_history": training_history,
        "baseline": baseline_metrics,
        "final": {
            "train": final_train_metrics,
            "validation": final_val_metrics,
            "test": final_test_metrics,
        },
        "best": {
            "epoch": best_epoch,
            "validation_accuracy": best_val_accuracy,
            "train": best_train_metrics,
            "validation": best_val_metrics,
            "test": best_test_metrics,
        },
    }


def save_artifacts(
    config: TrainingConfig,
    result: dict[str, Any],
) -> dict[str, Path]:
    output_dir = allocate_run_output_dir(config.output_dir)
    logger.info("Saving run artifacts under %s", output_dir)

    checkpoint_path = output_dir / "training_checkpoint.pt"
    weights_path = output_dir / "atom_scorer_weights.pt"
    metrics_path = output_dir / "training_metrics.json"
    config_copy_path = output_dir / "config.json"
    program_path = output_dir / "training_program.pl"
    bundle_manifest_path = output_dir / "dpl_pipeline_bundle.json"
    plot_paths = save_training_curves(
        training_history=result["training_history"],
        baseline_validation_metrics=result["baseline"].get("validation"),
        output_dir=output_dir,
    )

    scorer = result["runtime"]["scorer"]
    optimizer = result["runtime"]["optimizer"]

    torch.save(
        {
            "config": config.to_json_dict(),
            "model_state_dict": scorer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_history": result["training_history"],
        },
        checkpoint_path,
    )
    torch.save(scorer.state_dict(), weights_path)

    metrics_payload = {
        "config": config.to_json_dict(),
        "artifact_output_dir": str(output_dir),
        "selected_checkpoint": {
            "epoch": result["best"]["epoch"],
            "validation_accuracy": result["best"]["validation_accuracy"],
        },
        "split_sizes": {
            "train": len(result["train_indices"]),
            "validation": len(result["val_indices"]),
            "test": len(result["test_indices"]),
        },
        "baseline": {
            split: {key: value for key, value in metrics.items() if key != "rows"}
            for split, metrics in result["baseline"].items()
        },
        "final": {
            split: {key: value for key, value in metrics.items() if key != "rows"}
            for split, metrics in result["final"].items()
        },
        "best": {
            "epoch": result["best"]["epoch"],
            "validation_accuracy": result["best"]["validation_accuracy"],
            "train": {
                key: value
                for key, value in result["best"]["train"].items()
                if key != "rows"
            },
            "validation": {
                key: value
                for key, value in result["best"]["validation"].items()
                if key != "rows"
            },
            "test": {
                key: value
                for key, value in result["best"]["test"].items()
                if key != "rows"
            },
        },
        "training_history": result["training_history"],
        "program_preview": "\n".join(result["program"].splitlines()[:12]),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    config_copy_path.write_text(
        json.dumps(config.to_json_dict(), indent=2),
        encoding="utf-8",
    )
    program_path.write_text(result["program"], encoding="utf-8")
    bundle_manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "dpl_pipeline_bundle",
                "estimator_type": "DPLPipeline",
                "config_path": str(config_copy_path),
                "weights_path": str(weights_path),
                "checkpoint_path": str(checkpoint_path),
                "program_path": str(program_path),
                "metrics_path": str(metrics_path),
                "best_epoch": result["best"]["epoch"],
                "best_validation_accuracy": result["best"]["validation_accuracy"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "checkpoint": checkpoint_path,
        "weights": weights_path,
        "metrics": metrics_path,
        "config": config_copy_path,
        "program": program_path,
        "bundle_manifest": bundle_manifest_path,
        **plot_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the minimal DeepProbLog pipeline from a JSON config"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/dpl_pipeline_train_config.json"),
        help="JSON config path",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the pre-training baseline metric sweep over train/validation/test",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with log_major_operation(f"Loading training config from {args.config}"):
        config = load_training_config(args.config)
    if args.skip_baseline:
        config.compute_baseline_metrics = False
    with log_major_operation("Configuring library logging"):
        configure_library_logging(config.problog_log_level)
    with log_major_operation("Configuring MLflow"):
        mlflow = configure_mlflow(config)
    run_context = (
        mlflow.start_run(run_name=config.mlflow_run_name)
        if mlflow is not None
        else nullcontext()
    )
    with run_context:
        if mlflow is not None:
            with log_major_operation("Logging MLflow run parameters"):
                log_mlflow_params(mlflow, config)
        with log_major_operation("Training DeepProbLog pipeline"):
            result = train_model_from_config(config, mlflow=mlflow)
        with log_major_operation(f"Saving artifacts to {config.output_dir}"):
            artifact_paths = save_artifacts(config, result)
        if mlflow is not None:
            with log_major_operation("Logging MLflow artifacts"):
                log_mlflow_artifacts(mlflow, config, result, artifact_paths)

    logger.info("Saved checkpoint to %s", artifact_paths["checkpoint"])
    logger.info("Saved weights to %s", artifact_paths["weights"])
    logger.info("Saved metrics to %s", artifact_paths["metrics"])


if __name__ == "__main__":
    main()
