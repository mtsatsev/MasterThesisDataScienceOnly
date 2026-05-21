from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from deepproblog.dataset import DataLoader as DeepProbLogDataLoader
from deepproblog.dataset import Dataset as DeepProbLogDataset
from deepproblog.engines import ExactEngine
from deepproblog.model import Model as DeepProbLogModel
from deepproblog.network import Network
from deepproblog.query import Query
from deepproblog.train import train_model
from problog.logic import Constant, Term
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from llm_bayesian_reasoning.data.deepproblog_dataset import (
    AtomClassificationDataset,
    AtomClassificationExample,
    DeepProbLogGroupedExample,
    flatten_atom_supervision_examples,
)
from llm_bayesian_reasoning.problog_models.problog_models import ProblogFormula

logger = logging.getLogger(__name__)

DEFAULT_ENCODER_MODEL_NAME = "prajjwal1/bert-tiny"
DEFAULT_ENTITY_CONSTANT = 0
CONFIG_FILENAME = "config.json"
MODEL_FILENAME = "classifier.pt"
TOKENIZER_DIRNAME = "tokenizer"


def build_atom_classifier_text(atom: str, entity: str, document_text: str) -> str:
    return f"Entity: {entity}\nAtom: {atom}\nDocument: {document_text.strip()}"


class DeepProbLogModelConfig(BaseModel):
    model_name: str = Field(default=DEFAULT_ENCODER_MODEL_NAME, min_length=1)
    max_length: int = Field(default=256, ge=16)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


@dataclass(slots=True)
class StageOneTrainingSummary:
    losses: list[float]


@dataclass(slots=True)
class DeepProbLogTrainingSummary:
    losses: list[float]
    program: str


class AtomClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_ENCODER_MODEL_NAME,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = (input_ids != 0).long()
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled))
        return torch.softmax(logits, dim=1)

    def predict_true_probability(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids)[:, 1]


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    return tokenizer


def tokenize_atom_inputs(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int,
) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return encoded["input_ids"]


def _collate_stage_one_batch(
    batch: list[AtomClassificationExample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, torch.Tensor]:
    texts = [
        build_atom_classifier_text(
            atom=example.atom,
            entity=example.entity,
            document_text=example.text,
        )
        for example in batch
    ]
    input_ids = tokenize_atom_inputs(tokenizer, texts, max_length=max_length)
    targets = torch.tensor([example.target for example in batch], dtype=torch.float32)
    weights = torch.tensor([example.weight for example in batch], dtype=torch.float32)
    return {
        "input_ids": input_ids,
        "targets": targets,
        "weights": weights,
    }


def train_atom_classifier(
    grouped_examples: list[DeepProbLogGroupedExample],
    model: AtomClassifier,
    tokenizer: PreTrainedTokenizerBase,
    model_config: DeepProbLogModelConfig,
    device: str,
    batch_size: int = 8,
    epochs: int = 1,
    learning_rate: float = 2e-5,
) -> StageOneTrainingSummary:
    examples = flatten_atom_supervision_examples(grouped_examples)
    dataset = AtomClassificationDataset(examples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate_stage_one_batch(
            batch,
            tokenizer=tokenizer,
            max_length=model_config.max_length,
        ),
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.to(device)
    losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            weights = batch["weights"].to(device)

            optimizer.zero_grad()
            probabilities = model.predict_true_probability(input_ids)
            loss = nn.functional.binary_cross_entropy(
                probabilities,
                targets,
                weight=weights,
            )
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            batch_count += 1

        epoch_loss = running_loss / max(batch_count, 1)
        logger.info("Stage-1 epoch %d loss %.4f", epoch + 1, epoch_loss)
        losses.append(epoch_loss)

    return StageOneTrainingSummary(losses=losses)


class _DeepProbLogTensorSource(dict[tuple[Constant], torch.Tensor]):
    pass


class _DeepProbLogEntityDataset(DeepProbLogDataset):
    def __init__(self, query_terms: list[Query]):
        self.query_terms = query_terms

    def __len__(self) -> int:
        return len(self.query_terms)

    def to_query(self, i: int) -> Query:
        return self.query_terms[i]


def _feature_key(feature_id: int) -> tuple[Constant]:
    return (Constant(feature_id),)


def build_feature_tensor_source(
    grouped_examples: list[DeepProbLogGroupedExample],
    tokenizer: PreTrainedTokenizerBase,
    model_config: DeepProbLogModelConfig,
    device: str,
) -> tuple[_DeepProbLogTensorSource, dict[tuple[int, int, int], int]]:
    flattened = flatten_atom_supervision_examples(grouped_examples)
    texts = [
        build_atom_classifier_text(
            atom=example.atom,
            entity=example.entity,
            document_text=example.text,
        )
        for example in flattened
    ]
    input_ids = tokenize_atom_inputs(tokenizer, texts, model_config.max_length).to(
        device
    )

    tensor_source = _DeepProbLogTensorSource()
    feature_map: dict[tuple[int, int, int], int] = {}
    for example, tensor in zip(flattened, input_ids, strict=True):
        tensor_source[_feature_key(example.feature_id)] = tensor
        feature_map[(example.query_id, example.candidate_id, example.atom_index)] = (
            example.feature_id
        )

    return tensor_source, feature_map


def _render_formula_body(
    formula: ProblogFormula,
    atom_lookup: dict[str, str],
) -> str:
    return formula.render_formula_body(
        atom_lookup=atom_lookup,
        entity_expr=str(DEFAULT_ENTITY_CONSTANT),
    )


def build_deepproblog_program(
    grouped_examples: list[DeepProbLogGroupedExample],
    feature_map: dict[tuple[int, int, int], int],
) -> tuple[str, list[Query]]:
    lines = [
        "nn(atom_classifier, [Input], Label, [false, true]) :: atom_truth(Input, Label)."
    ]
    queries: list[Query] = []

    for query_index, grouped_example in enumerate(grouped_examples):
        for candidate_index, candidate in enumerate(grouped_example.candidates):
            atom_lookup: dict[str, str] = {}
            for atom_index, atom in enumerate(grouped_example.atoms):
                predicate_name = f"q{query_index}_c{candidate_index}_a{atom_index}"
                atom_lookup[atom] = predicate_name
                atom_lookup[atom.replace("{x}", "{X}")] = predicate_name
                feature_id = feature_map[(query_index, candidate_index, atom_index)]
                lines.append(
                    f"{predicate_name}({DEFAULT_ENTITY_CONSTANT}) :- "
                    f"atom_truth(tensor(features({feature_id})), true)."
                )

            head = f"q{query_index}_c{candidate_index}_formula"
            formula = ProblogFormula(
                formula=grouped_example.logical_query.replace("{x}", "{X}"),
                head=head,
            )
            formula_body = _render_formula_body(formula, atom_lookup)
            lines.append(f"{head}({DEFAULT_ENTITY_CONSTANT}) :-")
            lines.append(f"    {formula_body}.")
            queries.append(
                Query(
                    Term(head, Constant(DEFAULT_ENTITY_CONSTANT)),
                    p=float(candidate.relevance),
                )
            )

    return "\n".join(lines), queries


def run_deepproblog_training(
    grouped_examples: list[DeepProbLogGroupedExample],
    model: AtomClassifier,
    tokenizer: PreTrainedTokenizerBase,
    model_config: DeepProbLogModelConfig,
    device: str,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    epochs: int = 1,
) -> DeepProbLogTrainingSummary:
    model.to(device)
    tensor_source, feature_map = build_feature_tensor_source(
        grouped_examples,
        tokenizer,
        model_config,
        device=device,
    )
    program, queries = build_deepproblog_program(grouped_examples, feature_map)
    dataset = _DeepProbLogEntityDataset(queries)

    network = Network(
        model,
        "atom_classifier",
        optimizer=AdamW(model.parameters(), lr=learning_rate),
        batching=True,
    )
    if device.startswith("cuda"):
        network.cuda(device=device)

    deepproblog_model = DeepProbLogModel(program, [network], load=False)
    deepproblog_model.add_tensor_source("features", tensor_source)
    deepproblog_model.set_engine(ExactEngine(deepproblog_model), cache=True)

    loader = DeepProbLogDataLoader(
        dataset, batch_size=batch_size, shuffle=True, seed=13
    )
    train_object = train_model(
        deepproblog_model,
        loader,
        epochs,
        verbose=0,
        initial_test=False,
    )
    losses = [value[1] for value in train_object.logger.log_data.get("loss", [])]
    return DeepProbLogTrainingSummary(losses=losses, program=program)


def save_model_bundle(
    output_dir: str | Path,
    model: AtomClassifier,
    tokenizer: PreTrainedTokenizerBase,
    model_config: DeepProbLogModelConfig,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path / MODEL_FILENAME)
    tokenizer.save_pretrained(output_path / TOKENIZER_DIRNAME)
    with (output_path / CONFIG_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **model_config.model_dump(),
                "metadata": metadata or {},
            },
            handle,
            indent=2,
        )
    return output_path


def load_model_bundle(
    model_dir: str | Path,
    device: str,
) -> tuple[
    AtomClassifier, PreTrainedTokenizerBase, DeepProbLogModelConfig, dict[str, Any]
]:
    model_path = Path(model_dir)
    with (model_path / CONFIG_FILENAME).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    metadata = payload.pop("metadata", {})
    model_config = DeepProbLogModelConfig.model_validate(payload)
    tokenizer = AutoTokenizer.from_pretrained(model_path / TOKENIZER_DIRNAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

    model = AtomClassifier(
        model_name=model_config.model_name,
        dropout=model_config.dropout,
    )
    state_dict = torch.load(model_path / MODEL_FILENAME, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer, model_config, metadata


@torch.inference_mode()
def score_atoms(
    model: AtomClassifier,
    tokenizer: PreTrainedTokenizerBase,
    model_config: DeepProbLogModelConfig,
    device: str,
    atoms: list[str],
    entity: str,
    document_texts: list[str],
) -> list[float]:
    texts = [
        build_atom_classifier_text(
            atom=atom,
            entity=entity,
            document_text=document_text,
        )
        for atom, document_text in zip(atoms, document_texts, strict=True)
    ]
    input_ids = tokenize_atom_inputs(tokenizer, texts, model_config.max_length).to(
        device
    )
    probabilities = model.predict_true_probability(input_ids)
    return [float(probability) for probability in probabilities.cpu().tolist()]
