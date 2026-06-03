import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol

import pandas as pd
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from llm_bayesian_reasoning.data.deepproblog_dataset import (
    DeepProbLogRow,
    flatten_atom_supervision_examples,
    group_deepproblog_rows,
    read_deepproblog_rows,
)

logger = logging.getLogger("train_dpl_atom_only")


class TensorizerProtocol(Protocol):
    def encode_segments(self, segments: tuple[str, ...]) -> torch.Tensor: ...


def _load_dpl_training_module(repo_root: Path) -> ModuleType:
    script_path = repo_root / "scripts" / "train_dpl_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "train_dpl_pipeline_module",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training helpers from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AtomBatchExample(BaseModel):
    query: str = Field(min_length=1)
    entity: str = Field(min_length=1)
    text: str = Field(default="")
    atom: str = Field(min_length=1)
    target: float = Field(ge=0.0, le=1.0)
    binary_target: int = Field(ge=0, le=1)
    weight: float = Field(ge=0.0)

    model_config = ConfigDict(extra="forbid", frozen=True)


class AtomOnlyDataset(Dataset):
    def __init__(
        self,
        examples: list[AtomBatchExample],
        tensorizer: TensorizerProtocol,
    ):
        self.examples = examples
        self.tensorizer = tensorizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        packed_input = self.tensorizer.encode_segments(
            (example.query, example.atom, example.text)
        )
        return {
            "packed_input": packed_input,
            "target": torch.tensor(example.target, dtype=torch.float32),
            "binary_target": torch.tensor(example.binary_target, dtype=torch.long),
            "weight": torch.tensor(example.weight, dtype=torch.float32),
        }


def _select_row_subset(
    rows: list[DeepProbLogRow],
    limit: int | None,
    seed: int,
) -> list[DeepProbLogRow]:
    if limit is None or limit >= len(rows):
        return rows
    if limit < 1:
        raise ValueError("max_rows must be >= 1 when provided")

    row_indices = list(range(len(rows)))
    row_targets = [row.relevance for row in rows]
    selected_indices, _ = train_test_split(
        row_indices,
        train_size=limit,
        random_state=seed,
        shuffle=True,
        stratify=row_targets,
    )
    return [rows[index] for index in selected_indices]


def _split_atom_examples(
    examples: list[AtomBatchExample],
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[list[AtomBatchExample], list[AtomBatchExample], list[AtomBatchExample]]:
    if not abs((train_fraction + val_fraction + test_fraction) - 1.0) < 1e-9:
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1")

    indices = list(range(len(examples)))
    targets = [example.binary_target for example in examples]

    train_indices, holdout_indices = train_test_split(
        indices,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True,
        stratify=targets,
    )
    holdout_targets = [targets[index] for index in holdout_indices]
    val_share = val_fraction / (val_fraction + test_fraction)
    val_indices, test_indices = train_test_split(
        holdout_indices,
        train_size=val_share,
        random_state=seed,
        shuffle=True,
        stratify=holdout_targets,
    )

    return (
        [examples[index] for index in train_indices],
        [examples[index] for index in val_indices],
        [examples[index] for index in test_indices],
    )


def _make_metrics(
    losses: list[float],
    predictions: list[float],
    binary_targets: list[int],
    soft_targets: list[float],
) -> dict[str, float]:
    hard_predictions = [int(value >= 0.5) for value in predictions]
    metrics = {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "accuracy": float(accuracy_score(binary_targets, hard_predictions)),
        "f1": float(f1_score(binary_targets, hard_predictions, zero_division=0)),
        "target_mean": float(sum(soft_targets) / max(len(soft_targets), 1)),
        "prediction_mean": float(sum(predictions) / max(len(predictions), 1)),
    }
    unique_targets = set(binary_targets)
    if len(unique_targets) > 1:
        metrics["roc_auc"] = float(roc_auc_score(binary_targets, predictions))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def run_epoch(
    loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    training: bool,
) -> dict[str, float]:
    model.train(training)
    predictions: list[float] = []
    binary_targets: list[int] = []
    soft_targets: list[float] = []
    losses: list[float] = []

    for batch in loader:
        packed_input = batch["packed_input"].to(device)
        target = batch["target"].to(device)
        weight = batch["weight"].to(device)

        if training:
            optimizer.zero_grad()

        probability = model(packed_input)[:, 1]
        loss = nn.functional.binary_cross_entropy(probability, target, weight=weight)

        if training:
            loss.backward()
            optimizer.step()

        predictions.extend(float(value) for value in probability.detach().cpu().tolist())
        binary_targets.extend(int(value) for value in batch["binary_target"].tolist())
        soft_targets.extend(float(value) for value in batch["target"].tolist())
        losses.append(float(loss.detach().cpu()))

    return _make_metrics(losses, predictions, binary_targets, soft_targets)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the hashed DPL atom scorer only and save a pipeline-compatible bundle"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("llm_bayesian_reasoning/data/preprocessed_data/dpppl.jsonl"),
        help="DeepProbLog-style JSONL with weak atom supervision fields",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    torch.manual_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    dpl_module = _load_dpl_training_module(repo_root)

    rows = read_deepproblog_rows(args.data_path)
    rows = _select_row_subset(rows, args.max_rows, args.seed)
    grouped_examples = group_deepproblog_rows(rows)
    flat_examples_raw = flatten_atom_supervision_examples(grouped_examples)
    flat_examples = [
        AtomBatchExample(
            query=example.query,
            entity=example.entity,
            text=example.text,
            atom=example.atom,
            target=example.target,
            binary_target=int(example.target >= 0.5),
            weight=example.weight,
        )
        for example in flat_examples_raw
    ]

    train_examples, val_examples, test_examples = _split_atom_examples(
        flat_examples,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    tensorizer = dpl_module.HashingTextTensorizer(
        vocab_size=args.vocab_size,
        max_length=args.max_length,
    )
    train_dataset = AtomOnlyDataset(train_examples, tensorizer)
    val_dataset = AtomOnlyDataset(val_examples, tensorizer)
    test_dataset = AtomOnlyDataset(test_examples, tensorizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    requested_device = args.device
    device = requested_device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        device = "cpu"

    model = dpl_module.TransformerAtomScorer.from_random_distilbert(
        vocab_size=args.vocab_size,
        max_length=args.max_length,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    total_parameters, trainable_parameters = dpl_module.count_model_parameters(model)
    logger.info(
        "Loaded %d rows, %d grouped examples, %d atom examples",
        len(rows),
        len(grouped_examples),
        len(flat_examples),
    )
    logger.info(
        "Atom split sizes train/validation/test: %d/%d/%d",
        len(train_examples),
        len(val_examples),
        len(test_examples),
    )
    logger.info(
        "Model parameters: total=%d trainable=%d",
        total_parameters,
        trainable_parameters,
    )
    logger.info("Device configuration: requested=%s effective=%s", requested_device, device)

    history: list[dict[str, float]] = []
    best_state = None
    best_val_f1 = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(train_loader, model, optimizer, device, training=True)
        with torch.no_grad():
            val_metrics = run_epoch(val_loader, model, optimizer, device, training=False)

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_roc_auc": train_metrics["roc_auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(epoch_record)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        logger.info("Epoch %d/%d metrics: %s", epoch, args.epochs, epoch_record)

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        test_metrics = run_epoch(test_loader, model, optimizer, device, training=False)

    output_dir = dpl_module.allocate_run_output_dir(args.output_dir)
    weights_path = output_dir / "atom_scorer_weights.pt"
    checkpoint_path = output_dir / "training_checkpoint.pt"
    config_path = output_dir / "config.json"
    metrics_path = output_dir / "training_metrics.json"
    manifest_path = output_dir / "dpl_pipeline_bundle.json"
    history_path = output_dir / "training_history.csv"

    config_payload = {
        "tensor_source_name": "atom_inputs",
        "max_length": args.max_length,
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "metadata": {
            "artifact_type": "dpl_pipeline_bundle",
            "training_mode": "atom_only",
            "data_path": str(args.data_path),
            "max_rows": args.max_rows,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "requested_device": requested_device,
            "effective_device": device,
        },
    }

    torch.save(model.state_dict(), weights_path)
    torch.save(
        {
            "config": config_payload,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_history": history,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "split_sizes": {
                    "train": len(train_examples),
                    "validation": len(val_examples),
                    "test": len(test_examples),
                },
                "best_validation_f1": best_val_f1,
                "test_metrics": test_metrics,
                "training_history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "dpl_pipeline_bundle",
                "estimator_type": "DPLPipeline",
                "training_mode": "atom_only",
                "config_path": str(config_path),
                "weights_path": str(weights_path),
                "checkpoint_path": str(checkpoint_path),
                "metrics_path": str(metrics_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(history).to_csv(history_path, index=False)

    logger.info("Saved atom-only bundle to %s", output_dir)
    logger.info("Weights: %s", weights_path)
    logger.info("Config: %s", config_path)
    logger.info("Metrics: %s", metrics_path)
    logger.info("Test metrics: %s", test_metrics)


if __name__ == "__main__":
    main()