import json
import logging
import re
from hashlib import blake2b
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from transformers import DistilBertConfig, DistilBertModel, PreTrainedTokenizerBase

from llm_bayesian_reasoning.estimators.base import BaseEstimator
from llm_bayesian_reasoning.problog_models.problog_models import ProblogAtom

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)
QUERY_PREFIX = "Query: "


class DPLPipelineBundleConfig(BaseModel):
    tensor_source_name: str
    max_length: int
    vocab_size: int
    hidden_size: int
    n_layers: int
    n_heads: int

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DPLPipelineBundleConfig":
        return cls(
            tensor_source_name=str(payload.get("tensor_source_name", "atom_inputs")),
            max_length=int(payload.get("max_length", 256)),
            vocab_size=int(payload.get("vocab_size", 32768)),
            hidden_size=int(payload.get("hidden_size", 32)),
            n_layers=int(payload.get("n_layers", 3)),
            n_heads=int(payload.get("n_heads", 4)),
        )


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
        input_ids = packed_inputs[:, 0, :].long()
        attention_mask = packed_inputs[:, 1, :].long()
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding)).squeeze(-1)
        positive_probability = torch.sigmoid(logits)
        negative_probability = 1.0 - positive_probability
        return torch.stack((negative_probability, positive_probability), dim=1)


class DPLPipelineEstimator(BaseEstimator):
    def __init__(
        self,
        model: TransformerAtomScorer,
        tensorizer: HashingTextTensorizer,
        model_config: DPLPipelineBundleConfig,
        device: str = "cuda",
    ):
        super().__init__(
            model=model,
            tokenizer=cast(PreTrainedTokenizerBase, None),
            device=device,
        )
        self.tensorizer = tensorizer
        self.model_config = model_config
        self.model.to(device)
        self.model.eval()
        self._warned_missing_query = False

    @classmethod
    def from_model_bundle(
        cls,
        model_dir: str | Path,
        device: str = "cuda",
    ) -> "DPLPipelineEstimator":
        model_path = Path(model_dir)
        config_path = model_path / "config.json"
        weights_path = model_path / "atom_scorer_weights.pt"
        checkpoint_path = model_path / "training_checkpoint.pt"

        payload = json.loads(config_path.read_text(encoding="utf-8"))
        model_config = DPLPipelineBundleConfig.from_dict(payload)
        tensorizer = HashingTextTensorizer(
            vocab_size=model_config.vocab_size,
            max_length=model_config.max_length,
        )
        model = TransformerAtomScorer.from_random_distilbert(
            vocab_size=model_config.vocab_size,
            max_length=model_config.max_length,
            hidden_size=model_config.hidden_size,
            n_layers=model_config.n_layers,
            n_heads=model_config.n_heads,
        )

        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint["model_state_dict"]

        model.load_state_dict(state_dict)
        return cls(
            model=model,
            tensorizer=tensorizer,
            model_config=model_config,
            device=device,
        )

    def _extract_query_and_document(self, atom: ProblogAtom) -> tuple[str, str]:
        context = (atom.context or "").strip()
        if not context:
            return "", ""
        if context.startswith(QUERY_PREFIX):
            payload = context.removeprefix(QUERY_PREFIX)
            query_text, separator, document_text = payload.partition("\n\n")
            return query_text.strip(), document_text.strip() if separator else ""
        return "", context

    @torch.inference_mode()
    def score_probability(
        self,
        predicates: list[ProblogAtom] | list[tuple[ProblogAtom, ProblogAtom]],
        entity: str,
    ) -> list[ProblogAtom]:
        if predicates and isinstance(predicates[0], tuple):
            raise ValueError(
                "DPLPipelineEstimator does not support contrastive predicate pairs"
            )

        atoms = cast(list[ProblogAtom], predicates)
        if not atoms:
            return []

        encoded_inputs: list[torch.Tensor] = []
        retained_atoms: list[ProblogAtom] = []
        for atom in atoms:
            query_text, document_text = self._extract_query_and_document(atom)
            if not query_text and not self._warned_missing_query:
                logger.warning(
                    "DPLPipelineEstimator did not receive query context; falling back to empty query text."
                )
                self._warned_missing_query = True
            encoded_inputs.append(
                self.tensorizer.encode_segments((query_text, atom.atom, document_text))
            )
            retained_atoms.append(atom)

        packed_inputs = torch.stack(encoded_inputs, dim=0).to(self.device)
        probabilities = self.model(packed_inputs)[:, 1].detach().cpu().tolist()
        return [
            ProblogAtom(
                atom=atom.atom, probability=float(probability), context=atom.context
            )
            for atom, probability in zip(retained_atoms, probabilities, strict=True)
        ]
