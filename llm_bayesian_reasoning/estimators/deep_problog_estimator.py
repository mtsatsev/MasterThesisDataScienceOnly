from pathlib import Path
from typing import cast

from llm_bayesian_reasoning.estimators.base import BaseEstimator
from llm_bayesian_reasoning.problog_models.problog_models import ProblogAtom
from llm_bayesian_reasoning.training.deepproblog_module import (
    DeepProbLogModelConfig,
    load_model_bundle,
    score_atoms,
)


class DeepProbLogEstimator(BaseEstimator):
    def __init__(
        self,
        model,
        tokenizer,
        model_config: DeepProbLogModelConfig,
        device: str = "cuda",
    ):
        super().__init__(model=model, tokenizer=tokenizer, device=device)
        self.model_config = model_config

    @classmethod
    def from_model_bundle(
        cls,
        model_dir: str | Path,
        device: str = "cuda",
    ) -> "DeepProbLogEstimator":
        model, tokenizer, model_config, _metadata = load_model_bundle(model_dir, device)
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            device=device,
        )

    def score_probability(
        self,
        predicates: list[ProblogAtom] | list[tuple[ProblogAtom, ProblogAtom]],
        entity: str,
    ) -> list[ProblogAtom]:
        if predicates and isinstance(predicates[0], tuple):
            raise ValueError(
                "DeepProbLogEstimator does not support contrastive predicate pairs"
            )

        atoms = cast(list[ProblogAtom], predicates)
        if not atoms:
            return []

        probabilities = score_atoms(
            model=self.model,
            tokenizer=self.tokenizer,
            model_config=self.model_config,
            device=self.device,
            atoms=[atom.atom for atom in atoms],
            entity=entity,
            document_texts=[atom.context or "" for atom in atoms],
        )
        return [
            ProblogAtom(
                atom=atom.atom,
                probability=probability,
                context=atom.context,
            )
            for atom, probability in zip(atoms, probabilities, strict=True)
        ]
