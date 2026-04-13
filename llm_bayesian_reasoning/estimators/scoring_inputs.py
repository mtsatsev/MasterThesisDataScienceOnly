from dataclasses import dataclass

from llm_bayesian_reasoning.problog_models.problog_models import ProblogAtom


@dataclass(frozen=True)
class PositiveScoringInput:
    atom: ProblogAtom


@dataclass(frozen=True)
class ContrastiveScoringInput:
    atom: ProblogAtom
    negated_atom: ProblogAtom


ScoringInput = PositiveScoringInput | ContrastiveScoringInput


def _combine_context(
    atom_context: str | None,
    document_text: str | None,
    separator: str = "\n\n",
) -> str | None:
    normalized_document_text = document_text.strip() if document_text else None
    if not normalized_document_text:
        return atom_context
    if atom_context is None:
        return normalized_document_text
    return atom_context + separator + normalized_document_text


def _clone_atom_with_context(
    atom: ProblogAtom,
    document_text: str | None,
) -> ProblogAtom:
    return ProblogAtom(
        atom=atom.atom,
        probability=atom.probability,
        context=_combine_context(atom.context, document_text),
    )


def clone_scoring_inputs_with_document_context(
    scoring_inputs: list[ScoringInput],
    document_text: str | None,
) -> list[ScoringInput]:
    cloned_inputs: list[ScoringInput] = []
    for scoring_input in scoring_inputs:
        if isinstance(scoring_input, ContrastiveScoringInput):
            cloned_inputs.append(
                ContrastiveScoringInput(
                    atom=_clone_atom_with_context(scoring_input.atom, document_text),
                    negated_atom=_clone_atom_with_context(
                        scoring_input.negated_atom,
                        document_text,
                    ),
                )
            )
        else:
            cloned_inputs.append(
                PositiveScoringInput(
                    atom=_clone_atom_with_context(scoring_input.atom, document_text),
                )
            )
    return cloned_inputs