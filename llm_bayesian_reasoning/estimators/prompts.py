from llm_bayesian_reasoning.estimators.atom_estimator import Candidate


def fill_entity(template: str, candidate: Candidate) -> str:
    """Return an atom text or negated text with {X} replaced by the quoted entity."""
    return template.replace("{X}", repr(candidate.entity))


def context_block(candidate: Candidate) -> str:
    """Return "Context:\\n<stripped document>\\n\\n", or "" for a blank document."""
    document = candidate.text.strip()
    return f"Context:\n{document}\n\n" if document else ""


def question_prompt(
    context: str, candidate: Candidate, statement: str, question: str
) -> str:
    """
    Return the shared question layout of the yes/no-style estimators.

    Args:
        context (str):
            A context block, or "" for none.
        candidate (Candidate):
            The entity, quoted after "Entity:".
        statement (str):
            The statement with the entity already filled in.
        question (str):
            The question, followed by "Answer:".

    Returns:
        str: The prompt.
    """
    return (
        f"{context}Entity: {candidate.entity!r}\n"
        f"Statement: {statement}\n\n{question}\nAnswer:"
    )
