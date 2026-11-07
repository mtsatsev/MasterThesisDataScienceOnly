from problog import get_evaluatable
from problog.engine import UnknownClause
from problog.program import PrologString


def build_problog_program(formula: str, atom_probabilities: dict[str, float]) -> str:
    """
    Return the ProbLog program that queries a formula over probabilistic atoms.

    Args:
        formula (str):
            The ProbLog formula to query.
        atom_probabilities (dict[str, float]):
            A dictionary mapping atom names to their probabilities.

    Returns:
        str: The program text: one fact per atom, the rule and the query.
    """
    facts = "\n".join(
        f"{probability}::{atom}." for atom, probability in atom_probabilities.items()
    )
    return f"{facts}\nformula_holds :- {formula}.\nquery(formula_holds)."


def evaluate_formula(formula: str, atom_probabilities: dict[str, float]) -> float:
    """
    Return the probability that a ProbLog formula holds.
    
    Args:
        formula (str):
            The ProbLog formula to evaluate.
        atom_probabilities (dict[str, float]):
            A dictionary mapping atom names to their probabilities.

    Returns: 
        float: The probability that the formula holds.
    """
    for probability in atom_probabilities.values():
        if not 0.0 <= probability <= 1.0:
            raise ValueError

    program = build_problog_program(formula, atom_probabilities)

    try:
        result = get_evaluatable().create_from(PrologString(program)).evaluate()
    except UnknownClause:
        raise ValueError

    return next(iter(result.values()))
