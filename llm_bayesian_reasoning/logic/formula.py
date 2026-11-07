from problog import get_evaluatable
from problog.engine import UnknownClause
from problog.program import PrologString


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

    facts = "\n".join(
        f"{probability}::{atom}." for atom, probability in atom_probabilities.items()
    )
    program = f"{facts}\nformula_holds :- {formula}.\nquery(formula_holds)."

    try:
        result = get_evaluatable().create_from(PrologString(program)).evaluate()
    except UnknownClause:
        raise ValueError

    return next(iter(result.values()))
