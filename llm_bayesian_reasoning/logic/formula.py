from problog import get_evaluatable
from problog.engine import UnknownClause
from problog.logic import Constant, Term
from problog.program import PrologString, SimpleProgram


def build_problog_program(
    formula: str, atom_probabilities: dict[str, float]
) -> SimpleProgram:
    """
    Return the ProbLog program that queries a formula over probabilistic atoms.

    Args:
        formula (str):
            The ProbLog formula to query.
        atom_probabilities (dict[str, float]):
            A dictionary mapping atom names to their probabilities.

    Returns:
        SimpleProgram: One fact per atom, followed by the rule and the query.
    """
    program = SimpleProgram()
    for atom, probability in atom_probabilities.items():
        program += Term(atom, p=Constant(probability))
    for statement in PrologString(f"formula_holds :- {formula}.\nquery(formula_holds)."):
        program += statement
    return program


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
        result = get_evaluatable().create_from(program).evaluate()
    except UnknownClause:
        raise ValueError

    return next(iter(result.values()))
