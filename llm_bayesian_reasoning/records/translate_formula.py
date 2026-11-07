import re


def translate_logical_query(logical_query: str, atom_names: dict[str, str]) -> str:
    """
    Return the ProbLog formula for a logical query written in natural language.

    Args:
        logical_query (str):
            The query text, with atoms joined by AND, OR and NOT.
        atom_names (dict[str, str]):
            A dictionary mapping atom texts to their ProbLog names.

    Returns:
        str: The ProbLog formula, using the atom names and , ; \\+ as operators.
    """
    if not logical_query:
        raise ValueError("Cannot translate an empty logical query")

    formula = logical_query
    for text in sorted(atom_names, key=len, reverse=True):
        formula = formula.replace(text, atom_names[text])

    known = "|".join(atom_names.values())
    leftover = re.sub(rf"\b(?:AND|OR|NOT|{known})\b|[()]", " ", formula)
    if leftover.strip():
        raise ValueError(
            f"Query {logical_query!r} has text that is not an atom: "
            f"{' '.join(leftover.split())!r}"
        )

    missing_brackets = formula.count("(") - formula.count(")")
    if missing_brackets < 0:
        raise ValueError(f"Query {logical_query!r} has an extra closing bracket")

    formula = formula.replace("AND", ",").replace("OR", ";").replace("NOT", "\\+")
    return formula + ")" * missing_brackets
