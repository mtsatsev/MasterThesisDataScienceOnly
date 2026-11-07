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
    formula = logical_query
    for text in sorted(atom_names, key=len, reverse=True):
        formula = formula.replace(text, atom_names[text])
    formula = formula.replace("AND", ",").replace("OR", ";").replace("NOT", "\\+")
    return formula + ")" * (formula.count("(") - formula.count(")"))
