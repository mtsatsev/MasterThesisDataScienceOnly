from pydantic import BaseModel


class Atom(BaseModel):
    """A query condition with its ProbLog name and its two prompt texts."""

    name: str
    text: str
    negated_text: str


def build_atoms(atom_texts: list[str], negated_texts: list[str]) -> list[Atom]:
    """
    Return one named atom per distinct atom text, in input order.

    Args:
        atom_texts (list[str]):
            The raw atom texts of a query.
        negated_texts (list[str]):
            The raw negated text of each atom, in the same order.

    Returns:
        list[Atom]: The atoms, with placeholders normalised to {X}.
    """
    if not atom_texts:
        raise ValueError("Query has no atoms")
    if len(atom_texts) != len(negated_texts):
        raise ValueError(
            f"Query has {len(atom_texts)} atoms but {len(negated_texts)} negated atoms"
        )

    atoms: dict[str, Atom] = {}
    for text, negated_text in zip(atom_texts, negated_texts):
        text = text.replace("{x}", "{X}")
        if text not in atoms:
            atoms[text] = Atom(
                name=f"a{len(atoms)}",
                text=text,
                negated_text=negated_text.replace("{x}", "{X}"),
            )
    return list(atoms.values())
