import logging
import re

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
)

_predicate_mismatch_warned: set[tuple[frozenset, frozenset]] = set()


def phrase_to_predicate_name(phrase: str) -> str:
    """
    Convert a phrase with a placeholder '{X}' into a valid Prolog predicate name.
    Of the form "{X} is a restaurant" -> "is_a_restaurant"

    Args:
        phrase (str): The input phrase containing the placeholder '{X}'.

    Returns:
        str: A valid Prolog predicate name.

    """
    s = phrase.replace("{X}", "").strip().lower()
    s = re.sub(r"\W+", "_", s).strip("_")
    if not s or not s[0].isalpha():
        s = "p_" + s
    return s


class ProblogAtom(BaseModel):
    """A Problog atom representing a probabilistic fact."""

    atom: str = Field(
        min_length=1, description="The atomic statement in Problog format"
    )
    probability: float = Field(
        ge=0, le=1, default=0, description="The probability of the atomic statement"
    )
    context: str | None = Field(
        default=None,
        description="Optional context or additional information",
        min_length=1,
    )

    model_config = ConfigDict(arbitrary_types_allowed=False, extra="forbid")

    @computed_field
    @property
    def problog_atom_format(self) -> str:
        return phrase_to_predicate_name(self.atom)

    def to_proposition(self, entity: str | None = None) -> str:
        """Convert the ProblogAtom to a Problog string representation.

        Returns:
            PrologString: The Problog string representation of the atom.
        """
        entity_str = entity if entity is not None else "{X}"
        return f"{self.probability}::{self.problog_atom_format}({entity_str!r})."

    def to_deepproblog_fact(self, entity: str | None = None) -> str:
        """Convert the atom to a DeepProbLog-compatible fact.

        The initial DeepProbLog integration uses fixed probabilistic facts,
        which share syntax with ProbLog. Neural predicates can replace this in a
        later step.
        """
        return self.to_proposition(entity)

    def to_prompt_with_context(self, entity: str, separator: str = "\n\n") -> str:
        """Convert the ProblogAtom to a prompt format for LLMs.

        Args:
            entity (str): The entity to replace the placeholder '{X}'.
            separator (str): The separator between context and atom.

        Returns:
            str: The prompt string.
        """
        context_str = self.context + separator if self.context is not None else ""
        # pylint: disable=maybe-no-member
        return context_str + self.atom.replace("{X}", f"{entity!r}")

    def to_prompt(self, entity: str) -> str:
        """Convert the ProblogAtom to a prompt format for LLMs.

        Args:
            entity (str): The entity to replace the placeholder '{X}'.

        Returns:
            str: The prompt string.
        """
        # pylint: disable=maybe-no-member
        return self.atom.replace("{X}", f"{entity!r}")


class ProblogFormula(BaseModel):
    """A Problog formula representing a logical formula."""

    formula: str = Field(
        min_length=1, description="The logical formula in Problog format"
    )
    head: str = Field(
        min_length=1,
        default="formula",
        description="The head predicate name for the formula",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @computed_field
    def problog_formula_format(self) -> str:
        """
        Convert a logical string with {X}, AND, OR, NOT into a simple ProbLog-style rule.

        Example input:
            '({X} is a toy AND ({X} yells when squeezed OR {X} is red in color)) AND NOT({X} is a dog)'

        Example output (note: uses {X} instead of Prolog's X):
            formula({X}) :-
                toy({X}),
                (yells_when_squeezed({X}); red_in_color({X})),
                \\+ dog({X}).
        """

        # 1) Find all atomic phrases of the form "{X} ... <before next AND/OR/NOT/)/end>"
        atom_pattern = re.compile(
            r"\{X\}[^()]*?(?=(\bAND\b|\bOR\b|\bNOT\b|\)|$))",
            flags=re.IGNORECASE,
        )

        phrases: list[str] = []
        for m in atom_pattern.finditer(self.formula):
            phrase = m.group(0).strip()
            if phrase not in phrases:
                phrases.append(phrase)

        # 2) Map each phrase to a predicate name
        phrase_to_pred = {p: phrase_to_predicate_name(p) for p in phrases}

        # 3) Replace phrases with predicate({X})
        prolog_body = self.formula
        for phrase, pred in phrase_to_pred.items():
            # pylint: disable=maybe-no-member
            prolog_body = prolog_body.replace(phrase, f"{pred}({{X}})")

        # 4) Replace logical operators
        prolog_body = re.sub(r"\bAND\b", ",", prolog_body, flags=re.IGNORECASE)
        prolog_body = re.sub(r"\bOR\b", ";", prolog_body, flags=re.IGNORECASE)
        prolog_body = re.sub(r"\bNOT\b", r"\\+", prolog_body, flags=re.IGNORECASE)

        # 5) Normalize whitespace and pretty-print with newlines after commas
        prolog_body = re.sub(r"\s+", " ", prolog_body).strip()  # squeeze spaces
        prolog_body = prolog_body.replace(", ", ", ")
        prolog_body = prolog_body.replace(",", ",\n    ")

        # 6) Build final rule; keep {X} as requested
        result = f"{self.head}({{X}}) :-\n    {prolog_body}."
        return result

    def to_problog(self, atoms: list[ProblogAtom], entity: str) -> str:
        """
        Convert the ProblogFormula to a Problog string representation,
        including the given atoms and replacing '{X}' with the entity.
        Args:
            atoms (list[ProblogAtom]): List of Problog atoms to include.
            entity (str): The entity to replace the placeholder '{X}'.

        Returns:
            str: The Problog string representation of the formula with atoms.
        """
        _log = logging.getLogger(__name__)

        entity_str = entity if entity is not None else "{X}"
        propositions = "\n".join(atom.to_proposition(entity) for atom in atoms)
        formula_rule = self.problog_formula_format.replace("{X}", f"{entity_str!r}")
        query_directive = f"query({self.head}({entity_str!r}))."
        provided_predicates = frozenset(atom.problog_atom_format for atom in atoms)
        formula_predicates = frozenset(re.findall(r"(\w+)\(", formula_rule)) - {
            self.head
        }
        missing = formula_predicates - provided_predicates
        if missing:
            cache_key = (formula_predicates, provided_predicates)
            if cache_key not in _predicate_mismatch_warned:
                _predicate_mismatch_warned.add(cache_key)
                _log.warning(
                    "ProbLog predicate mismatch: formula references predicates not "
                    "supplied by any atom: %s. This will produce wrong probabilities. "
                    "Formula predicates: %s | Atom predicates: %s",
                    missing,
                    formula_predicates,
                    provided_predicates,
                )

        return propositions + "\n" + formula_rule + "\n" + query_directive

    def to_deepproblog(self, atoms: list[ProblogAtom], entity: str) -> str:
        """Convert the formula to a DeepProbLog-compatible program.

        The initial DeepProbLog integration reuses fixed probabilistic facts and
        standard rule syntax so the logical structure stays identical.
        """
        entity_str = entity if entity is not None else "{X}"
        propositions = "\n".join(atom.to_deepproblog_fact(entity) for atom in atoms)
        formula_rule = self.problog_formula_format.replace("{X}", f"{entity_str!r}")
        query_directive = f"query({self.head}({entity_str!r}))."
        return propositions + "\n" + formula_rule + "\n" + query_directive
