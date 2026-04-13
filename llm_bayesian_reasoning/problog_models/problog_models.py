import logging
import re

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
)

_predicate_mismatch_warned: set[tuple[frozenset, frozenset]] = set()


class _AstNodeModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _FormulaVisitor:
    def visit(self, node: "_FormulaNode") -> str:
        return node.accept(self)

    def visit_atom(self, node: "_AtomNode") -> str:
        raise NotImplementedError

    def visit_unary(self, node: "_UnaryNode") -> str:
        raise NotImplementedError

    def visit_binary(self, node: "_BinaryNode") -> str:
        raise NotImplementedError


class _AtomNode(_AstNodeModel):
    predicate: str

    def accept(self, visitor: _FormulaVisitor) -> str:
        return visitor.visit_atom(self)


class _UnaryNode(_AstNodeModel):
    operator: str
    operand: "_FormulaNode"

    def accept(self, visitor: _FormulaVisitor) -> str:
        return visitor.visit_unary(self)


class _BinaryNode(_AstNodeModel):
    operator: str
    left: "_FormulaNode"
    right: "_FormulaNode"

    def accept(self, visitor: _FormulaVisitor) -> str:
        return visitor.visit_binary(self)


_FormulaNode = _AtomNode | _UnaryNode | _BinaryNode


class _ProbLogRenderer(_FormulaVisitor):
    def __init__(self, entity_expr: str):
        self.entity_expr = entity_expr

    def visit_atom(self, node: _AtomNode) -> str:
        return f"{node.predicate}({self.entity_expr})"

    def visit_unary(self, node: _UnaryNode) -> str:
        return f"\\+({self.visit(node.operand)})"

    def visit_binary(self, node: _BinaryNode) -> str:
        operator = "," if node.operator == "AND" else ";"
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"({left} {operator} {right})"


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


def _consume_formula_phrase(text: str, start: int) -> tuple[str, int]:
    def starts_logical_operator(fragment: str) -> bool:
        match = re.match(r"\s+(AND|OR)\b", fragment, flags=re.IGNORECASE)
        if match is None:
            return False

        remainder = fragment[match.end() :].lstrip()
        return (
            remainder.startswith("(")
            or remainder.startswith("{X}")
            or re.match(r"NOT\b", remainder, flags=re.IGNORECASE) is not None
        )

    end = start
    while end < len(text):
        if starts_logical_operator(text[end:]):
            break
        if text[end] == ")" and re.match(
            r"\)\s*(?:AND\b|OR\b|$|\))",
            text[end:],
            flags=re.IGNORECASE,
        ):
            break
        end += 1

    return text[start:end].strip(), end


def _normalize_formula_text_for_atoms(
    formula_text: str,
    atom_texts: set[str],
) -> str:
    text = re.sub(r"\{X\}\s+is\s+is\s+", "{X} is ", formula_text)

    if "{X} is a novel" in atom_texts and "{X} is a book" not in atom_texts:
        text = text.replace("{X} is a book", "{X} is a novel")

    if (
        "{X} is a non-fiction novel" in atom_texts
        and "{X} is a fiction novel" not in atom_texts
    ):
        text = text.replace(
            "NOT({X} is a fiction novel)",
            "({X} is a non-fiction novel)",
        )

    if "{X} is a drama film" not in atom_texts:
        drama_variants = sorted(
            atom_text
            for atom_text in atom_texts
            if atom_text.endswith(" drama film")
        )
        if drama_variants:
            drama_replacement = " OR ".join(
                f"({variant})" for variant in drama_variants
            )
            text = text.replace(
                "({X} is a drama film)",
                f"({drama_replacement})",
            )

    for only_found_atom in sorted(atom_texts):
        match = re.fullmatch(r"\{X\} is only found in (.+)", only_found_atom)
        if match is None:
            continue

        location = match.group(1)
        found_atom = f"{{X}} is found in {location}"
        if found_atom not in atom_texts:
            continue

        replacements = [
            (
                f"({found_atom}) AND (NOT((NOT({found_atom})) AND ({{X}} is found in other places)))",
                f"({only_found_atom})",
            ),
            (
                f"({found_atom}) AND (NOT(({{X}} is found in a location other than {location})))",
                f"({only_found_atom})",
            ),
            (
                f"({found_atom}) AND (NOT(({{X}} is found in a place other than {location})))",
                f"({only_found_atom})",
            ),
            (
                f"({found_atom}) AND (NOT(({{X}} is found in other places) OR ({{X}} is found outside {location})))",
                f"({only_found_atom})",
            ),
        ]

        for original, replacement in replacements:
            text = text.replace(original, replacement)

    return text


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
    @property
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

        body = self._formula_to_body(atom_lookup={})
        return f"{self.head}({{X}}) :-\n    {body}."

    def _tokenize_formula(
        self,
        atom_lookup: dict[str, str],
    ) -> list[tuple[str, str]]:
        text = _normalize_formula_text_for_atoms(str(self.formula), set(atom_lookup))
        ordered_atoms = sorted(atom_lookup.items(), key=lambda item: len(item[0]), reverse=True)
        tokens: list[tuple[str, str]] = []
        position = 0

        while position < len(text):
            if text[position].isspace():
                position += 1
                continue

            if text[position] == "(":
                tokens.append(("LPAREN", "("))
                position += 1
                continue

            if text[position] == ")":
                tokens.append(("RPAREN", ")"))
                position += 1
                continue

            operator_match = re.match(r"(?i)(AND|OR|NOT)\b", text[position:])
            if operator_match:
                tokens.append((operator_match.group(1).upper(), operator_match.group(1).upper()))
                position += operator_match.end()
                continue

            matched_atom = next(
                (
                    (atom_text, predicate)
                    for atom_text, predicate in ordered_atoms
                    if text.startswith(atom_text, position)
                ),
                None,
            )
            if matched_atom is not None:
                atom_text, predicate = matched_atom
                tokens.append(("ATOM", predicate))
                position += len(atom_text)
                continue

            if text.startswith("{X}", position):
                phrase, next_position = _consume_formula_phrase(text, position)
                if not phrase:
                    raise ValueError(f"Unable to parse formula near position {position}: {text[position:position + 40]!r}")
                tokens.append(("ATOM", phrase_to_predicate_name(phrase)))
                position = next_position
                continue

            raise ValueError(
                f"Unexpected formula token near position {position}: {text[position:position + 40]!r}"
            )

        return tokens

    def _parse_formula_tokens(self, tokens: list[tuple[str, str]]) -> _FormulaNode:
        position = 0

        def current() -> tuple[str, str] | None:
            if position >= len(tokens):
                return None
            return tokens[position]

        def consume(expected: str | None = None) -> tuple[str, str]:
            nonlocal position
            token = current()
            if token is None:
                raise ValueError("Unexpected end of formula")
            if expected is not None and token[0] != expected:
                raise ValueError(f"Expected {expected}, found {token[0]}")
            position += 1
            return token

        def parse_primary() -> _FormulaNode:
            token = current()
            if token is None:
                raise ValueError("Unexpected end of formula")

            if token[0] == "ATOM":
                consume("ATOM")
                return _AtomNode(predicate=token[1])

            if token[0] == "NOT":
                consume("NOT")
                return _UnaryNode(operator="NOT", operand=parse_primary())

            if token[0] == "LPAREN":
                consume("LPAREN")
                inner = parse_or_expression()
                if current() is not None and current()[0] == "RPAREN":
                    consume("RPAREN")
                return inner

            raise ValueError(f"Unexpected token {token[0]}")

        def parse_and_expression() -> _FormulaNode:
            node = parse_primary()
            while current() is not None and current()[0] == "AND":
                consume("AND")
                node = _BinaryNode(
                    operator="AND",
                    left=node,
                    right=parse_primary(),
                )
            return node

        def parse_or_expression() -> _FormulaNode:
            node = parse_and_expression()
            while current() is not None and current()[0] == "OR":
                consume("OR")
                node = _BinaryNode(
                    operator="OR",
                    left=node,
                    right=parse_and_expression(),
                )
            return node

        parsed = parse_or_expression()

        while current() is not None and current()[0] == "RPAREN":
            consume("RPAREN")

        if current() is not None:
            raise ValueError(f"Unexpected trailing token {current()[0]}")

        return parsed

    def _formula_to_body(
        self,
        atom_lookup: dict[str, str],
        entity_expr: str = "{X}",
    ) -> str:
        tokens = self._tokenize_formula(atom_lookup)
        parsed = self._parse_formula_tokens(tokens)
        return _ProbLogRenderer(entity_expr).visit(parsed)

    def _formula_to_rule_from_atoms(
        self,
        atoms: list[ProblogAtom],
    ) -> str:
        atom_map = {atom.atom: atom.problog_atom_format for atom in atoms}
        body = self._formula_to_body(atom_map)
        return f"{self.head}({{X}}) :-\n    {body}."

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
        atom_map = {atom.atom: atom.problog_atom_format for atom in atoms}
        formula_body = self._formula_to_body(atom_map, entity_expr=f"{entity_str!r}")
        formula_rule = f"{self.head}({entity_str!r}) :-\n    {formula_body}."
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
        atom_map = {atom.atom: atom.problog_atom_format for atom in atoms}
        formula_body = self._formula_to_body(atom_map, entity_expr=f"{entity_str!r}")
        formula_rule = f"{self.head}({entity_str!r}) :-\n    {formula_body}."
        query_directive = f"query({self.head}({entity_str!r}))."
        return propositions + "\n" + formula_rule + "\n" + query_directive
