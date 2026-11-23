import json
from pathlib import Path

TITLES_FILE = "titles.jsonl"
TEXTS_FILE = "entities.jsonl"


def write_index_documents(
    index_path: Path, titles: list[str], texts: list[str]
) -> None:
    """
    Save the title and text of every indexed document, one per line, in index order.

    Args:
        index_path (Path):
            The index folder.
        titles (list[str]):
            The entity name of each document.
        texts (list[str]):
            The text of each document, in the same order.
    """
    with (index_path / TITLES_FILE).open("w", encoding="utf-8") as file:
        for title in titles:
            file.write(json.dumps({"title": title}, ensure_ascii=False) + "\n")
    with (index_path / TEXTS_FILE).open("w", encoding="utf-8") as file:
        for text in texts:
            file.write(json.dumps({"entity": text}, ensure_ascii=False) + "\n")


def _read_field(path: Path, field: str) -> list[str]:
    """Return one field of every line of a JSON Lines file."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line)[field] for line in file]


def read_index_documents(index_path: Path) -> tuple[list[str], list[str]]:
    """
    Return the titles and texts of every indexed document, in index order.

    Args:
        index_path (Path):
            The index folder, holding titles.jsonl and entities.jsonl.

    Returns:
        tuple[list[str], list[str]]: The titles and the texts.
    """
    titles = _read_field(index_path / TITLES_FILE, "title")
    texts = _read_field(index_path / TEXTS_FILE, "entity")
    if len(titles) != len(texts):
        raise ValueError(
            f"{TITLES_FILE} has {len(titles)} rows but {TEXTS_FILE} has {len(texts)}"
        )
    return titles, texts
