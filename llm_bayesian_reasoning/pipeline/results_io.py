import json
from pathlib import Path
from typing import Any, TextIO


def safe_name(value: str) -> str:
    return Path(value).stem.replace("/", "_").replace(" ", "_")


def write_result_row(
    file_handle: TextIO,
    record_id: int | str,
    record_result: dict[str, Any],
) -> None:
    row = {"id": record_id, **record_result}
    file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    file_handle.flush()