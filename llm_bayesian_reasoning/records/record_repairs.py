from pathlib import Path

from pydantic import TypeAdapter

from llm_bayesian_reasoning.records.prepare import RecordRepair

RECORD_REPAIRS_PATH = Path(__file__).with_name("record_repairs_draft.json")


def load_record_repairs(path: Path = RECORD_REPAIRS_PATH) -> dict[int, RecordRepair]:
    """
    Return the record repairs of a JSON file, keyed by query id.

    Args:
        path (Path):
            The JSON file, mapping query ids to their repairs.

    Returns:
        dict[int, RecordRepair]: The validated repairs.
    """
    return TypeAdapter(dict[int, RecordRepair]).validate_json(path.read_bytes())
