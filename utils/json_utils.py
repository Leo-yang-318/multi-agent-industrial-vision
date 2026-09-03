import json
from pathlib import Path


def read_json(json_path: str) -> dict:
    path = Path(json_path)

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict, json_path: str) -> None:
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)