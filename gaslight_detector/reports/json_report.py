from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(result: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
