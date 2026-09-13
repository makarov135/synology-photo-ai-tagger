from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class ManifestStore:
    """Atomic JSON checkpoint so a long run can resume after interruption."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "updated_at": None,
            "assets": {},
        }
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if loaded.get("schema_version") != self.SCHEMA_VERSION:
                raise RuntimeError(f"不支持的清单版本：{loaded.get('schema_version')}")
            self.data = loaded

    @property
    def assets(self) -> dict[str, dict[str, Any]]:
        return self.data.setdefault("assets", {})

    def get(self, key: str) -> dict[str, Any] | None:
        value = self.assets.get(key)
        return value if isinstance(value, dict) else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        self.assets[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
