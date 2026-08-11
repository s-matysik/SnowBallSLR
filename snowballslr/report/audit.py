"""Append-only JSONL audit trail."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

__all__ = ["AuditLog"]


class AuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, iteration: int, event: str, **fields: Any) -> None:
        record = {"iteration": iteration, "event": event, **fields}
        line = json.dumps(record, sort_keys=True, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return iter(())
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
