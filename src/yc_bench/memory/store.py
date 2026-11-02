"""Append-only JSONL persistence for structured memory events.

One event per line, schema-versioned. The file is opened in append mode and
flushed on every write so a crashed run keeps whatever it had already observed
(the same durability property the transcript gives the rest of the harness).

Nothing in this module ever rewrites or removes a line: variant (ii) is
add-only by construction, which is what makes the later "does revising help?"
comparison ((iii)) measurable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from .schema import MemoryEvent, validate_event

logger = logging.getLogger(__name__)


class MemoryStore:
    """A tiny append-only event log with an in-memory mirror."""

    def __init__(self, path: str | Path | None = None) -> None:
        """Open the log at ``path`` (``None`` keeps it memory-only)."""
        self.path = Path(path) if path is not None else None
        self._events: list[MemoryEvent] = []
        self._load()

    # -- reading ----------------------------------------------------------

    def _load(self) -> None:
        """Read whatever is already on disk; a corrupt tail is skipped, not fatal."""
        if self.path is None or not self.path.exists():
            return
        bad = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._events.append(MemoryEvent.from_dict(json.loads(line)))
                except Exception:  # noqa: BLE001 - a corrupt tail must not break a run
                    bad += 1
        if bad:
            logger.warning("MemoryStore: skipped %d unreadable line(s) in %s", bad, self.path)
        logger.info(
            "MemoryStore: loaded %d event(s) from %s", len(self._events), self.path
        )

    def events(self) -> list[MemoryEvent]:
        """A copy of every event, in write order."""
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    # -- writing ----------------------------------------------------------

    def append(self, event: MemoryEvent) -> None:
        """Append one validated event and flush it to disk immediately."""
        validate_event(event)
        self._events.append(event)
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def append_many(self, events: Iterable[MemoryEvent]) -> int:
        """Append each event in turn; returns how many were written."""
        count = 0
        for ev in events:
            self.append(ev)
            count += 1
        return count


__all__ = ["MemoryStore"]
