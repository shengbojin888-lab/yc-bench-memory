"""Facade wiring extraction, persistence and rendering together.

The agent loop talks to :class:`StructuredMemory` only:

    memory.observe_turn(turn=..., tool_calls=..., sim_time=..., executor=...)
    prompt_text = memory.context_block()

Everything the module needs comes from observations the agent could itself
have produced, so enabling it does not grant extra observation access — it
only changes what is remembered and how it is recalled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from .extractor import StructuredExtractor
from .store import MemoryStore
from .view import (
    build_records,
    client_stats,
    render_client_view,
    render_staff_view,
    staff_stats,
)

logger = logging.getLogger(__name__)

MODE_OFF = "off"
MODE_STRUCTURED = "structured"


@dataclass
class MemorySettings:
    mode: str = MODE_OFF
    inject_client_view: bool = True
    inject_staff_view: bool = False
    client_token_budget: int = 300
    staff_token_budget: int = 150
    # Issue the read-only follow-up (`task inspect` after an accept) so the
    # accepted workload is captured even if the agent never inspects itself.
    followup_reads: bool = True

    @property
    def enabled(self) -> bool:
        return self.mode != MODE_OFF


class StructuredMemory:
    def __init__(
        self,
        store: MemoryStore | None = None,
        settings: MemorySettings | None = None,
    ) -> None:
        self.settings = settings or MemorySettings()
        # NB: `store or MemoryStore(None)` would silently discard a real store,
        # because MemoryStore defines __len__ and a fresh (empty) log is falsy.
        # That bug disabled JSONL persistence on every run; always compare to None.
        self.store = store if store is not None else MemoryStore(None)
        self._extractor = StructuredExtractor()
        self._extractor.prime(self.store.events())
        self._last_block = ""
        self.followups_issued = 0
        self.followups_failed = 0

    # -- write side -------------------------------------------------------

    def observe_turn(
        self,
        *,
        turn: int,
        tool_calls: list[dict[str, Any]] | None,
        sim_time: str | None = None,
        executor: Callable[[str], Any] | None = None,
    ) -> int:
        """Extract events from one finished turn. Returns number persisted."""
        events, followups = self._extractor.observe_turn(
            tool_calls, turn=turn, sim_time=sim_time
        )
        written = self.store.append_many(events)

        if (
            followups
            and self.settings.followup_reads
            and executor is not None
        ):
            for command in followups:
                written += self._run_followup(
                    command, executor, turn=turn, sim_time=sim_time
                )

        return written

    def _run_followup(
        self,
        command: str,
        executor: Callable[[str], Any],
        *,
        turn: int,
        sim_time: str | None,
    ) -> int:
        try:
            result = executor(command)
        except Exception as exc:  # noqa: BLE001 - memory must never kill a run
            self.followups_failed += 1
            logger.warning("Memory follow-up failed (%s): %s", command, exc)
            return 0
        if isinstance(result, dict) and result.get("ok") is False:
            # The executor reports subprocess problems as ok=False rather than
            # raising; count them so a silently broken memory layer is visible
            # in the run summary instead of looking like "no data".
            self.followups_failed += 1
            logger.warning(
                "Memory follow-up returned no data (%s): %s",
                command,
                (result.get("stderr") or "").strip()[:200],
            )
            return 0
        self.followups_issued += 1
        ev = self._extractor.observe_followup(
            command, result, turn=turn, sim_time=sim_time
        )
        if ev is None:
            return 0
        self.store.append(ev)
        return 1

    # -- read side --------------------------------------------------------

    def context_block(self) -> str:
        """The text appended to the turn's user message (empty when disabled)."""
        if not self.settings.enabled:
            return ""
        records = build_records(self.store.events())
        if not records:
            return ""

        blocks: list[str] = []
        if self.settings.inject_client_view:
            block = render_client_view(
                client_stats(records),
                token_budget=self.settings.client_token_budget,
            )
            if block:
                blocks.append(block)
        if self.settings.inject_staff_view:
            block = render_staff_view(
                staff_stats(records),
                token_budget=self.settings.staff_token_budget,
            )
            if block:
                blocks.append(block)

        self._last_block = "\n\n".join(blocks)
        return self._last_block

    # -- diagnostics ------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        records = build_records(self.store.events())
        stats = client_stats(records)
        return {
            "events": len(self.store),
            "tasks": len(records),
            "clients": len(stats),
            "creep_measured": sum(1 for r in records.values() if r.creep_ratio is not None),
            "view_tokens": len(self._last_block) // 4,
            "followups_issued": self.followups_issued,
            "followups_failed": self.followups_failed,
        }


__all__ = [
    "MODE_OFF",
    "MODE_STRUCTURED",
    "MemorySettings",
    "StructuredMemory",
]
