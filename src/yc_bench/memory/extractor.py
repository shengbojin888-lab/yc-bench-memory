"""Deterministic extraction: observation -> structured events (variant (ii)).

The extractor is a *reader*, not a writer of natural language: it parses the
JSON that the CLI already returned and maps fixed fields onto
:class:`~yc_bench.memory.schema.MemoryEvent`. There is no LLM call anywhere in
this module, which keeps prompt-token / API-cost comparisons against the
baseline honest.

Two sources of observations are consumed:

* the agent's own command results in the turn that just ended
  (``raw_result["tool_calls"]``, which holds the full stdout — unlike the
  transcript, which truncates to 500 chars);
* ``sim resume`` wake events, which already carry the outcome of finished tasks
  (success, funds delta, deadline margin, salary bump).

When ``task accept`` returns, the accepted workload is *not* in the response
(only title/status/accepted_at/deadline/replacement). The extractor therefore
emits a follow-up read of ``task inspect`` — the exact read-only command the
agent could have issued — so that advertised vs. actual workload can be
compared at all. This follow-up never advances the clock and never enters the
agent's visible history.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any, Iterable

from .schema import (
    ACTOR_AGENT,
    ACTOR_EXTRACTOR,
    EVENT_TASK_ACCEPTED,
    EVENT_TASK_ASSIGNED,
    EVENT_TASK_CANCELLED,
    EVENT_TASK_COMPLETED,
    EVENT_TASK_INSPECTED,
    MemoryEvent,
    make_evidence,
    make_task_key,
)

_MARGIN_RE = re.compile(r"(ahead|late)\s+by\s+([0-9.]+)\s*h", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_command(command: str) -> tuple[str, ...]:
    """Split ``yc-bench task accept --task-id X`` into its meaningful tokens."""
    if not command:
        return ()
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        tokens = command.split()
    if tokens:
        head = tokens[0].strip('"').lower()
        if head.endswith("yc-bench") or head.endswith("yc-bench.exe"):
            tokens = tokens[1:]
    return tuple(t.strip('"') for t in tokens)


def flag(tokens: tuple[str, ...], name: str) -> str | None:
    """Return the value of ``--name value`` / ``--name=value`` if present."""
    for i, tok in enumerate(tokens):
        if tok == name and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith(name + "="):
            return tok.split("=", 1)[1]
    return None


def parse_tool_result(result: Any) -> dict[str, Any] | None:
    """Unwrap the command executor envelope and return the CLI's JSON payload.

    The executor returns ``{"ok": bool, "exit_code": int, "stdout": "<json>",
    "stderr": ..., "command": ...}``. Older/mocked callers may hand over the
    payload directly, which is also accepted.
    """
    if not result:
        return None
    payload: Any = result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(payload, dict):
        return None
    if "stdout" in payload:
        if payload.get("ok") is False:
            return None
        stdout = payload.get("stdout") or ""
        if not isinstance(stdout, str) or not stdout.strip():
            return None
        try:
            inner = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            return None
        return inner if isinstance(inner, dict) else None
    return payload


def parse_deadline_margin(text: str | None) -> float | None:
    """``"ahead by 97h"`` -> +97.0, ``"late by 5h"`` -> -5.0."""
    if not text:
        return None
    m = _MARGIN_RE.search(text)
    if not m:
        return None
    hours = float(m.group(2))
    return hours if m.group(1).lower() == "ahead" else -hours


def _qty_map(requirements: Iterable[dict[str, Any]] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for req in requirements or []:
        domain = req.get("domain")
        qty = req.get("required_qty")
        if domain is None or qty is None:
            continue
        try:
            out[str(domain)] = float(qty)
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class StructuredExtractor:
    """Turns one turn's observations into zero or more append-only events."""

    def __init__(self) -> None:
        # title -> what ``market browse`` advertised for it.
        self.market: dict[str, dict[str, Any]] = {}
        # title -> task_keys accepted but not yet resolved, oldest first.
        self._open: dict[str, list[str]] = {}
        self._closed: set[str] = set()
        self._client_map: dict[str, str | None] = {}
        # Last simulation time seen, used as the timestamp for observations
        # whose payload does not carry one (assign / cancel).
        self._clock: str = ""

    # -- state ------------------------------------------------------------

    def prime(self, events: Iterable[MemoryEvent]) -> None:
        """Rebuild join state from a persisted log (used on resume)."""
        for ev in events:
            if ev.type == EVENT_TASK_ACCEPTED:
                self._open.setdefault(ev.title, []).append(ev.task_key)
                advertised = ev.payload.get("advertised_qty")
                if advertised:
                    self.market.setdefault(
                        ev.title,
                        {
                            "client": ev.client,
                            "qty": dict(advertised),
                            "reward_cents": ev.payload.get("advertised_reward_cents"),
                        },
                    )
            elif ev.type in (
                EVENT_TASK_COMPLETED,
                EVENT_TASK_CANCELLED,
            ):
                self._closed.add(ev.task_key)
        for keys in self._open.values():
            keys[:] = [k for k in keys if k not in self._closed]

    def _resolve(
        self,
        title: str,
        client: str | None = None,
        *,
        allow_closed: bool = False,
    ) -> str | None:
        """Find the task_key an observation refers to.

        Titles alone are ambiguous (market generations reuse them), so prefer a
        matching client and fall back to the newest accepted key. This is the
        only heuristic join in the module and it is recorded in evidence.

        ``allow_closed`` exists for read-only observations. A task that is
        dispatched and then completes inside the *same* turn (the usual case:
        ``sim resume`` right after ``task dispatch``) is already closed by the
        time the extractor's follow-up ``task inspect`` runs, so an open-only
        join silently drops the only observation of its real workload. The
        accepted workload stays true after completion, so attaching the read to
        the closed record is correct.
        """
        all_keys = self._open.get(title, [])
        keys = [k for k in all_keys if k not in self._closed]
        if not keys:
            return all_keys[-1] if (allow_closed and all_keys) else None
        if client:
            matching = [
                k for k in keys if self._client_of(k) == client
            ]
            if matching:
                return matching[-1]
        return keys[-1]

    def _client_of(self, task_key: str) -> str | None:
        return self._client_map.get(task_key)

    def _time_for(self, task_key: str, sim_time: str | None) -> str:
        """Best available timestamp for an observation without one of its own."""
        if sim_time:
            return sim_time
        if self._clock:
            return self._clock
        _, _, accepted_at = task_key.partition("@")
        return accepted_at or "unknown"

    # -- main entry point -------------------------------------------------

    def observe_turn(
        self,
        tool_calls: Iterable[dict[str, Any]] | None,
        *,
        turn: int,
        sim_time: str | None = None,
    ) -> tuple[list[MemoryEvent], list[str]]:
        """Parse one turn of command results.

        Returns ``(events, followup_commands)``. Follow-ups are read-only
        commands the memory module wants executed (currently: ``task inspect``
        right after an accept) so the record can carry the real workload.
        """
        events: list[MemoryEvent] = []
        followups: list[str] = []
        if sim_time:
            self._clock = sim_time

        for tc in tool_calls or []:
            command = tc.get("command") or ""
            tokens = parse_command(command)
            if len(tokens) < 2:
                continue
            sub = (tokens[0].lower(), tokens[1].lower())
            payload = parse_tool_result(tc.get("result"))
            if payload is None:
                continue

            if sub == ("market", "browse"):
                self._index_market(payload)
            elif sub == ("task", "accept"):
                ev, follow = self._on_accept(payload, command, turn, sim_time)
                if ev is not None:
                    events.append(ev)
                if follow:
                    followups.append(follow)
            elif sub == ("task", "inspect"):
                ev = self._on_inspect(payload, command, turn, sim_time)
                if ev is not None:
                    events.append(ev)
            elif sub == ("task", "assign"):
                ev = self._on_assign(payload, command, turn, sim_time)
                if ev is not None:
                    events.append(ev)
            elif sub == ("task", "cancel"):
                ev = self._on_cancel(payload, command, turn, sim_time)
                if ev is not None:
                    events.append(ev)
            elif sub == ("sim", "resume"):
                events.extend(self._on_resume(payload, command, turn, sim_time))

        return events, followups

    # -- per-command handlers --------------------------------------------

    def _index_market(self, payload: dict[str, Any]) -> None:
        for task in payload.get("tasks") or []:
            title = task.get("task_id")
            if not title:
                continue
            self.market[title] = {
                "client": task.get("client_name"),
                "qty": _qty_map(task.get("requirements")),
                "reward_cents": task.get("reward_funds_cents"),
            }

    def _on_accept(
        self,
        payload: dict[str, Any],
        command: str,
        turn: int,
        sim_time: str | None,
    ) -> tuple[MemoryEvent | None, str | None]:
        title = payload.get("task_id")
        accepted_at = payload.get("accepted_at")
        if not title or not accepted_at:
            return None, None

        seen = self.market.get(title, {})
        advertised_qty = dict(seen.get("qty") or {})
        client = seen.get("client")

        task_key = make_task_key(title, accepted_at)
        ev = MemoryEvent(
            type=EVENT_TASK_ACCEPTED,
            task_key=task_key,
            title=title,
            client=client,
            occurred_at=accepted_at,
            turn=turn,
            payload={
                "accepted_at": accepted_at,
                "deadline": payload.get("deadline"),
                "advertised_qty": advertised_qty,
                "advertised_reward_cents": seen.get("reward_cents"),
                "replacement_task_id": payload.get("replacement_task_id"),
            },
            evidence=[
                make_evidence(
                    command,
                    turn,
                    sim_time=accepted_at,
                    actor=ACTOR_AGENT,
                )
            ],
        )
        self._open.setdefault(title, []).append(task_key)
        self._client_map[task_key] = client
        # The agent's own view of the accepted workload is the next read.
        return ev, f"yc-bench task inspect --task-id {title}"

    def _on_inspect(
        self,
        payload: dict[str, Any],
        command: str,
        turn: int,
        sim_time: str | None,
        *,
        actor: str = ACTOR_AGENT,
    ) -> MemoryEvent | None:
        title = payload.get("title") or payload.get("task_id")
        client = payload.get("client_name")
        if not title:
            return None

        task_key = self._resolve(title, client, allow_closed=True)
        if task_key is None:
            # Inspecting a task that was never accepted (still in market): no
            # memory value, the agent already saw exactly this.
            return None

        observed_at = payload.get("accepted_at") or self._time_for(task_key, sim_time)
        ev = MemoryEvent(
            type=EVENT_TASK_INSPECTED,
            task_key=task_key,
            title=title,
            client=client or self._client_of(task_key),
            occurred_at=observed_at,
            turn=turn,
            payload={
                "observed_at": observed_at,
                "client": client,
                "qty": _qty_map(payload.get("requirements")),
                "deadline": payload.get("deadline"),
                "status": payload.get("status"),
            },
            evidence=[
                make_evidence(command, turn, sim_time=sim_time, actor=actor)
            ],
        )
        self._client_map[task_key] = client or self._client_of(task_key)
        return ev

    def _on_assign(
        self,
        payload: dict[str, Any],
        command: str,
        turn: int,
        sim_time: str | None,
    ) -> MemoryEvent | None:
        title = payload.get("task_id")
        employees = payload.get("newly_assigned") or []
        if not title:
            return None
        task_key = self._resolve(title)
        if task_key is None:
            return None
        return MemoryEvent(
            type=EVENT_TASK_ASSIGNED,
            task_key=task_key,
            title=title,
            client=self._client_of(task_key),
            occurred_at=self._time_for(task_key, sim_time),
            turn=turn,
            payload={"employees": list(employees)},
            evidence=[make_evidence(command, turn, sim_time=sim_time)],
        )

    def _on_cancel(
        self,
        payload: dict[str, Any],
        command: str,
        turn: int,
        sim_time: str | None,
    ) -> MemoryEvent | None:
        title = payload.get("task_id")
        if not title:
            return None
        task_key = self._resolve(title)
        if task_key is None:
            return None
        self._closed.add(task_key)
        return MemoryEvent(
            type=EVENT_TASK_CANCELLED,
            task_key=task_key,
            title=title,
            client=self._client_of(task_key),
            occurred_at=self._time_for(task_key, sim_time),
            turn=turn,
            payload={"reason": payload.get("reason")},
            evidence=[make_evidence(command, turn, sim_time=sim_time)],
        )

    def _on_resume(
        self,
        payload: dict[str, Any],
        command: str,
        turn: int,
        sim_time: str | None,
    ) -> list[MemoryEvent]:
        events: list[MemoryEvent] = []
        new_time = payload.get("new_sim_time") or sim_time or ""
        for wake in payload.get("wake_events") or []:
            if wake.get("type") != "task_completed":
                continue
            title = wake.get("task_title")
            if not title:
                continue
            client = wake.get("client_name")
            task_key = self._resolve(title, client)
            if task_key is None:
                continue
            self._closed.add(task_key)
            events.append(
                MemoryEvent(
                    type=EVENT_TASK_COMPLETED,
                    task_key=task_key,
                    title=title,
                    client=client or self._client_of(task_key),
                    occurred_at=new_time,
                    turn=turn,
                    payload={
                        "success": bool(wake.get("success")),
                        "funds_delta_cents": wake.get("funds_delta"),
                        "listed_reward_cents": wake.get("listed_reward"),
                        "trust_delta": wake.get("trust_delta"),
                        "deadline_margin": wake.get("deadline_margin"),
                        "margin_hours": parse_deadline_margin(
                            wake.get("deadline_margin")
                        ),
                        "employees_assigned": wake.get("employees_assigned"),
                        "salary_bump_total_cents": wake.get(
                            "salary_bump_total_cents"
                        ),
                        "task_uuid": wake.get("task_id"),
                    },
                    evidence=[
                        make_evidence(command, turn, sim_time=new_time)
                    ],
                )
            )
        return events

    # -- extractor-issued reads -------------------------------------------

    def observe_followup(
        self,
        command: str,
        result: Any,
        *,
        turn: int,
        sim_time: str | None = None,
    ) -> MemoryEvent | None:
        """Ingest the result of a follow-up read the extractor issued itself."""
        tokens = parse_command(command)
        payload = parse_tool_result(result)
        if payload is None or len(tokens) < 2:
            return None
        if (tokens[0].lower(), tokens[1].lower()) == ("task", "inspect"):
            return self._on_inspect(
                payload, command, turn, sim_time, actor=ACTOR_EXTRACTOR
            )
        return None


__all__ = [
    "StructuredExtractor",
    "flag",
    "parse_command",
    "parse_deadline_margin",
    "parse_tool_result",
]
