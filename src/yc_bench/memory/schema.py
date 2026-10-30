"""Event schema for the structured-memory variant (plan variant (ii)).

Design constraints encoded here (see proposal, "Proposed methodology"):

* Records are derived **only** from observations the agent itself could have
  issued: CLI stdout of its own commands and ``sim resume`` wake events. No
  hidden simulator attributes (client loyalty, rat flags, future outcomes).
* The log is **append-only**: an event is never revised or deleted. Turning
  observations into corrected/merged records is variant (iii), not (ii).
* Every event carries ``evidence`` so any line rendered into the prompt can be
  traced back to a concrete command, turn and simulation timestamp.

Event vocabulary
----------------
``task_accepted``   observed from ``task accept`` (+ market browse for the
                    advertised workload). Carries the advertised vs. accepted
                    workload, which is what makes scope creep measurable.
``task_inspected``  observed from ``task inspect``. Used to backfill the real
                    workload / client when the agent inspected on its own, or
                    when the extractor issued the read itself.
``task_assigned``   observed from ``task assign``.
``task_completed``  observed from a ``sim resume`` wake event.
``task_cancelled``  observed from ``task cancel``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = 1

EVENT_TASK_ACCEPTED = "task_accepted"
EVENT_TASK_INSPECTED = "task_inspected"
EVENT_TASK_ASSIGNED = "task_assigned"
EVENT_TASK_COMPLETED = "task_completed"
EVENT_TASK_CANCELLED = "task_cancelled"

EventType = Literal[
    "task_accepted",
    "task_inspected",
    "task_assigned",
    "task_completed",
    "task_cancelled",
]

ALL_EVENT_TYPES = (
    EVENT_TASK_ACCEPTED,
    EVENT_TASK_INSPECTED,
    EVENT_TASK_ASSIGNED,
    EVENT_TASK_COMPLETED,
    EVENT_TASK_CANCELLED,
)

# Who produced the observation. "agent" = the agent's own command in this turn.
# "extractor" = a read the memory module issued itself, using exactly the same
# read-only command the agent has access to.
ACTOR_AGENT = "agent"
ACTOR_EXTRACTOR = "extractor"

# Outcome vocabulary used by the derived record model.
OUTCOME_IN_FLIGHT = "in_flight"
OUTCOME_SUCCESS = "success"
OUTCOME_FAIL = "fail"
OUTCOME_CANCELLED = "cancelled"

_REQUIRED_PAYLOAD_FIELDS: dict[str, tuple[str, ...]] = {
    EVENT_TASK_ACCEPTED: ("accepted_at",),
    EVENT_TASK_INSPECTED: ("observed_at",),
    EVENT_TASK_ASSIGNED: ("employees",),
    EVENT_TASK_COMPLETED: ("success",),
    EVENT_TASK_CANCELLED: (),
}


def make_task_key(title: str, accepted_at: str | None) -> str:
    """Stable key for one accepted task.

    Task titles are reused across market generations (a replacement task may
    take the same title later), so the title alone is not unique. The accept
    timestamp is the discriminator the agent can actually observe.
    """
    return f"{title}@{accepted_at or 'unknown'}"


@dataclass
class MemoryEvent:
    """One append-only observation about a task."""

    type: EventType
    task_key: str
    title: str
    client: str | None
    occurred_at: str
    turn: int
    payload: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    v: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": self.v,
            "type": self.type,
            "task_key": self.task_key,
            "title": self.title,
            "client": self.client,
            "occurred_at": self.occurred_at,
            "turn": self.turn,
            "payload": self.payload,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MemoryEvent":
        return cls(
            type=raw["type"],
            task_key=raw["task_key"],
            title=raw.get("title") or "",
            client=raw.get("client"),
            occurred_at=raw.get("occurred_at") or "",
            turn=int(raw.get("turn") or 0),
            payload=raw.get("payload") or {},
            evidence=raw.get("evidence") or [],
            v=int(raw.get("v") or SCHEMA_VERSION),
        )


def validate_event(ev: MemoryEvent) -> None:
    """Fail fast on malformed events instead of silently rendering garbage."""
    if ev.type not in ALL_EVENT_TYPES:
        raise ValueError(f"unknown event type: {ev.type!r}")
    if not ev.task_key:
        raise ValueError("event has empty task_key")
    if not ev.occurred_at:
        raise ValueError(f"event {ev.type}/{ev.task_key} has empty occurred_at")
    for name in _REQUIRED_PAYLOAD_FIELDS[ev.type]:
        if name not in ev.payload:
            raise ValueError(
                f"event {ev.type}/{ev.task_key} missing payload field {name!r}"
            )


def make_evidence(
    command: str,
    turn: int,
    *,
    sim_time: str | None = None,
    actor: str = ACTOR_AGENT,
) -> dict[str, Any]:
    return {
        "cmd": command,
        "turn": turn,
        "sim_time": sim_time,
        "actor": actor,
    }


__all__ = [
    "ACTOR_AGENT",
    "ACTOR_EXTRACTOR",
    "ALL_EVENT_TYPES",
    "EVENT_TASK_ACCEPTED",
    "EVENT_TASK_ASSIGNED",
    "EVENT_TASK_CANCELLED",
    "EVENT_TASK_COMPLETED",
    "EVENT_TASK_INSPECTED",
    "EventType",
    "MemoryEvent",
    "OUTCOME_CANCELLED",
    "OUTCOME_FAIL",
    "OUTCOME_IN_FLIGHT",
    "OUTCOME_SUCCESS",
    "SCHEMA_VERSION",
    "make_evidence",
    "make_task_key",
    "validate_event",
]
