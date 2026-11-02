"""Read side of structured memory: fold events into records, render a view.

Everything here is a **pure function of the append-only event log**. Aggregate
values are recomputed on every call and never persisted, so variant (ii) has no
state to keep consistent: same events in, same text out.

What the rendered view deliberately does *not* do (these belong to later
variants in the plan):

* it never revises or merges records  -> variant (iii);
* it never scores relevance or builds a decision-specific view -> variant (iv);
* it never renders mutable live values (client trust, salaries), because a
  stale-value correction is exactly the capability (iii) is supposed to add.

What it *does* do is arithmetic only. ``0 ok / 3 fail`` is a count; "this client
is adversarial" is a judgement and is left to the model, so that a later
"memory was right but the agent ignored it" case can be attributed to acting
rather than to retrieval.

Two things that look like judgement but are not: the ``qty on accept`` ratio is
a division, and a cancellation reason is quoted verbatim from the agent's own
``task cancel --reason`` text. Both are observations the agent made; neither is
a conclusion drawn for it. The legend that names the ratio is a unit label, not
advice — an unlabelled ``x4.0`` invites the model to invent a meaning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from .schema import (
    EVENT_TASK_ACCEPTED,
    EVENT_TASK_ASSIGNED,
    EVENT_TASK_CANCELLED,
    EVENT_TASK_COMPLETED,
    EVENT_TASK_INSPECTED,
    OUTCOME_CANCELLED,
    OUTCOME_FAIL,
    OUTCOME_IN_FLIGHT,
    OUTCOME_SUCCESS,
    MemoryEvent,
)

CLIENT_VIEW_HEADER = (
    "### Client memory (recorded automatically from your own past commands)"
)
STAFF_VIEW_HEADER = "### Staff memory (auto-recorded from your own commands)"

# Legend for the workload field. The number is useless without its unit: a bare
# "qty x4.0" can be read as reward, task count or a rating. Naming the ratio is
# not a hint about what to do with it — it is what the field measures.
CLIENT_VIEW_LEGEND = (
    "  qty on accept = accepted workload / listed quantity (x1.0 = as listed)"
)

# Characters-per-token used for budget accounting. Deliberately crude and
# identical across variants: the budget is a comparison device, not a meter.
CHARS_PER_TOKEN = 4

MAX_LINE_CHARS = 160
MAX_CLIENT_NAME_CHARS = 24
MAX_REASON_CHARS = 32


# ---------------------------------------------------------------------------
# Derived models
# ---------------------------------------------------------------------------


@dataclass
class TaskRecord:
    """One accepted task, folded from its events."""

    task_key: str
    title: str
    client: str | None = None
    accepted_at: str = ""
    deadline: str | None = None
    advertised_qty: dict[str, float] = field(default_factory=dict)
    observed_qty: dict[str, float] = field(default_factory=dict)
    advertised_reward_cents: int | None = None
    employees: list[str] = field(default_factory=list)
    outcome: str = OUTCOME_IN_FLIGHT
    completed_at: str | None = None
    margin_hours: float | None = None
    funds_delta_cents: int | None = None
    cancel_reason: str | None = None
    last_activity: str = ""
    event_count: int = 0

    @property
    def advertised_max(self) -> float | None:
        """Largest listed quantity across domains (listings price per domain)."""
        return max(self.advertised_qty.values()) if self.advertised_qty else None

    @property
    def observed_max(self) -> float | None:
        """Largest accepted quantity across domains, as reported by inspect."""
        return max(self.observed_qty.values()) if self.observed_qty else None

    @property
    def creep_ratio(self) -> float | None:
        """Accepted workload / advertised workload, when both sides are known."""
        adv, obs = self.advertised_max, self.observed_max
        if not adv or not obs or adv <= 0:
            return None
        return round(obs / adv, 1)

    @property
    def creep_source(self) -> str:
        """Which observations the ratio rests on, so its weight is visible."""
        if self.advertised_max and self.observed_max:
            return "browse+inspect"
        if self.observed_max:
            return "inspect_only"
        return "none"


@dataclass
class ClientStat:
    """Per-client tallies folded from every accepted task of that client."""

    client: str
    accepted: int = 0
    ok: int = 0
    fail: int = 0
    cancelled: int = 0
    in_flight: int = 0
    creep_ratios: list[float] = field(default_factory=list)
    last_activity: str = ""
    last_outcome: str = ""
    last_margin_hours: float | None = None
    last_cancel_reason: str | None = None

    @property
    def creep_mean(self) -> float | None:
        """Average accepted/listed ratio, or None if nothing was measured.

        A plain mean, not a recency-weighted one: revising a stale figure is the
        capability variant (iii) adds, so doing it here would blur the two.
        """
        if not self.creep_ratios:
            return None
        return round(sum(self.creep_ratios) / len(self.creep_ratios), 1)

    def check_invariant(self) -> None:
        """Assert every accepted task falls into exactly one outcome bucket."""
        total = self.ok + self.fail + self.cancelled + self.in_flight
        if total != self.accepted:
            raise AssertionError(
                f"client {self.client}: ok+fail+cancel+in_flight={total} "
                f"!= accepted={self.accepted}"
            )


@dataclass
class StaffStat:
    """Per-employee tallies folded from the tasks they were assigned to."""

    employee: str
    tasks: int = 0
    ok: int = 0
    fail: int = 0
    in_flight: int = 0
    margins: list[float] = field(default_factory=list)
    domains: dict[str, int] = field(default_factory=dict)

    @property
    def margin_mean(self) -> float | None:
        """Mean deadline margin in hours; negative means late on average."""
        if not self.margins:
            return None
        return round(sum(self.margins) / len(self.margins), 1)


# ---------------------------------------------------------------------------
# Folding events -> records
# ---------------------------------------------------------------------------


def build_records(events: Iterable[MemoryEvent]) -> dict[str, TaskRecord]:
    """Fold the event log into one record per accepted task."""
    records: dict[str, TaskRecord] = {}
    ordered = sorted(events, key=lambda e: (e.occurred_at, e.turn))

    for ev in ordered:
        if ev.type == EVENT_TASK_ACCEPTED:
            rec = records.get(ev.task_key)
            if rec is None:
                rec = TaskRecord(task_key=ev.task_key, title=ev.title)
                records[ev.task_key] = rec
            rec.client = rec.client or ev.client
            rec.accepted_at = ev.payload.get("accepted_at") or ev.occurred_at
            rec.deadline = ev.payload.get("deadline")
            rec.advertised_qty = dict(ev.payload.get("advertised_qty") or {})
            rec.advertised_reward_cents = ev.payload.get("advertised_reward_cents")
            rec.last_activity = max(rec.last_activity, ev.occurred_at)
            rec.event_count += 1
            continue

        rec = records.get(ev.task_key)
        if rec is None:
            # An observation about a task we never saw accepted (e.g. memory
            # enabled mid-run). Skip rather than invent a record.
            continue

        if ev.type == EVENT_TASK_INSPECTED:
            # Earliest observation after accept wins: required_qty is fixed at
            # accept time, later reads only add progress noise.
            if not rec.observed_qty:
                rec.observed_qty = dict(ev.payload.get("qty") or {})
            rec.client = rec.client or ev.payload.get("client") or ev.client
            rec.deadline = rec.deadline or ev.payload.get("deadline")
        elif ev.type == EVENT_TASK_ASSIGNED:
            rec.employees = list(ev.payload.get("employees") or []) or rec.employees
        elif ev.type == EVENT_TASK_COMPLETED:
            rec.outcome = (
                OUTCOME_SUCCESS if ev.payload.get("success") else OUTCOME_FAIL
            )
            rec.completed_at = ev.occurred_at
            rec.margin_hours = ev.payload.get("margin_hours")
            rec.funds_delta_cents = ev.payload.get("funds_delta_cents")
        elif ev.type == EVENT_TASK_CANCELLED:
            rec.outcome = OUTCOME_CANCELLED
            rec.completed_at = ev.occurred_at
            rec.cancel_reason = ev.payload.get("reason") or rec.cancel_reason

        rec.last_activity = max(rec.last_activity, ev.occurred_at)
        rec.event_count += 1

    return records


def client_stats(records: dict[str, TaskRecord]) -> list[ClientStat]:
    """Aggregate task records per client, most recently active first."""
    stats: dict[str, ClientStat] = {}
    for rec in records.values():
        if not rec.client:
            continue
        stat = stats.setdefault(rec.client, ClientStat(client=rec.client))
        stat.accepted += 1
        if rec.outcome == OUTCOME_SUCCESS:
            stat.ok += 1
        elif rec.outcome == OUTCOME_FAIL:
            stat.fail += 1
        elif rec.outcome == OUTCOME_CANCELLED:
            stat.cancelled += 1
        else:
            stat.in_flight += 1
        ratio = rec.creep_ratio
        if ratio is not None:
            stat.creep_ratios.append(ratio)
        if rec.last_activity >= stat.last_activity:
            stat.last_activity = rec.last_activity
            stat.last_outcome = rec.outcome
            stat.last_margin_hours = rec.margin_hours
            stat.last_cancel_reason = rec.cancel_reason

    for stat in stats.values():
        stat.check_invariant()

    return sorted(stats.values(), key=lambda s: s.last_activity, reverse=True)


def staff_stats(records: dict[str, TaskRecord]) -> list[StaffStat]:
    """Aggregate assignments per employee, busiest first.

    One task counts once per employee assigned to it, so summing ``tasks``
    across staff gives staffing slots, not the number of jobs done.
    """
    stats: dict[str, StaffStat] = {}
    for rec in records.values():
        domain = max(rec.observed_qty, key=rec.observed_qty.get) if rec.observed_qty else None
        for name in rec.employees:
            stat = stats.setdefault(name, StaffStat(employee=name))
            stat.tasks += 1
            if rec.outcome == OUTCOME_SUCCESS:
                stat.ok += 1
            elif rec.outcome == OUTCOME_FAIL:
                stat.fail += 1
            elif rec.outcome == OUTCOME_IN_FLIGHT:
                stat.in_flight += 1
            if rec.margin_hours is not None:
                stat.margins.append(rec.margin_hours)
            if domain:
                stat.domains[domain] = stat.domains.get(domain, 0) + 1
    return sorted(
        stats.values(), key=lambda s: (s.tasks, s.ok), reverse=True
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token count, used only to enforce the view's budget.

    See ``CHARS_PER_TOKEN`` for why this is a division rather than a real
    tokeniser: the budget has to mean the same thing in every variant.
    """
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _short_date(iso: str | None) -> str:
    if not iso or len(iso) < 10:
        return "?"
    return iso[5:10]


def _margin(hours: float | None) -> str:
    if hours is None:
        return ""
    return f"late {abs(hours):.0f}h" if hours < 0 else f"ahead {hours:.0f}h"


def _shorten(text: str, limit: int) -> str:
    """Collapse whitespace and cut at a word boundary when possible."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def render_client_line(stat: ClientStat) -> str:
    """One pipe-separated client line: counts, workload ratio, last outcome."""
    name = stat.client
    if len(name) > MAX_CLIENT_NAME_CHARS:
        name = name[: MAX_CLIENT_NAME_CHARS - 1] + "…"
    parts = [
        f"{name}: {stat.accepted} acc",
        f"{stat.ok} ok {stat.fail} fail {stat.cancelled} cancel "
        f"{stat.in_flight} open",
    ]
    mean = stat.creep_mean
    if mean is not None:
        measured = len(stat.creep_ratios)
        parts.append(f"qty on accept {mean:.1f}x ({measured}/{stat.accepted})")
    last = f"last {_short_date(stat.last_activity)}"
    if stat.last_outcome == OUTCOME_CANCELLED and stat.last_cancel_reason:
        # The agent's own --reason text. Without it a cancel reads as a
        # non-event, which is exactly the wrong reading: a cancel after an
        # inflated accept is the strongest signal in the whole log.
        last += f' cancelled ("{_shorten(stat.last_cancel_reason, MAX_REASON_CHARS)}")'
    else:
        tail = _margin(stat.last_margin_hours)
        if tail:
            last += f" {tail}"
    parts.append(last)
    line = " | ".join(parts)
    return line if len(line) <= MAX_LINE_CHARS else line[: MAX_LINE_CHARS - 1] + "…"


def render_staff_line(stat: StaffStat) -> str:
    """One pipe-separated employee line: volume, top domains, mean margin."""
    parts = [f"{stat.employee}: {stat.tasks} task"]
    parts.append(f"{stat.ok} ok / {stat.fail} fail / {stat.in_flight} in flight")
    if stat.domains:
        top = sorted(stat.domains.items(), key=lambda kv: kv[1], reverse=True)[:2]
        parts.append("domains " + ", ".join(f"{d} x{n}" for d, n in top))
    mean = stat.margin_mean
    if mean is not None:
        parts.append(f"avg margin {_margin(mean)}")
    return " | ".join(parts)


def _render(
    lines: list[str], header: str, token_budget: int, legend: str | None = None
) -> str:
    if not lines:
        return ""
    kept: list[str] = []
    used = estimate_tokens(header)
    if legend:
        # Always kept: it is what makes the numbers below readable at all.
        kept.append(legend)
        used += estimate_tokens(legend) + 1
    omitted = 0
    for line in lines:
        cost = estimate_tokens(line) + 1
        if used + cost > token_budget:
            omitted += 1
            continue
        kept.append(line)
        used += cost
    if len(kept) == (1 if legend else 0):
        return ""
    if omitted:
        note = f"… {omitted} more client(s) omitted by memory budget"
        if used + estimate_tokens(note) <= token_budget:
            kept.append(note)
    return "\n".join([header, *kept])


def render_client_view(
    stats: Iterable[ClientStat], *, token_budget: int = 300
) -> str:
    """Recency-ordered client lines, truncated to a fixed token budget."""
    stats = list(stats)
    lines = [render_client_line(s) for s in stats]
    # Only explain the workload field when at least one ratio exists.
    legend = (
        CLIENT_VIEW_LEGEND
        if any(s.creep_ratios for s in stats)
        else None
    )
    return _render(lines, CLIENT_VIEW_HEADER, token_budget, legend=legend)


def render_staff_view(stats: Iterable[StaffStat], *, token_budget: int = 200) -> str:
    """Attendance-ordered staff lines, truncated to a fixed token budget."""
    lines = [render_staff_line(s) for s in stats]
    return _render(lines, STAFF_VIEW_HEADER, token_budget)


__all__ = [
    "CLIENT_VIEW_HEADER",
    "CLIENT_VIEW_LEGEND",
    "ClientStat",
    "StaffStat",
    "TaskRecord",
    "build_records",
    "client_stats",
    "estimate_tokens",
    "render_client_line",
    "render_client_view",
    "render_staff_line",
    "render_staff_view",
    "staff_stats",
]
