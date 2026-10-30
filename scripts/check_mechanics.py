"""Offline audit of the mechanisms the system prompt describes.

Nothing here calls a model or a database. It answers two questions that used
to cost a full run to discover:

1. What does a given staffing policy actually produce per business day?
   Uses the real `apply_progress_window` from yc_bench.core.progress, so the
   numbers are the simulator's own, not a reimplementation.

2. Does SYSTEM_PROMPT state those mechanisms *correctly*? The prompt is allowed
   to stay silent about a mechanic — variant (ii) exists precisely so the agent
   discovers client behaviour from its own history, and a prompt that spells
   those out would hand the memory module its answers. What is never allowed is
   saying something the simulator contradicts, or leaking the very pattern the
   memory variant is supposed to surface. Both fail here instead of in a $0.35
   run.

Usage:
    uv run python scripts/check_mechanics.py            # table + audit
    uv run python scripts/check_mechanics.py --sim-only
    uv run python scripts/check_mechanics.py --audit-only
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from decimal import Decimal

from yc_bench.agent.prompt import SYSTEM_PROMPT, build_initial_user_prompt
from yc_bench.core import progress as prog
from yc_bench.core.progress import (
    AssignmentState,
    EmployeeRateState,
    RequirementState,
    TaskProgressState,
    apply_progress_window,
)

# ---------------------------------------------------------------------------
# A representative company, built from the `default` preset.
# 8 employees, tier shares 50/35/15, salaries and rates at tier midpoints.
# ---------------------------------------------------------------------------
JUNIOR, MID, SENIOR = "junior", "mid", "senior"

# (employee_id, tier, base rate units/hour, salary cents/month)
ROSTER = [
    ("Emp_1", JUNIOR, 2.5, 300_000),
    ("Emp_2", JUNIOR, 2.5, 300_000),
    ("Emp_3", JUNIOR, 2.5, 300_000),
    ("Emp_4", JUNIOR, 2.5, 300_000),
    ("Emp_5", MID, 5.5, 700_000),
    ("Emp_6", MID, 5.5, 700_000),
    ("Emp_7", MID, 5.5, 700_000),
    ("Emp_8", SENIOR, 8.5, 1_250_000),
]

WORK_HOURS_PER_DAY = 9
DEADLINE_QTY_PER_DAY = 150
DEADLINE_MIN_BIZ_DAYS = 7
EMPLOYEES = 8

START = datetime(2025, 3, 3, 9, 0)   # Monday 09:00
END = datetime(2025, 3, 3, 18, 0)    # same day 18:00 -> exactly 9 business hours


def _rates():
    return [
        EmployeeRateState(employee_id=eid, domain="ml", rate_domain_per_hour=Decimal(str(r)))
        for eid, _tier, r, _s in ROSTER
    ]


def _salaries(eids):
    by_id = {eid: s for eid, _t, _r, s in ROSTER}
    return sum(by_id[e] for e in eids)


def day_of_work(tasks: dict[str, Decimal], assign: dict[str, list[str]]):
    """Run one business day. tasks maps task_id -> required qty.

    assign maps task_id -> employee ids. Returns (per-task delta, total delta).
    """
    task_states = [
        TaskProgressState(
            task_id=tid,
            status="active",
            requirements=(
                RequirementState(domain="ml", required_qty=qty, completed_qty=Decimal("0")),
            ),
        )
        for tid, qty in tasks.items()
    ]
    assignments = [
        AssignmentState(task_id=tid, employee_id=eid)
        for tid, eids in assign.items()
        for eid in eids
    ]
    _updated, deltas, _summaries = apply_progress_window(
        tasks=task_states,
        assignments=assignments,
        rates=_rates(),
        t0=START,
        t1=END,
    )
    per_task = {}
    for d in deltas:
        per_task[d.task_id] = per_task.get(d.task_id, Decimal("0")) + d.delta_qty
    return per_task, sum(per_task.values(), Decimal("0"))


def simulate():
    print("=" * 78)
    print("PART 1 — what each staffing policy yields per business day")
    print(f"  roster: {EMPLOYEES} employees, {WORK_HOURS_PER_DAY}h/day, 1 domain")
    print(f"  efficient team size = {prog._EFFICIENT_TEAM_SIZE}, "
          f"overcrowd factor = {prog._OVERCROWD_PENALTY}")
    print("=" * 78)

    eids = [e for e, *_ in ROSTER]
    policies = [
        ("4 slowest, 1 task",          {"T1": [e for e in sorted(eids, key=_rate_of)[:4]]}),
        ("4 fastest, 1 task",          {"T1": [e for e in sorted(eids, key=_rate_of)[-4:]]}),
        ("6 people, 1 task",           {"T1": eids[:6]}),
        ("8 people, 1 task",           {"T1": eids}),
        ("8 split best4 + other4, 2 tasks", {"T1": eids[:4], "T2": eids[4:]}),
        ("8 people on BOTH of 2 tasks", {"T1": eids, "T2": eids}),
        ("best4 + 2+2 across 3 tasks", {"T1": eids[:4], "T2": eids[4:6], "T3": eids[6:]}),
    ]

    print(f"\n{'policy':38s} {'per task /day':>16s} {'total/day':>10s} {'payroll/day':>12s}")
    print("-" * 78)
    baseline_8 = None
    for label, assign in policies:
        per_task, total = day_of_work({tid: Decimal("100000") for tid in assign}, assign)
        working = sorted({e for v in assign.values() for e in v})
        pay = _salaries(working) / 30
        detail = ", ".join(f"{v:.0f}" for v in per_task.values())
        print(f"{label:38s} {detail:>16s} {total:>10.0f} {pay:>12,.0f}")
        if label.startswith("8 people, 1 task"):
            baseline_8 = total

    print("\n  Key invariants (asserted, not just printed):")

    # Which 4 contribute matters: the cap keeps the four FASTEST on the task.
    best4 = sorted(eids, key=lambda e: -_rate_of(e))[:4]
    worst4 = sorted(eids, key=lambda e: _rate_of(e))[:4]

    eight = day_of_work({"T1": Decimal("100000")}, {"T1": eids})[1]
    top4 = day_of_work({"T1": Decimal("100000")}, {"T1": best4})[1]
    low4 = day_of_work({"T1": Decimal("100000")}, {"T1": worst4})[1]

    assert four_equals(eight, top4), "all 8 must equal the best 4 alone"
    print(f"    PASS  all 8 on one task ({eight:.0f}/day) == best 4 alone "
          f"({top4:.0f}/day)  -> the other 4 are paid and produce nothing")
    assert low4 < top4, "the 4 slowest must be worse than the 4 fastest"
    print(f"    PASS  slowest 4 alone = {low4:.0f}/day, i.e. WHICH four is "
          f"assigned matters more than how many")

    # Two tasks, disjoint groups: each task keeps its own top-4.
    two = day_of_work({"T1": Decimal("100000"), "T2": Decimal("100000")},
                      {"T1": best4, "T2": [e for e in eids if e not in best4]})[1]
    assert two > eight, "disjoint 4+4 must beat stacking everyone on one task"
    print(f"    PASS  best4 + the remaining 4 on two tasks = {two:.0f}/day vs "
          f"{eight:.0f}/day on one ({two / eight:.2f}x)")

    stacked = day_of_work({"T1": Decimal("100000"), "T2": Decimal("100000")},
                          {"T1": eids, "T2": eids})[1]
    assert stacked < two, "assigning everyone to both tasks must be worse"
    print(f"    PASS  everyone-on-both = {stacked:.0f}/day < 4+4 = {two:.0f}/day "
          f"(rate split by concurrency)")

    # Deadline feasibility for the preset's task sizes.
    best4 = sorted(eids, key=lambda e: -_rate_of(e))[:4]
    per_day = day_of_work({"T1": Decimal("100000")}, {"T1": best4})[1]
    print("\n  Deadline feasibility (deadline = max(7, qty/150) business days;")
    print(f"  best 4 employees deliver {per_day:.0f} qty/day):")
    print(f"    {'qty':>5s} {'days allowed':>13s} {'needed/day':>11s} {'days needed':>12s} {'fits?':>6s}")
    for qty in (400, 800, 1200, 1500, 3000, 6000):
        days_allowed = max(DEADLINE_MIN_BIZ_DAYS, int(qty / DEADLINE_QTY_PER_DAY))
        need = Decimal(qty) / days_allowed
        days_needed = Decimal(qty) / per_day
        fits = "yes" if days_needed <= days_allowed else "NO"
        print(f"    {qty:>5d} {days_allowed:>13d} {need:>11.0f} {days_needed:>12.1f} {fits:>6s}")

    print("    (the deadline is computed from the ADVERTISED quantity, so a task")
    print("     that inflates on accept cannot be rescued by working faster — PART 3)")


def rat_feasibility():
    """Scope creep sets a deadline the inflated workload can never meet."""
    from yc_bench.config import get_world_config

    wc = get_world_config()
    inflation = max(3.0, wc.scope_creep_max * 1.0)  # intensity is |loyalty|, ~1..3
    hours = wc.workday_end_hour - wc.workday_start_hour
    # Absolute ceiling: 4 contributors at the skill cap of 10 units/hour.
    ceil_per_day = prog._EFFICIENT_TEAM_SIZE * 10.0 * hours
    eids = [e for e, *_ in ROSTER]
    roi = day_of_work(
        {"T1": Decimal("100000")}, {"T1": sorted(eids, key=lambda e: -_rate_of(e))[:4]}
    )[1]

    print("\n" + "=" * 78)
    print("PART 3 — a scope-creeped task vs its own deadline")
    print("=" * 78)
    print(f"  real qty = advertised x (1 + {inflation:.1f})  "
          f"[scope_creep_max={wc.scope_creep_max}, floor 3.0]")
    print(f"  deadline = max({wc.deadline_min_biz_days}, advertised/"
          f"{int(wc.deadline_qty_per_day)}) business days")
    print(f"  capacity: best 4 of this roster = {roi:.0f}/day; absolute ceiling "
          f"(4 x 10/h x {hours:.0f}h) = {ceil_per_day:.0f}/day")

    print(f"\n  {'advertised':>10s} {'days':>5s} {'real qty':>9s} "
          f"{'needed/day':>11s} {'possible?':>22s}")
    first_impossible = None
    for adv in (200, 400, 630, 800, 1200, 2000):
        days = max(wc.deadline_min_biz_days, int(adv / wc.deadline_qty_per_day))
        real = adv * (1 + inflation)
        need = real / days
        if need <= roi:
            verdict = "yes"
        elif need <= ceil_per_day:
            verdict = "only with a maxed roster"
        else:
            verdict = "NO — impossible"
            if first_impossible is None:
                first_impossible = adv
        print(f"  {adv:>10d} {days:>5d} {real:>9.0f} {need:>11.0f} {verdict:>22s}")

    required_rate = (1 + inflation) * wc.deadline_qty_per_day
    assert ceil_per_day < required_rate, (
        "past the 7-day floor the required rate must exceed even a maxed roster"
    )
    print(f"\n  PASS  past the 7-day floor a creeped task demands "
          f"{required_rate:.0f}/day; a maxed roster tops out at "
          f"{ceil_per_day:.0f}/day")
    print(f"  =>  from about {first_impossible:.0f} advertised qty up, accepting "
          f"an inflating client is a certain failure, not a risky bet.")

    reward_cents = 500_000  # mode of world.dist.reward_funds_cents
    cash = reward_cents * wc.penalty_fail_funds_pct
    delta = 0.175  # mode-ish of reward_prestige_delta (beta 1.2/2.8, scale .35)
    print(f"\n  cost of that certain failure (modal ${reward_cents/100:,.0f} reward):")
    print(f"    cash clawback    ${cash/100:>8,.0f}  "
          f"({wc.penalty_fail_funds_pct:.0%} of advertised reward)")
    print(f"    prestige         -{wc.penalty_fail_multiplier * delta:.3f}")
    print(f"    payroll          the days the team was tied up")
    print(f"  cancelling instead: prestige -{wc.penalty_cancel_multiplier * delta:.3f} "
          f"and a trust hit; cash $0")


def four_equals(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) < Decimal("0.01")


def _rate_of(eid: str) -> float:
    for e, _tier, r, _s in ROSTER:
        if e == eid:
            return r
    raise KeyError(eid)


# ---------------------------------------------------------------------------
# PART 2 — does the prompt say what the code does?
# ---------------------------------------------------------------------------
PASS, SILENT, FAIL = "PASS", "SILENT", "FAIL"

# Phrases that would tell the agent, up front, the pattern variant (ii) is
# supposed to learn from its own event log. If any of these appear in the
# prompt, the two arms stop differing in *memory* and start differing in
# *hints*, and the comparison measures nothing.
MEMORY_LEAK_PHRASES = (
    "above the listing",
    "worse than advertised",
    "set the real requirement",
    "whose past work went badly",
    "actually took on",
    "scope creep",
    "scope-creep",
)
def _cli_commands() -> set[tuple[str, str | None]]:
    """Every (group, command) the CLI really registers, read from the Typer apps."""
    import typer

    from yc_bench.cli import app

    out: set[tuple[str, str | None]] = set()
    for group in app.registered_groups:
        name = group.name
        sub = group.typer_instance
        if sub is None:
            continue
        for cmd in sub.registered_commands:
            out.add((name, cmd.name))
    for cmd in app.registered_commands:
        out.add((cmd.name, None))
    return out


def _commands_in(text: str) -> list[tuple[str, str | None]]:
    """Pull `yc-bench <group> <cmd>` references out of prompt text."""
    import re

    found = []
    for m in re.finditer(r"`?yc-bench\s+([a-z][a-z-]*)(?:\s+([a-z][a-z-]*))?", text):
        group, cmd = m.group(1), m.group(2)
        # `yc-bench task accept --task-id` -> ("task", "accept"); `yc-bench --help`
        # style flags are ignored.
        if cmd is not None and cmd.startswith("--"):
            cmd = None
        found.append((group, cmd))
    return found


# A claim about a number is checked against the constant, not against memory of
# what we wrote. Silence is a pass: see the module docstring.
def audit():
    print("\n" + "=" * 78)
    print("PART 2 — prompt vs code")
    print("=" * 78)

    initial = build_initial_user_prompt(
        sim_time="2025-01-01T09:00:00",
        horizon_end="2026-01-01T09:00:00",
        funds_cents=20_000_000,
        active_tasks=0,
        planned_tasks=0,
        employee_count=EMPLOYEES,
        monthly_payroll_cents=4_650_000,
        bankrupt=False,
    )
    combined = SYSTEM_PROMPT + "\n" + initial
    # Whitespace-normalised so that a phrase broken across a wrapped line still
    # matches: the prose wrapping is not part of the claim.
    sp = " ".join(SYSTEM_PROMPT.split())
    low = " ".join(combined.lower().split())

    checks: list[tuple[str, str, str]] = []

    from yc_bench.config import get_world_config

    wc = get_world_config()

    def claim(cid: str, triggers: tuple[str, ...], ok: bool, detail: str) -> None:
        """Check a statement only if the prompt actually makes one.

        `triggers` are the phrases that indicate the prompt is talking about
        this mechanic at all. If none is present the prompt is silent, which is
        a legitimate choice for this experiment, so the result is SILENT rather
        than a failure.
        """
        if not any(t in sp for t in triggers):
            checks.append((cid, SILENT, f"{detail} — prompt stays silent"))
            return
        checks.append((cid, PASS if ok else FAIL, detail))

    def must(cid: str, ok: bool, detail: str) -> None:
        """Unconditional check: this must hold whether or not the prompt talks."""
        checks.append((cid, PASS if ok else FAIL, detail))

    # 1. team cap — if stated, the number must be the code's number
    claim(
        "team-cap",
        (f"{prog._EFFICIENT_TEAM_SIZE} fastest", "team size"),
        f"{prog._EFFICIENT_TEAM_SIZE} fastest" in sp
        and prog._EFFICIENT_TEAM_SIZE == 4,
        f"code caps a task's full-rate contributors at {prog._EFFICIENT_TEAM_SIZE}",
    )
    # 2. overcrowd penalty is exactly zero, so "contribute nothing" is correct
    claim(
        "overcrowd-zero",
        ("contribute nothing", "5th"),
        prog._OVERCROWD_PENALTY == 0 and "contribute nothing" in sp,
        f"_OVERCROWD_PENALTY = {prog._OVERCROWD_PENALTY}: 5th+ adds zero output",
    )
    # 3. rate split by concurrency — upstream writes `rate/N`, the code divides by k
    claim(
        "rate-split",
        ("split their rate",),
        "rate/k" in sp or "rate/n" in sp.lower(),
        "_effective_rate_for_task_domain divides each contribution by k",
    )
    # 4. if the payoff of a second task is described, it must be described right.
    #    Only "other four" means the prompt is talking about who staffs task two;
    #    "dispatch a second task" is the upstream concurrency nudge, not a claim.
    claim(
        "parallel-payoff",
        ("other four",),
        "second task with the other four" in sp,
        "disjoint 4+4 beats stacking all 8 on one task",
    )
    # 5. no false claim that extra offers are always harmful (unconditional)
    must(
        "no-false-parallel-claim",
        "does not make you faster" not in sp,
        "that claim is false: a second task staffed with the other four adds output",
    )
    # 6. fail cost — any cash clawback figure stated must be the configured one
    claim(
        "fail-cost",
        (f"{int(round(wc.penalty_fail_funds_pct * 100))}%", "charged back"),
        f"{int(round(wc.penalty_fail_funds_pct * 100))}%" in sp,
        f"penalty_fail_funds_pct = {wc.penalty_fail_funds_pct} cash clawback on failure",
    )
    # 7. cancellation costs prestige but no cash
    claim(
        "cancel-cost",
        (f"{wc.penalty_cancel_multiplier}x", "no cash"),
        f"{wc.penalty_cancel_multiplier}x" in sp and "no cash" in sp,
        f"cancel = {wc.penalty_cancel_multiplier}x prestige delta, no cash penalty",
    )
    # 8. deadline formula, including the inclusive accept-day rule
    claim(
        "deadline-rate",
        ("max(7,", "advertised qty /"),
        f"max({wc.deadline_min_biz_days}, advertised qty / "
        f"{int(wc.deadline_qty_per_day)})" in sp
        and "counting the day you accept as day 1" in sp,
        f"deadline = end of business day max({wc.deadline_min_biz_days}, qty/"
        f"{int(wc.deadline_qty_per_day)}), accept day counted as day 1",
    )
    # 9. trust work reduction — stated upstream, so this one is live
    claim(
        "trust-reduction",
        ("less work per task",),
        "less work per task" in sp,
        f"trust_work_reduction_max = {wc.trust_work_reduction_max}",
    )
    # 10. prestige raises quantity as well as reward
    claim(
        "prestige-qty",
        ("required quantity", "raises both the reward"),
        wc.prestige_qty_scale > 0 and "required quantity" in sp,
        f"prestige_qty_scale = {wc.prestige_qty_scale} makes tasks heavier, not just richer",
    )
    # 11. every command named in the prompt must exist in the CLI (unconditional)
    real = _cli_commands()
    named = _commands_in(combined)
    unknown = sorted({c for c in named if c not in real})
    must(
        "commands-exist",
        not unknown,
        f"{len(named)} command references, unknown: "
        f"{unknown if unknown else 'none'}",
    )
    # 12. no reference to a disabled command anywhere it is read (unconditional)
    must(
        "no-assign-all",
        "assign-all" not in combined,
        "`task assign-all` is disabled and returns an error",
    )
    # 13. resume gate described as a rejection, not as something that skips time
    must(
        "resume-gate",
        "rejected" in low and "zero revenue" not in sp,
        "sim resume is refused when idle; it does not skip forward",
    )
    # 14. the prompt must not hand the memory variant its conclusions
    leaked = [p for p in MEMORY_LEAK_PHRASES if p in sp]
    must(
        "no-memory-leak",
        not leaked,
        "prompt does not pre-state the client-inflation pattern that (ii) is "
        f"meant to learn from history; leaked: {leaked if leaked else 'none'}",
    )

    width = max(len(c[0]) for c in checks)
    for cid, state, detail in checks:
        print(f"  [{state:<6}] {cid:<{width}}  {detail}")

    failed = [c for c in checks if c[1] == FAIL]
    silent = [c for c in checks if c[1] == SILENT]
    print(
        f"\n  {len(checks) - len(failed) - len(silent)} checked and correct, "
        f"{len(silent)} silent (allowed), {len(failed)} contradicts the code"
    )
    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-only", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    if not args.audit_only:
        simulate()
        rat_feasibility()
    failed = []
    if not args.sim_only:
        failed = audit()

    if failed:
        print("\nRESULT: the prompt disagrees with the simulator in "
              f"{len(failed)} place(s): {', '.join(c[0] for c in failed)}")
        return 1
    print("\nRESULT: prompt and simulator agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
