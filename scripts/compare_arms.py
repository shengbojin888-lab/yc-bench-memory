"""Compare two completed yc-bench runs on equal footing.

Why this exists
---------------
The runner stops on `--max-turns`, not on a simulation date. Two arms given the
same turn budget therefore end on *different* calendar days: whichever arm
wastes time (grinding on an oversized task, stacking all eight employees on one
job) advances the clock further, pays more monthly payroll, and can still show a
lower final balance for a reason that has nothing to do with the variable under
test. Comparing `final funds` across such runs is not a measurement.

Use `yc-bench run --until YYYY-MM-DD` to make both arms stop on the same sim
day. When the runs you already have stopped on different days, this script
still gives you the comparable number: funds at a common date.

Reports per arm
    stop date, sim days elapsed, funds curve, reward/payroll split,
    accepted/finished/success counts, team sizes, calendar efficiency,
    idle windows (no active task = payroll with no revenue).

Usage
    python scripts/compare_arms.py                     # auto-detect both arms
    python scripts/compare_arms.py base.db mem.db
    python scripts/compare_arms.py base.db mem.db --start 2025-01-01
"""

from __future__ import annotations

import glob
import sqlite3
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

INITIAL_FUNDS_CENTS = 20_000_000


def _pick(pattern: str) -> str | None:
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def _ts(value) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(str(value))


def load(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    state = conn.execute("select * from sim_state").fetchone()
    if state is None:
        raise SystemExit(f"{db_path}: no sim_state row")

    ledger = [
        (_ts(r["occurred_at"]), r["amount_cents"], r["category"])
        for r in conn.execute(
            "select occurred_at, amount_cents, category from ledger_entries"
        )
    ]
    ledger.sort(key=lambda row: row[0])

    curve = []
    bal = INITIAL_FUNDS_CENTS
    for when, amount, _cat in ledger:
        bal += amount
        curve.append((when, bal))

    tasks = []
    for r in conn.execute(
        """
        select t.title, c.name as client, t.status, t.success,
               t.accepted_at, t.deadline, t.completed_at, t.reward_funds_cents,
               (select required_qty from task_requirements q
                 where q.task_id = t.id limit 1) as qty,
               (select completed_qty from task_requirements q
                 where q.task_id = t.id limit 1) as done_qty,
               (select count(distinct employee_id) from task_assignments a
                 where a.task_id = t.id) as team
          from tasks t left join clients c on c.id = t.client_id
         where t.status <> 'market'
         order by t.accepted_at
        """
    ):
        tasks.append(dict(r))

    by_cat: dict[str, int] = {}
    for _when, amount, cat in ledger:
        by_cat[cat] = by_cat.get(cat, 0) + amount

    # A failed task books its clawback as a *negative* task_reward row, so the
    # category total hides it. Split the two before reporting.
    rewards = sum(a for _w, a, c in ledger if c == "task_reward" and a > 0)
    penalties = sum(a for _w, a, c in ledger if c == "task_reward" and a < 0)
    payroll = sum(a for _w, a, c in ledger if c == "monthly_payroll")

    return {
        "path": db_path,
        "stop": _ts(state["sim_time"]),
        "horizon": _ts(state["horizon_end"]),
        "funds": INITIAL_FUNDS_CENTS + sum(row[1] for row in ledger),
        "curve": curve,
        "tasks": tasks,
        "by_cat": by_cat,
        "rewards": rewards,
        "penalties": penalties,
        "payroll": payroll,
    }


def derive_start(run: dict, override: str | None) -> tuple[datetime, str]:
    if override:
        return (
            datetime.combine(date.fromisoformat(override), time(9, 0)),
            "--start",
        )
    stamps = [_ts(t["accepted_at"]) for t in run["tasks"] if t["accepted_at"]]
    stamps += [when for when, _ in run["curve"]]
    if not stamps:
        return run["stop"], "stop (no events)"
    first = min(stamps)
    return first.replace(hour=9, minute=0, second=0, microsecond=0), "earliest event"


def funds_at(curve, when: datetime) -> int:
    val = INITIAL_FUNDS_CENTS
    for stamp, bal in curve:
        if stamp <= when:
            val = bal
        else:
            break
    return val


def idle_windows(run: dict) -> list[tuple[datetime, datetime, float]]:
    """Calendar windows where no accepted task was in progress."""
    spans = []
    for t in run["tasks"]:
        start, end = _ts(t["accepted_at"]), _ts(t["completed_at"])
        if start and end:
            spans.append((start, end))
    if not spans:
        return []
    spans.sort()
    gaps = []
    cursor = spans[0][1]
    for start, end in spans[1:]:
        if start > cursor:
            gaps.append((cursor, start, (start - cursor).total_seconds() / 86400))
        cursor = max(cursor, end)
    return gaps


def concurrency(run: dict, start: datetime, stop: datetime) -> tuple[float, float]:
    """Average simultaneous tasks, and the share of days with only one running.

    With eight employees and a four-per-task cap, a single running task leaves
    four people producing nothing while still drawing salary, so "one task at a
    time" is a real loss even though the clock is not idle.
    """
    spans = []
    for t in run["tasks"]:
        a, b = _ts(t["accepted_at"]), _ts(t["completed_at"])
        if a and b:
            spans.append((max(a, start), min(b, stop)))
    spans = [(a, b) for a, b in spans if b > a]
    if not spans:
        return 0.0, 1.0

    marks = sorted({a for a, _ in spans} | {b for _, b in spans})
    area = solo = 0.0
    for left, right in zip(marks, marks[1:]):
        running = sum(1 for a, b in spans if a <= left and b >= right)
        span = (right - left).total_seconds() / 86400
        area += running * span
        if running <= 1:
            solo += span
    total = (stop - start).total_seconds() / 86400
    return (area / total if total else 0.0), (solo / total if total else 0.0)


def report(name: str, run: dict, start: datetime, start_src: str) -> dict:
    days = max((run["stop"] - start).total_seconds() / 86400, 0.0)
    tasks = run["tasks"]
    done = [t for t in tasks if str(t["status"]).startswith("completed")]
    ok = [t for t in done if t["success"] == 1]
    reward = run["rewards"]
    payroll = run["payroll"]
    teams = [t["team"] for t in tasks]
    off_cap = sum(1 for n in teams if n != 4)
    gaps = idle_windows(run)
    avg_par, solo_share = concurrency(run, start, run["stop"])

    print(f"\n{'=' * 76}")
    print(f"{name}")
    print(f"{'=' * 76}")
    print(f"  db                  {Path(run['path']).name}")
    print(f"  start -> stop       {start.date()} -> {run['stop'].date()}"
          f"   ({days:.0f} sim days, start from {start_src})")
    print(f"  final funds         ${run['funds'] / 100:,.0f}")
    print(f"  reward / payroll    ${reward / 100:,.0f}  /  ${-payroll / 100:,.0f}"
          f"   (fail clawback ${-run['penalties'] / 100:,.0f})")
    print(f"  accepted / finished {len(tasks)} / {len(done)}"
          f"   ({len(ok)} succeeded, {len(done) - len(ok)} failed)")
    print(f"  net per sim day     ${(run['funds'] - INITIAL_FUNDS_CENTS) / 100 / days:,.0f}")
    print(f"  reward per sim day  ${reward / 100 / days:,.0f}")
    print(f"  team sizes          {teams}   ({off_cap} of {len(teams)} != 4)")
    print(f"  avg team            {sum(teams) / len(teams):.2f}" if teams else "  avg team  n/a")
    if gaps:
        print(f"  idle windows        {len(gaps)}, {sum(g[2] for g in gaps):.0f} days total")
        for a, b, d in gaps:
            print(f"      {a.date()} -> {b.date()}   {d:.0f} days with nothing running")
    else:
        print("  idle windows        none")
    print(f"  tasks in flight     {avg_par:.2f} avg"
          f"   ({solo_share * 100:.0f}% of days had <=1 task running)")

    return {
        "name": name,
        "stop": run["stop"],
        "funds": run["funds"],
        "days": days,
        "net_per_day": (run["funds"] - INITIAL_FUNDS_CENTS) / 100 / days,
        "reward_per_day": reward / 100 / days,
        "accepted": len(tasks),
        "done": len(done),
        "off_cap": off_cap,
        "avg_team": sum(teams) / len(teams) if teams else 0.0,
        "idle_days": sum(g[2] for g in gaps),
        "avg_par": avg_par,
        "solo_share": solo_share,
        "curve": run["curve"],
    }


def main(argv: list[str]) -> int:
    override = None
    if "--start" in argv:
        i = argv.index("--start")
        override = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    if len(argv) >= 2:
        base_path, mem_path = argv[0], argv[1]
    else:
        base_path = _pick("db/debug_gemini_[0-9]*.db")
        mem_path = _pick("db/debug_gemini_memory_[0-9]*.db")
    if not base_path or not mem_path:
        print("could not find both arm databases; pass two paths", file=sys.stderr)
        return 2

    base_run, mem_run = load(base_path), load(mem_path)
    base_start, base_src = derive_start(base_run, override)
    mem_start, mem_src = derive_start(mem_run, override)

    b = report("BASE  (no memory)", base_run, base_start, base_src)
    m = report("MEM   (structured memory)", mem_run, mem_start, mem_src)

    print(f"\n{'=' * 76}")
    print("HEAD TO HEAD")
    print(f"{'=' * 76}")
    print(f"  {'':<24}{'BASE':>26}{'MEM':>26}")
    rows = [
        ("start -> stop", f"{base_start.date()} -> {b['stop'].date()}",
         f"{mem_start.date()} -> {m['stop'].date()}"),
        ("sim days", f"{b['days']:.0f}", f"{m['days']:.0f}"),
        ("final funds", f"${b['funds'] / 100:,.0f}", f"${m['funds'] / 100:,.0f}"),
        ("net per sim day", f"${b['net_per_day']:,.0f}", f"${m['net_per_day']:,.0f}"),
        ("reward per sim day", f"${b['reward_per_day']:,.0f}", f"${m['reward_per_day']:,.0f}"),
        ("accepted", b["accepted"], m["accepted"]),
        ("finished", b["done"], m["done"]),
        ("teams != 4", b["off_cap"], m["off_cap"]),
        ("avg team", f"{b['avg_team']:.2f}", f"{m['avg_team']:.2f}"),
        ("idle days", f"{b['idle_days']:.0f}", f"{m['idle_days']:.0f}"),
        ("tasks in flight", f"{b['avg_par']:.2f}", f"{m['avg_par']:.2f}"),
        ("days with <=1 task", f"{b['solo_share'] * 100:.0f}%", f"{m['solo_share'] * 100:.0f}%"),
    ]
    for label, bv, mv in rows:
        print(f"  {label:<24}{str(bv):>26}{str(mv):>26}")

    common = min(b["stop"], m["stop"])
    bf, mf = funds_at(b["curve"], common), funds_at(m["curve"], common)
    print(f"\n  --- funds at a common date ({common.date()}, the earlier stop) ---")
    print(f"  {'BASE':<24}{bf / 100:>17,.0f}")
    print(f"  {'MEM':<24}{mf / 100:>17,.0f}")
    print(f"  {'MEM - BASE':<24}{(mf - bf) / 100:>+17,.0f}")

    warn = []
    if abs(b["days"] - m["days"]) > 7:
        warn.append(
            f"arms stopped {abs(b['days'] - m['days']):.0f} sim days apart "
            f"({b['stop'].date()} vs {m['stop'].date()}): final funds are not "
            f"comparable, use the common-date figure above"
        )
    if b["off_cap"] or m["off_cap"]:
        warn.append(
            f"staffing drift: BASE {b['off_cap']} / MEM {m['off_cap']} tasks "
            f"staffed with something other than 4 people"
        )
    if warn:
        print("\n  --- warnings ---")
        for w in warn:
            print(f"  ! {w}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
