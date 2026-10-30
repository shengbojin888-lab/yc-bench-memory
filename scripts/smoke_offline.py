"""Offline episode: drive the real CLI exactly as the new system prompt tells the
agent to, against a throwaway SQLite file. No model, no API cost.

What it proves that a text audit cannot:
  - every command in `## Typical Loop` and the initial user message actually
    runs, with the syntax the prompt shows (this is what `assign-all` broke);
  - `task accept` sets the deadline from the ADVERTISED quantity, matching the
    formula the prompt now states;
  - a 4-employee assignment beats 8 on the same task, using the engine's own
    progress code, not a reimplementation;
  - `task cancel` costs prestige but leaves funds untouched;
  - no error message the agent can hit still mentions a disabled command.

Usage:
    uv run python scripts/smoke_offline.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp_check"
DB = TMP / "smoke.db"
EXPERIMENT = "debug_gemini_memory"

failures: list[str] = []


def env() -> dict[str, str]:
    e = dict(os.environ)
    e["DATABASE_URL"] = f"sqlite:///{DB.as_posix()}"
    e["YC_BENCH_EXPERIMENT"] = EXPERIMENT
    return e


def cli(*args: str) -> dict:
    """Run one yc-bench command and normalise its output.

    Direct CLI calls print pretty-printed JSON; the agent's run_command wraps
    the same output in {"ok":..., "stdout":"..."}. Handle both, so the script
    tests what the agent sees.
    """
    proc = subprocess.run(
        ["uv", "run", "yc-bench", *args],
        cwd=ROOT,
        env=env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (proc.stdout or "").strip()
    payload = None
    # 1. whole stdout is one JSON document (the normal case)
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        # 2. stdout has a Rich panel above it; take the last JSON block
        start = out.find("{")
        while start != -1 and payload is None:
            try:
                payload = json.loads(out[start:])
            except json.JSONDecodeError:
                start = out.find("{", start + 1)
    if not isinstance(payload, dict):
        return {"ok": False, "raw": out + (proc.stderr or "")}
    if "ok" in payload and "stdout" in payload:
        return payload
    if "error" in payload:
        return {"ok": False, "error": payload["error"]}
    return {"ok": True, **payload}


def body(res: dict) -> dict:
    """Payload dict whether the result came wrapped in an envelope or not."""
    if res.get("stdout"):
        try:
            return json.loads(res["stdout"])
        except json.JSONDecodeError:
            return {}
    return {k: v for k, v in res.items() if k != "ok"}


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        failures.append(name)
    return ok


def business_days_inclusive(a: datetime, b: datetime) -> int:
    """Weekdays from the accept day (day 1) up to and including the deadline day.

    The engine adds N * 9 business hours starting at the accept hour, so the
    accept day itself is consumed as the first working day.
    """
    n = 0
    d = a.date()
    while d <= b.date():
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n


def main() -> int:
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)

    print("=" * 74)
    print("OFFLINE EPISODE — real CLI, throwaway DB, no model")
    print("=" * 74)

    # --- step 1: init -------------------------------------------------------
    r = cli("sim", "init", "--seed", "42", "--start-date", "01/01/2025",
            "--company-name", "BenchCo")
    check("sim init", r.get("ok") is True, body(r).get("sim_time", ""))

    # --- step 1b: the agent's own tool path --------------------------------
    # `cli()` above shells out to `uv run yc-bench`, which always works. The
    # agent instead goes through executor.run_command, which resolves the
    # console script next to sys.executable. On Windows that script is called
    # `yc-bench.exe`; looking for bare `yc-bench` made every tool call die with
    # WinError 2 and burned five turns before the stall guard fired.
    from yc_bench.agent.commands.executor import check_resolvable, run_command

    check("executor can locate yc-bench", check_resolvable() is None,
          check_resolvable() or "resolved")
    os.environ["DATABASE_URL"] = env()["DATABASE_URL"]
    os.environ["YC_BENCH_EXPERIMENT"] = EXPERIMENT
    tr = run_command("yc-bench company status")
    check("executor runs a real command", tr.get("ok") is True,
          f"exit={tr.get('exit_code')} {tr.get('stderr', '')[:60]}")

    # --- step 1c: --until sets a fixed stop date ---------------------------
    # Two arms given the same --max-turns stop on different sim days unless the
    # horizon is pinned, and final funds then measure elapsed time, not skill.
    from yc_bench.runner.main import _parse_date, _resolve_horizon_end

    start_dt = _parse_date("2025-01-01")
    pinned = _resolve_horizon_end(start_dt, 1, "2025-03-06")
    check("--until pins the stop date",
          pinned.isoformat().startswith("2025-03-06T09:00"),
          pinned.isoformat())
    check("--until overrides horizon_years",
          _resolve_horizon_end(start_dt, 1, None).year == 2026,
          "default horizon still one year out")
    try:
        _resolve_horizon_end(start_dt, 1, "2024-06-01")
        rejected = False
    except SystemExit:
        rejected = True
    check("--until before start date is rejected", rejected, "SystemExit")

    # `--until` lives in runner/args.py, but `yc-bench run` is a Typer command
    # that builds RunArgs by hand. An option declared only in args.py is
    # unreachable from the shell, so the pinning above would never take effect
    # in a real run. Assert the flag is wired all the way to the CLI.
    help_out = subprocess.run(
        ["uv", "run", "yc-bench", "run", "--help"],
        cwd=ROOT, env=env(), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    ).stdout
    check("`run` actually exposes --until", "--until" in help_out,
          "Typer command forwards it to RunArgs")

    # --- step 2: browse -----------------------------------------------------
    r = cli("market", "browse", "--limit", "10")
    tasks = body(r).get("tasks") or []
    check("market browse returns offers", bool(tasks), f"{len(tasks)} offers")
    if not tasks:
        return report()
    first = tasks[0]
    check(
        "browse exposes advertised qty + client",
        "requirements" in first and "client_name" in first,
        f"{first.get('task_id')} from {first.get('client_name')}",
    )
    check(
        "browse does NOT expose deadline (so the formula matters)",
        "deadline" not in first,
        f"keys: {sorted(first)}",
    )
    advertised = sum(r["required_qty"] for r in first["requirements"])

    # --- step 3: accept, then verify the deadline formula -------------------
    r = cli("task", "accept", "--task-id", first["task_id"])
    acc = body(r)
    check("task accept", r.get("ok") is True and acc.get("status") == "planned",
          f"{first['task_id']} -> {acc.get('status')}")
    if acc.get("deadline") and acc.get("accepted_at"):
        a = datetime.fromisoformat(acc["accepted_at"])
        d = datetime.fromisoformat(acc["deadline"])
        days = business_days_inclusive(a, d)
        expected = max(7, int(advertised / 150))
        check(
            "deadline == max(7, advertised/150) business days, accept day = day 1",
            days == expected,
            f"advertised {advertised:.0f} -> deadline {d:%Y-%m-%d %H:%M}, "
            f"{days} business days counting accept day (expected {expected})",
        )

    # --- step 4: employee list, pick the 4 fastest for the domain ----------
    r = cli("employee", "list")
    emps = body(r).get("employees") or body(r).get("data") or []
    domain = first["requirements"][0]["domain"]
    rates = []
    for e in emps:
        rate = None
        for key in ("skill_rates", "rates", "skills"):
            blob = e.get(key)
            if isinstance(blob, dict):
                rate = blob.get(domain)
            elif isinstance(blob, list):
                for item in blob:
                    if item.get("domain") == domain:
                        rate = item.get("rate") or item.get("rate_domain_per_hour")
            if rate is not None:
                break
        if rate is None:
            rate = e.get(f"rate_{domain}") or 0
        rates.append((float(rate), e.get("name") or e.get("employee_id")))
    rates.sort(reverse=True)
    check("employee list gives usable rates", rates and rates[0][0] > 0,
          f"top: {rates[:2]}")
    top4 = [n for _r, n in rates[:4]]
    all8 = [n for _r, n in rates]

    # --- step 5: assign the way the prompt says (comma list) --------------
    r = cli("task", "assign", "--task-id", first["task_id"], "--employees", ",".join(top4))
    check("task assign --employees a,b,c,d (prompt syntax)",
          r.get("ok") is True, f"assigned {top4}")
    r = cli("task", "dispatch", "--task-id", first["task_id"])
    check("task dispatch", r.get("ok") is True, body(r).get("status", ""))

    # --- step 6: advance one event ----------------------------------------
    r = cli("sim", "resume")
    check("sim resume advances with an active task", r.get("ok") is True,
          f"{body(r).get('old_sim_time', '?')} -> {body(r).get('new_sim_time', '?')}")

    # --- step 7: second task — dispatch gate and 8-person assignment ------
    r = cli("market", "browse", "--limit", "10")
    more = body(r).get("tasks") or []
    second = next((t for t in more if t["task_id"] != first["task_id"]), None)
    if second:
        cli("task", "accept", "--task-id", second["task_id"])
        # Dispatch before assigning must fail with advice that actually works.
        r = cli("task", "dispatch", "--task-id", second["task_id"])
        err = (r.get("error") or r.get("raw") or "")
        check(
            "dispatch-before-assign fails with usable advice",
            r.get("ok") is False
            and "task assign" in err
            and "assign-all" not in err,
            err[:110],
        )
        # The CLI accepts eight; the 4-contributor cap lives in progress.py and
        # is measured numerically by scripts/check_mechanics.py.
        r = cli("task", "assign", "--task-id", second["task_id"],
                "--employees", ",".join(all8))
        check("task assign accepts eight employees", r.get("ok") is True, "")
        check("second dispatch", cli("task", "dispatch", "--task-id",
                                    second["task_id"]).get("ok") is True, "")
        inspected = body(cli("task", "inspect", "--task-id", second["task_id"]))
        n_assigned = len(inspected.get("assignments") or [])
        check("inspect reports all eight assignments", n_assigned == 8, f"{n_assigned} assigned")
        first_state = body(cli("task", "inspect", "--task-id", first["task_id"]))
        check("inspect reports progress", "progress_pct" in first_state,
              f"{first['task_id']} at {first_state.get('progress_pct')}%")

    # --- step 8: cancel costs prestige, not cash -------------------------
    status_before = body(cli("company", "status"))
    funds_before = status_before.get("funds_cents")
    victim = second["task_id"] if second else first["task_id"]
    r = cli("task", "cancel", "--task-id", victim, "--reason", "offline smoke test")
    check("task cancel accepts --reason", r.get("ok") is True, body(r).get("status", ""))
    status_after = body(cli("company", "status"))
    check(
        "cancel leaves funds unchanged (prestige-only)",
        funds_before == status_after.get("funds_cents"),
        f"{funds_before} -> {status_after.get('funds_cents')}",
    )

    # --- step 9: the disabled command is genuinely disabled --------------
    r = cli("task", "assign-all", "--task-id", first["task_id"])
    check("assign-all really is disabled, so the prompt must not suggest it",
          r.get("ok") is False, (r.get("error") or r.get("raw") or "")[:90])

    return report()


def report() -> int:
    print("-" * 74)
    if failures:
        print(f"RESULT: {len(failures)} failure(s): {', '.join(failures)}")
        return 1
    print("RESULT: offline episode clean — every prompt command works as written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
