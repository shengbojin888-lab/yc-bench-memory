"""Offline validation for the structured-memory module (variant (ii)).

No model calls, no simulator runs. Three independent checks:

A. fixture   — synthetic observations covering the two workload regimes the
               benchmark actually produces (scope creep x4 on adversarial
               clients, trust discount <=1 on normal ones), plus cancel and a
               still-in-flight task. Verifies the arithmetic and the
               ``acc == ok + fail + cancel + in_flight`` invariant.
B. replay    — replays a real ``*.session.json`` (the agent's own tool calls
               and their real stdout envelopes) through the extractor, so the
               parsing path is tested against genuine payloads rather than
               fixtures.
C. full year — folds a completed one-year run's task table into events (test
               -only adapter, not part of the memory module) to check
               aggregation, ordering and rendering at real scale.

Run:  python scripts/memory_selftest.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yc_bench.memory import (  # noqa: E402
    MemoryEvent,
    StructuredExtractor,
    build_records,
    client_stats,
    render_client_view,
    render_staff_view,
    staff_stats,
)
from yc_bench.memory.view import estimate_tokens  # noqa: E402

SESSION = ROOT / "db" / "quick_test_1_deepseek_deepseek-chat.session.json"
FULL_YEAR_DB = (
    ROOT
    / "db"
    / "src"
    / "yc_bench"
    / "config"
    / "presets"
    / "default.toml_42_deepseek_deepseek-chat.db"
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def envelope(payload: dict, command: str = "yc-bench") -> str:
    """Wrap a CLI payload the way the command executor does."""
    return json.dumps(
        {
            "ok": True,
            "exit_code": 0,
            "stdout": json.dumps(payload, indent=2),
            "stderr": "",
            "sim_time": None,
            "command": command,
        }
    )


def browse_payload(entries: list[dict]) -> dict:
    return {"total": len(entries), "offset": 0, "limit": 20, "tasks": entries}


def market_entry(title: str, client: str, qty: float, reward: int) -> dict:
    return {
        "task_id": title,
        "client_name": client,
        "required_prestige": 1,
        "required_trust": 0,
        "reward_funds_cents": reward,
        "reward_prestige_delta": 0.05,
        "skill_boost_pct": 0.1,
        "requirements": [{"domain": "data_entry", "required_qty": qty}],
    }


def inspect_payload(
    title: str, client: str, qty: float, deadline: str, accepted_at: str
) -> dict:
    return {
        "task_id": title,
        "title": title,
        "status": "planned",
        "client_name": client,
        "required_prestige": 1,
        "required_trust": 0,
        "reward_funds_cents": 0,
        "reward_prestige_delta": 0.0,
        "skill_boost_pct": 0.0,
        "accepted_at": accepted_at,
        "deadline": deadline,
        "completed_at": None,
        "success": None,
        "progress_pct": 0.0,
        "requirements": [
            {"domain": "data_entry", "required_qty": qty, "completed_qty": 0.0,
             "remaining_qty": qty}
        ],
        "assignments": [],
    }


# ---------------------------------------------------------------------------
# A. fixture
# ---------------------------------------------------------------------------


def check_fixture() -> None:
    print("\nA. fixture: creep / trust-discount / cancel / in-flight")
    ex = StructuredExtractor()
    t0 = "2025-03-10T09:00:00"

    browse = browse_payload(
        [
            market_entry("Task-1", "Helix Labs", 800, 9_900),
            market_entry("Task-2", "Northwind", 1000, 12_000),
        ]
    )
    accept_1 = {
        "task_id": "Task-1",
        "status": "planned",
        "accepted_at": t0,
        "deadline": "2025-03-17T09:00:00",
        "replacement_task_id": "Task-900001",
    }
    accept_2 = {
        "task_id": "Task-2",
        "status": "planned",
        "accepted_at": t0,
        "deadline": "2025-03-20T09:00:00",
        "replacement_task_id": "Task-900002",
    }

    events, followups = ex.observe_turn(
        [
            {
                "command": "yc-bench market browse --limit 20",
                "result": envelope(browse, "yc-bench market browse --limit 20"),
            },
            {
                "command": "yc-bench task accept --task-id Task-1",
                "result": envelope(accept_1),
            },
            {
                "command": "yc-bench task accept --task-id Task-2",
                "result": envelope(accept_2),
            },
        ],
        turn=1,
        sim_time=t0,
    )
    check("两个 accept 各产生一条记录", len(events) == 2, f"got {len(events)}")
    check(
        "accept 后自动请求 task inspect 补齐真实工作量",
        followups == [
            "yc-bench task inspect --task-id Task-1",
            "yc-bench task inspect --task-id Task-2",
        ],
        str(followups),
    )
    check(
        "advertised_qty 从 browse 带到 accept 事件上",
        events[0].payload["advertised_qty"] == {"data_entry": 800.0},
        str(events[0].payload.get("advertised_qty")),
    )

    # Adversarial client: 800 -> 3200
    ev = ex.observe_followup(
        "yc-bench task inspect --task-id Task-1",
        envelope(inspect_payload("Task-1", "Helix Labs", 3200, "2025-03-17T09:00:00", t0)),
        turn=1,
        sim_time=t0,
    )
    check("follow-up inspect 产出 task_inspected", ev is not None)
    events.append(ev)
    # Normal client with trust discount: 1000 -> 600
    ev = ex.observe_followup(
        "yc-bench task inspect --task-id Task-2",
        envelope(inspect_payload("Task-2", "Northwind", 600, "2025-03-20T09:00:00", t0)),
        turn=1,
        sim_time=t0,
    )
    events.append(ev)

    assign_ev, _ = ex.observe_turn(
        [
            {
                "command": "yc-bench task assign --task-id Task-1 --employees Emp_3,Emp_5",
                "result": envelope(
                    {
                        "task_id": "Task-1",
                        "status": "planned",
                        "newly_assigned": ["Emp_3", "Emp_5"],
                        "total_assigned": ["Emp_3", "Emp_5"],
                    }
                ),
            }
        ],
        turn=2,
        sim_time=t0,
    )
    events.extend(assign_ev)

    resume = {
        "ok": True,
        "old_sim_time": t0,
        "new_sim_time": "2025-03-17T14:00:00",
        "checkpoint_event_type": "task_completed",
        "events_processed": 1,
        "payrolls_applied": 0,
        "balance_delta": -346_500,
        "wake_events": [
            {
                "type": "task_completed",
                "task_id": "d51f0509-06f9-419a-9162-74ba1408b2eb",
                "task_title": "Task-1",
                "client_name": "Helix Labs",
                "success": False,
                "funds_delta": -346_500,
                "listed_reward": 9_900,
                "trust_delta": -0.4,
                "deadline_margin": "late by 5h",
                "employees_assigned": 2,
                "salary_bump_total_cents": 0,
                "bankrupt": False,
            }
        ],
        "bankrupt": False,
        "horizon_reached": False,
        "terminal_reason": None,
    }
    done_ev, _ = ex.observe_turn(
        [{"command": "yc-bench sim resume", "result": envelope(resume)}],
        turn=3,
        sim_time="2025-03-17T14:00:00",
    )
    events.extend(done_ev)

    cancel_ev, _ = ex.observe_turn(
        [
            {
                "command": "yc-bench task cancel --task-id Task-2 --reason test",
                "result": envelope(
                    {
                        "task_id": "Task-2",
                        "status": "cancelled",
                        "reason": "test",
                        "cancel_penalty_per_domain": 0.08,
                        "trust_delta": 0.0,
                        "bankrupt": False,
                    }
                ),
            }
        ],
        turn=4,
        sim_time="2025-03-18T09:00:00",
    )
    events.extend(cancel_ev)

    # 2 accepted + 2 inspected + 1 assigned + 1 completed + 1 cancelled
    check("事件总数为 7", len(events) == 7, f"got {len(events)}")

    records = build_records(events)
    rec1 = records["Task-1@2025-03-10T09:00:00"]
    rec2 = records["Task-2@2025-03-10T09:00:00"]
    check("对抗客户 creep_ratio == 4.0", rec1.creep_ratio == 4.0, str(rec1.creep_ratio))
    check("普通客户 trust 折扣后 ratio == 0.6", rec2.creep_ratio == 0.6, str(rec2.creep_ratio))
    check("失败单 outcome == fail", rec1.outcome == "fail", rec1.outcome)
    check("失败单 margin == -5h", rec1.margin_hours == -5.0, str(rec1.margin_hours))
    check("取消单 outcome == cancelled", rec2.outcome == "cancelled", rec2.outcome)
    check("派工人员记录正确", rec1.employees == ["Emp_3", "Emp_5"], str(rec1.employees))

    stats = {s.client: s for s in client_stats(records)}
    helix = stats["Helix Labs"]
    check("Helix: 1 接 / 1 败", (helix.accepted, helix.fail) == (1, 1), str(helix))
    check("Helix: 平均 creep 4.0", helix.creep_mean == 4.0, str(helix.creep_mean))
    check("Northwind: 1 接 / 1 取消", (stats["Northwind"].accepted, stats["Northwind"].cancelled) == (1, 1))

    staff = {s.employee: s for s in staff_stats(records)}
    check("员工 Emp_3 记到 1 单 1 败", staff["Emp_3"].tasks == 1 and staff["Emp_3"].fail == 1)

    print("\n  渲染结果（client view）:")
    for line in render_client_view(client_stats(records), token_budget=300).splitlines():
        print("    " + line)
    print("\n  渲染结果（staff view）:")
    for line in render_staff_view(staff_stats(records), token_budget=200).splitlines():
        print("    " + line)


# ---------------------------------------------------------------------------
# B. replay a real session
# ---------------------------------------------------------------------------


def turns_from_session(path: Path) -> list[tuple[int, list[dict]]]:
    msgs = json.loads(path.read_text(encoding="utf-8"))
    by_id = {
        m.get("tool_call_id"): m.get("content")
        for m in msgs
        if m.get("role") == "tool"
    }
    turns: list[tuple[int, list[dict]]] = []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        calls = []
        for tc in m.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:  # noqa: BLE001
                continue
            calls.append(
                {"command": args.get("command", ""), "result": by_id.get(tc["id"])}
            )
        if calls:
            turns.append((0, calls))
    return turns


def check_replay() -> None:
    print("\nB. replay: 真实 session 中的工具返回")
    if not SESSION.exists():
        print(f"  [SKIP] {SESSION} 不存在")
        return
    ex = StructuredExtractor()
    events: list[MemoryEvent] = []
    accepts = 0
    for idx, (_, calls) in enumerate(turns_from_session(SESSION), start=1):
        for c in calls:
            if " task accept " in f" {c['command']} ":
                accepts += 1
        turn_events, _ = ex.observe_turn(calls, turn=idx, sim_time=None)
        events.extend(turn_events)

    by_type: dict[str, int] = {}
    for ev in events:
        by_type[ev.type] = by_type.get(ev.type, 0) + 1
    print(f"  抽到事件: {by_type}")
    check("解析出 accept 事件", by_type.get("task_accepted", 0) > 0, str(by_type))
    check(
        "accept 数量与命令数一致（成功回执才算）",
        by_type.get("task_accepted", 0) <= accepts,
        f"events={by_type.get('task_accepted', 0)} cmds={accepts}",
    )
    records = build_records(events)
    print(f"  折叠出 {len(records)} 条任务记录")
    for stats in client_stats(records):
        stats.check_invariant()
    view = render_client_view(client_stats(records), token_budget=300)
    print("\n  渲染结果:")
    for line in view.splitlines():
        print("    " + line)
    observed = sum(1 for r in records.values() if r.observed_max)
    print(
        f"  真实工作量覆盖 {observed}/{len(records)}"
        "（回放没有执行 follow-up inspect，未覆盖属预期）"
    )


# ---------------------------------------------------------------------------
# C. full-year aggregation from a finished run's database
# ---------------------------------------------------------------------------


def events_from_db(path: Path) -> list[MemoryEvent]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        select t.title, t.accepted_at, t.deadline, t.completed_at, t.success,
               t.status, c.name as client_name,
               (select max(r.required_qty) from task_requirements r
                 where r.task_id = t.id) as qty
        from tasks t left join clients c on c.id = t.client_id
        where t.company_id is not null
        order by t.accepted_at
        """
    ).fetchall()
    con.close()

    events: list[MemoryEvent] = []
    for row in rows:
        key = f"{row['title']}@{row['accepted_at']}"
        events.append(
            MemoryEvent(
                type="task_accepted",
                task_key=key,
                title=row["title"],
                client=row["client_name"],
                occurred_at=row["accepted_at"],
                turn=0,
                payload={
                    "accepted_at": row["accepted_at"],
                    "deadline": row["deadline"],
                    "advertised_qty": {},
                },
            )
        )
        if row["completed_at"] and row["status"] in (
            "completed_success",
            "completed_fail",
        ):
            success = bool(row["success"])
            events.append(
                MemoryEvent(
                    type="task_completed",
                    task_key=key,
                    title=row["title"],
                    client=row["client_name"],
                    occurred_at=row["completed_at"],
                    turn=0,
                    payload={"success": success, "margin_hours": None},
                )
            )
        elif row["status"] == "cancelled":
            events.append(
                MemoryEvent(
                    type="task_cancelled",
                    task_key=key,
                    title=row["title"],
                    client=row["client_name"],
                    occurred_at=row["completed_at"] or row["accepted_at"],
                    turn=0,
                    payload={},
                )
            )
    return events


def check_full_year() -> None:
    print("\nC. full year: 已完成一年的 run 的任务表")
    if not FULL_YEAR_DB.exists():
        print(f"  [SKIP] {FULL_YEAR_DB} 不存在")
        return
    events = events_from_db(FULL_YEAR_DB)
    records = build_records(events)
    stats = client_stats(records)
    total_ok = sum(s.ok for s in stats)
    total_fail = sum(s.fail for s in stats)
    total_cancel = sum(s.cancelled for s in stats)
    total_acc = sum(s.accepted for s in stats)

    check("折叠出 183 条任务记录", len(records) == 183, str(len(records)))
    check("成功 174", total_ok == 174, str(total_ok))
    check("失败 5", total_fail == 5, str(total_fail))
    check("取消 4", total_cancel == 4, str(total_cancel))
    check("接单总数 183", total_acc == 183, str(total_acc))
    check(
        "creep 在 DB-only 回放中不可测（预期为 0）",
        sum(1 for r in records.values() if r.creep_ratio is not None) == 0,
    )
    view = render_client_view(stats, token_budget=300)
    print(f"\n  渲染结果（{estimate_tokens(view)} tokens）:")
    for line in view.splitlines():
        print("    " + line)


# ---------------------------------------------------------------------------
# D. budget
# ---------------------------------------------------------------------------


def check_budget() -> None:
    print("\nD. budget: 预算截断")
    events: list[MemoryEvent] = []
    for i in range(40):
        client = f"Client_{i:02d}_with_a_long_name"
        stamp = f"2025-02-01T00:00:{i:02d}"
        key = f"Task-{i}@{stamp}"
        events.append(
            MemoryEvent(
                type="task_accepted",
                task_key=key,
                title=f"Task-{i}",
                client=client,
                occurred_at=stamp,
                turn=0,
                payload={
                    "accepted_at": stamp,
                    "advertised_qty": {"data_entry": 1000.0},
                },
            )
        )
        events.append(
            MemoryEvent(
                type="task_inspected",
                task_key=key,
                title=f"Task-{i}",
                client=client,
                occurred_at=stamp,
                turn=0,
                payload={"observed_at": "x", "qty": {"data_entry": 4000.0}},
            )
        )
    stats = client_stats(build_records(events))
    view = render_client_view(stats, token_budget=300)
    tokens = estimate_tokens(view)
    check("渲染 token 不超预算", tokens <= 300, f"{tokens} tokens")
    check("确实发生了截断", "omitted by memory budget" in view)
    check("越靠后的客户优先保留（recency）", "Client_39" in view, "最新客户被丢弃")
    print(f"  {tokens} tokens / 上限 300，共 {len(stats)} 个客户")
    print("\n  渲染结果:")
    for line in view.splitlines():
        print("    " + line)


# ---------------------------------------------------------------------------
# E. loop wiring
# ---------------------------------------------------------------------------


def check_loop_integration() -> None:
    """Exercise the agent loop's two memory hooks without a model or a simulator."""
    print("\nE. loop wiring: 注入点与抽取钩子（stub runtime，无模型调用）")

    from yc_bench.agent import loop as loop_mod
    from yc_bench.agent.run_state import RunState
    from yc_bench.agent.runtime.schemas import RuntimeTurnResult
    from yc_bench.memory import MemorySettings, MemoryStore, StructuredMemory

    snapshot = {
        "sim_time": "2025-03-10T09:00:00",
        "horizon_end": "2026-01-01T09:00:00",
        "funds_cents": 20_000_000,
        "active_tasks": 0,
        "planned_tasks": 0,
        "employee_count": 8,
        "monthly_payroll_cents": 4_000_000,
        "bankrupt": False,
        "scratchpad": None,
    }

    accept_turn = [
        {
            "command": "yc-bench market browse --limit 20",
            "result": envelope(
                browse_payload([market_entry("Task-77", "Helix Labs", 800, 9_900)])
            ),
        },
        {
            "command": "yc-bench task accept --task-id Task-77",
            "result": envelope(
                {
                    "task_id": "Task-77",
                    "status": "planned",
                    "accepted_at": "2025-03-10T09:00:00",
                    "deadline": "2025-03-17T09:00:00",
                    "replacement_task_id": "Task-900077",
                }
            ),
        },
        {
            "command": "yc-bench task inspect --task-id Task-77",
            "result": envelope(
                inspect_payload(
                    "Task-77", "Helix Labs", 3200, "2025-03-17T09:00:00",
                    "2025-03-10T09:00:00",
                )
            ),
        },
    ]

    class StubRuntime:
        def __init__(self, script):
            self.script = list(script)
            self.inputs: list[str] = []

        def run_turn(self, request):
            self.inputs.append(request.user_input)
            calls = self.script.pop(0) if self.script else []
            return RuntimeTurnResult(
                final_output="stub",
                raw_result={"tool_calls": calls},
                checkpoint_advanced=False,
            )

        def clear_session(self, session_id):
            pass

    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    memory = StructuredMemory(
        store=MemoryStore(None),
        settings=MemorySettings(mode="structured", followup_reads=False),
    )
    runtime = StubRuntime([[], accept_turn, []])
    run_state = RunState(
        session_id="stub", seed=1, model="stub/model", horizon_years=1
    )

    original = loop_mod._snapshot_state
    loop_mod._snapshot_state = lambda db, company_id: dict(snapshot)
    try:
        loop_mod.run_agent_loop(
            runtime=runtime,
            db_factory=lambda: _Ctx(),
            company_id="c1",
            run_state=run_state,
            command_executor=None,
            auto_advance_after_turns=99,
            max_turns=3,
            memory=memory,
        )
    finally:
        loop_mod._snapshot_state = original

    check("跑满 3 个回合", len(runtime.inputs) == 3, str(len(runtime.inputs)))
    check("第 1 回合没有记忆块（尚无记录）", "Client memory" not in runtime.inputs[0])
    check(
        "第 3 回合注入了客户记忆",
        "Client memory" in runtime.inputs[2] and "Helix Labs" in runtime.inputs[2],
        runtime.inputs[2][-200:] if len(runtime.inputs) > 2 else "n/a",
    )
    check("事件已落盘（accept + inspect）", len(memory.store) == 2, str(len(memory.store)))
    print("\n  第 3 回合 user 消息末尾:")
    for line in runtime.inputs[2].splitlines()[-3:]:
        print("    " + line)
    print(f"\n  diagnostics: {memory.diagnostics()}")


def check_persistence() -> None:
    """A store built from a path must actually write, even when it starts empty.

    Regression: `store or MemoryStore(None)` silently dropped the real store
    because MemoryStore defines __len__ and an empty log is falsy — so every
    run persisted nothing and the audit trail vanished.
    """
    import tempfile

    from yc_bench.memory import MemorySettings, MemoryStore, StructuredMemory

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.memory.jsonl"
        mem = StructuredMemory(
            store=MemoryStore(path),
            settings=MemorySettings(mode="structured"),
        )
        check("空的 store 不会被丢弃", mem.store.path == path, str(mem.store.path))

        mem.observe_turn(
            turn=1,
            sim_time="2025-03-10T09:00:00",
            tool_calls=[
                {
                    "command": "yc-bench task accept --task-id Task-9",
                    "result": envelope(
                        {
                            "task_id": "Task-9",
                            "status": "planned",
                            "accepted_at": "2025-03-10T09:00:00",
                            "deadline": "2025-03-17T09:00:00",
                        }
                    ),
                }
            ],
        )
        check("落盘文件已创建", path.exists(), str(path))
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        check("写入行数与事件数一致", len(lines) == len(mem.store), f"{len(lines)} vs {len(mem.store)}")

        # A second instance must rehydrate from the same file.
        again = MemoryStore(path)
        check("可从文件重放", len(again) == len(mem.store), f"{len(again)} vs {len(mem.store)}")


def main() -> int:
    print("structured memory self-test (no model calls)")
    check_fixture()
    check_replay()
    check_full_year()
    check_budget()
    check_loop_integration()
    check_persistence()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
