from __future__ import annotations

from dataclasses import dataclass
import argparse
import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunArgs:
    model: str
    seed: int
    horizon_years: int | None  # None = defer to sim.horizon_years in config
    company_name: str
    start_date: str
    config_name: str = "default"
    no_live: bool = False
    max_episodes: int = 1
    reasoning_effort: str | None = None
    # Overrides loop.max_turns from the config when set (debug/smoke runs).
    max_turns: int | None = None
    # Ignore any leftover DB / transcript / session / memory log and start over.
    fresh: bool = False
    # Fixed calendar stop date (YYYY-MM-DD). Overrides horizon_years so that two
    # arms compared at equal wall-clock exposure end on the same sim day.
    until: str | None = None


def build_parser():
    parser = argparse.ArgumentParser(
        prog="yc-bench run",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--horizon-years",
        type=int,
        default=None,
        help="Simulation horizon in years (default: read from config's sim.horizon_years)",
    )
    parser.add_argument("--company-name", default="BenchCo")
    parser.add_argument(
        "--start-date", default="2025-01-01", help="Simulation start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--config",
        dest="config_name",
        default="default",
        help="Preset name ('default', 'fast_test', 'high_reward') or path to a .toml file",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        default=False,
        help="Disable the live terminal dashboard (show raw log output instead)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1,
        help="Max episodes (restarts after bankruptcy with scratchpad carried over). Default: 1",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Reasoning-effort level passed to the model (e.g. minimal/low/medium/high/xhigh). "
        "Omit for provider-default reasoning. Also tags output filenames so sweeps don't collide.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Hard cap on LLM turns, overriding loop.max_turns from the config. "
        "Use a small value (e.g. 20) to smoke-test a model without running a full year.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help="Discard leftover DB / transcript / session / memory log for this "
        "config+seed+model and start a new episode. Without it a non-terminal "
        "simulation is resumed and prior turns count against --max-turns.",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="Fixed calendar date (YYYY-MM-DD) at which the episode ends. Overrides "
        "--horizon-years. Use this so two arms compared against each other stop on "
        "the same simulation day - otherwise the arm that advances the clock faster "
        "banks more elapsed time (and more payroll) for the same turn budget.",
    )
    return parser


def parse_run_args(argv):
    parser = build_parser()
    ns = parser.parse_args(argv)
    _validate(ns, parser)
    return RunArgs(
        model=ns.model,
        seed=ns.seed,
        horizon_years=ns.horizon_years,
        company_name=ns.company_name,
        start_date=ns.start_date,
        config_name=ns.config_name,
        no_live=ns.no_live,
        max_episodes=ns.max_episodes,
        reasoning_effort=ns.reasoning_effort,
        max_turns=ns.max_turns,
        fresh=ns.fresh,
        until=ns.until,
    )


def _validate(ns, parser):
    if ns.horizon_years is not None and ns.horizon_years <= 0:
        parser.error("--horizon-years must be int > 0")
    if ns.max_episodes < 1:
        parser.error("--max-episodes must be int >= 1")
    if ns.max_turns is not None and ns.max_turns < 1:
        parser.error("--max-turns must be int >= 1")
    if ns.until is not None:
        try:
            datetime.date.fromisoformat(ns.until)
        except ValueError:
            parser.error("--until must be a date in YYYY-MM-DD form")


__all__ = ["RunArgs", "build_parser", "parse_run_args"]
