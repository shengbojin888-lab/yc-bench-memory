from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .policy import parse_bench_command

# Windows console scripts are generated with an extension; on POSIX they are
# bare names. Checking only "yc-bench" silently fell back to a PATH lookup that
# fails unless the venv is activated, so every tool call errored with WinError 2.
_SCRIPT_NAMES = (
    ("yc-bench.exe", "yc-bench.cmd", "yc-bench.bat", "yc-bench")
    if os.name == "nt"
    else ("yc-bench",)
)


def _resolve_yc_bench() -> str:
    """Find the yc-bench script in the same directory as the running Python."""
    venv_bin = Path(sys.executable).parent
    for name in _SCRIPT_NAMES:
        candidate = venv_bin / name
        if candidate.exists():
            return str(candidate)
    found = shutil.which("yc-bench")
    if found:
        return found
    return "yc-bench"  # last resort: let subprocess report the failure


def check_resolvable() -> str | None:
    """Return a human-readable problem if tool calls could not run, else None."""
    resolved = _resolve_yc_bench()
    if Path(resolved).exists() or shutil.which(resolved):
        return None
    return (
        f"cannot locate the `yc-bench` executable (looked for "
        f"{', '.join(_SCRIPT_NAMES)} next to {sys.executable} and on PATH); "
        "every tool call would fail before the first turn"
    )


def run_command(
    command,
    *,
    cwd=None,
    timeout_seconds=60.0,
    command_exists=None,
):
    ok, err, argv = parse_bench_command(command)
    if not ok:
        return {
            "ok": False,
            "exit_code": 2,
            "stdout": "",
            "stderr": err or "invalid command",
            "sim_time": None,
            "command": command if isinstance(command, str) else str(command),
        }

    if command_exists is not None and not command_exists(argv):
        return {
            "ok": False,
            "exit_code": 127,
            "stdout": "",
            "stderr": f"command not found: {' '.join(argv)}",
            "sim_time": None,
            "command": command,
        }

    # Resolve yc-bench to the venv-local script
    if argv[0] == "yc-bench":
        argv[0] = _resolve_yc_bench()

    try:
        proc = subprocess.run(
            argv,
            shell=False,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "sim_time": None,
            "command": command,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "exit_code": 124,
            "stdout": exc.stdout or "",
            "stderr": f"command timed out after {timeout_seconds} seconds",
            "sim_time": None,
            "command": command,
        }
    except Exception as exc:
        return {
            "ok": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": str(exc),
            "sim_time": None,
            "command": command,
        }


__all__ = ["check_resolvable", "run_command"]
