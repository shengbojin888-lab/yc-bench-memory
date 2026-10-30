"""Preflight check for a model/provider wiring — no benchmark run involved.

Offline by default: resolves the provider, finds the matching API key env var,
and prints what the runtime would use. With ``--live`` it also sends one tiny
completion (no tools, a few tokens) so an auth/region/quota problem surfaces in
seconds instead of mid-run.

Usage::

    uv run python scripts/check_model.py --model gemini/gemini-2.5-flash
    uv run python scripts/check_model.py --model gemini/gemini-2.5-flash --live

The ``--live`` call costs well under a cent and writes nothing to db/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _mask(value: str | None) -> str:
    if not value:
        return "unset"
    if len(value) <= 8:
        return "set (short)"
    return f"set ({value[:4]}...{value[-4:]}, {len(value)} chars)"


# Regions the Gemini API (AI Studio key) rejects. Google documents these as
# unavailable; requests from them fail with 400 FAILED_PRECONDITION
# "User location is not supported for the API use."
_GEMINI_BLOCKED_REGIONS = {
    "CN": "mainland China",
    "HK": "Hong Kong SAR",
    "MO": "Macao SAR",
    "RU": "Russia",
    "BY": "Belarus",
    "IR": "Iran",
    "KP": "North Korea",
    "CU": "Cuba",
    "SY": "Syria",
}


def _geo_block_hint(exc: Exception, country: str | None) -> str:
    """Return remediation lines when the failure is Google's region check."""
    text = str(exc)
    if "location is not supported" not in text and "FAILED_PRECONDITION" not in text:
        return ""
    lines = ["", "This is a NETWORK EGRESS problem, not auth/config:"]
    if country:
        lines.append(f"  caller IP is geolocated in {country}, which the Gemini API rejects.")
    lines += [
        "  Fix: route HTTPS through a node in a supported region, then re-check.",
        "",
        "  PowerShell (adjust the port to your proxy/VPN client):",
        '    $env:HTTPS_PROXY = "http://127.0.0.1:7890"',
        '    $env:HTTP_PROXY  = "http://127.0.0.1:7890"',
        "    # verify the exit country first:  curl.exe -s https://ipinfo.io/json",
        "",
        "  TUN/system-wide VPN modes are not used by Python; only the env vars above are.",
        "  If a VPN client is running, confirm its node is US/JP/SG, not HK/CN.",
    ]
    return "\n".join(lines)


def _egress_report() -> str | None:
    """Print the public egress IP/country and the proxy actually in effect.

    A geo-block is invisible in a 400 body beyond "User location is not
    supported", so surfacing the egress here turns a 20-minute mystery into a
    one-line answer. Returns the two-letter country code, or None if unknown.
    """
    import json
    import urllib.request

    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    print(f"proxy (env)         : {https_proxy or http_proxy or 'none (direct connection)'}")

    try:
        req = urllib.request.Request(
            "https://ipinfo.io/json", headers={"User-Agent": "yc-bench-preflight"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310
            info = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"egress              : unknown ({type(exc).__name__})")
        return None

    country = (info.get("country") or "").upper()
    print(
        f"egress              : {info.get('ip')}  "
        f"{country or '??'} {info.get('city') or ''} "
        f"({info.get('org') or 'unknown ASN'})"
    )
    if country in _GEMINI_BLOCKED_REGIONS:
        print(
            f"  !! {_GEMINI_BLOCKED_REGIONS[country]} is NOT supported by the Gemini API.\n"
            "     Requests will fail with 400 FAILED_PRECONDITION before any\n"
            "     benchmark logic runs. Route the traffic through a supported\n"
            "     region (US / JP / SG) and re-check."
        )
    return country or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check model/provider wiring.")
    parser.add_argument("--model", required=True, help="LiteLLM model string")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also send one minimal completion to verify auth (costs <$0.001)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional preset name: also reports and applies its model settings",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Skip the function-calling probe (probe 2)",
    )
    args = parser.parse_args()
    cfg = None

    from dotenv import find_dotenv, load_dotenv

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
    print(f".env                : {dotenv_path or 'not found'}")

    from yc_bench.agent.runtime.litellm_runtime import (
        _PROVIDER_KEY_ENVS,
        resolve_api_key,
        resolve_provider,
    )

    provider = resolve_provider(args.model)
    _, env_name, key = resolve_api_key(args.model)
    expected = _PROVIDER_KEY_ENVS.get(provider, ())
    print(f"model               : {args.model}")
    print(f"provider            : {provider}")
    print(f"key env (expected)  : {', '.join(expected) if expected else 'none (ADC / no static key)'}")
    print(f"key env (resolved)  : {env_name or 'none'}")
    print(f"key value           : {_mask(key)}")

    try:
        from importlib.metadata import version

        print(f"litellm             : {version('litellm')}")
    except Exception:  # noqa: BLE001
        pass

    # Google enforces a per-request region check on the caller's IP, and it fires
    # before auth/model checks. Report it before spending a request on it.
    egress_country = None
    if args.live and provider in ("gemini", "vertex_ai"):
        egress_country = _egress_report()

    if args.config:
        from yc_bench.config import load_config

        cfg = load_config(args.config)
        print(f"config              : {cfg.name}")
        print(f"  agent.model       : {cfg.agent.model}")
        print(f"  loop.max_turns    : {cfg.loop.max_turns}")
        print(f"  memory.mode       : {cfg.memory.mode}")
        print(f"  reasoning_effort  : {cfg.agent.reasoning_effort}")
        print(f"  send_sampling     : {cfg.agent.send_sampling_params}")
        print(f"  temperature/top_p : {cfg.agent.temperature} / {cfg.agent.top_p}")

        # Show what the effort actually turns into for this provider.
        if cfg.agent.reasoning_effort:
            try:
                from litellm.utils import get_optional_params

                model_short = args.model.split("/", 1)[1] if "/" in args.model else args.model
                optional = get_optional_params(
                    model=model_short,
                    custom_llm_provider=provider,
                    reasoning_effort=cfg.agent.reasoning_effort,
                )
                print(f"  -> thinking       : {optional.get('thinkingConfig') or optional.get('reasoning_effort')}")
            except Exception as exc:  # noqa: BLE001
                print(f"  -> thinking       : could not resolve ({type(exc).__name__}: {exc})")

    if not key and expected:
        print(
            f"\nFAIL: no key for provider '{provider}'. "
            f"Set {' or '.join(expected)} in .env, then re-run.",
            file=sys.stderr,
        )
        return 2

    if not args.live:
        print("\nOK (offline). Add --live to verify authentication with one request.")
        return 0

    import litellm

    litellm.drop_params = True

    # Mirror what the benchmark actually sends: the preset's thinking effort and
    # sampling params. Probing without them tests a different model config than
    # the one the run will use.
    extra: dict = {"api_key": key} if key else {}
    if args.config:
        if cfg.agent.reasoning_effort:
            extra["reasoning_effort"] = cfg.agent.reasoning_effort
        if cfg.agent.send_sampling_params:
            extra["temperature"] = cfg.agent.temperature
            extra["top_p"] = cfg.agent.top_p
    print(f"request extras      : { {k: v for k, v in extra.items() if k != 'api_key'} or 'none'}")

    def _usage(response) -> None:
        usage = getattr(response, "usage", None)
        cost = getattr(response, "_hidden_params", {}).get("response_cost")
        if usage:
            print(
                f"tokens              : {getattr(usage, 'prompt_tokens', 0)} in / "
                f"{getattr(usage, 'completion_tokens', 0)} out"
            )
        if cost is not None:
            print(f"cost                : ${float(cost):.6f}")

    def _finish(response) -> str | None:
        try:
            return response.choices[0].finish_reason
        except Exception:  # noqa: BLE001
            return None

    # Probe 1 — plain text, generous cap so thinking tokens cannot swallow the
    # whole budget (Gemini counts internal thoughts as output tokens).
    print("\n[1/2] text completion ...")
    try:
        response = litellm.completion(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=256,
            timeout=90,
            **extra,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(_geo_block_hint(exc, egress_country), file=sys.stderr)
        return 1

    text = (response.choices[0].message.content or "").strip()
    print(f"reply               : {text!r}")
    print(f"finish_reason       : {_finish(response)}")
    _usage(response)

    # Probe 2 — the real thing the benchmark depends on: does the model emit a
    # `run_command` tool call under this exact thinking config? Gemini 2.5
    # rejects "thinking disabled + function calling", so this must be checked
    # before spending money on a run.
    if not args.no_tools:
        from yc_bench.agent.runtime.litellm_runtime import _RUN_COMMAND_TOOL

        print("\n[2/2] tool call (real run_command schema) ...")
        try:
            response = litellm.completion(
                model=args.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a benchmark agent. Use the tool when asked.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Call run_command exactly once with the command "
                            "'yc-bench company status'."
                        ),
                    },
                ],
                tools=[_RUN_COMMAND_TOOL],
                tool_choice="auto",
                max_tokens=512,
                timeout=90,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
            print(_geo_block_hint(exc, egress_country), file=sys.stderr)
            print(
                "\nThis is the tool-calling path. If it fails while probe 1 passed, "
                "the thinking config and function calling are incompatible — raise "
                "reasoning_effort to 'low' or drop it.",
                file=sys.stderr,
            )
            return 1

        message = response.choices[0].message
        calls = getattr(message, "tool_calls", None) or []
        print(f"tool_calls          : {len(calls)}")
        for tc in calls:
            print(f"  - {tc.function.name}({tc.function.arguments})")
        print(f"finish_reason       : {_finish(response)}")
        _usage(response)
        if not calls:
            print(
                "WARN: no tool call returned. The agent loop needs tool calls every "
                "turn; check the text output above.",
                file=sys.stderr,
            )
            return 1

    print("\nOK (live).")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
