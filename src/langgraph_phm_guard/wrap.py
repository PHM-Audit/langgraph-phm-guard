"""Deterministic physics/bounds guard for LangGraph tool execution.

A `ToolNode(wrap_=...)` interceptor: tool arguments are checked against a
fixed rule table (hard limits, negative-quantity bans, NaN/inf bans) BEFORE
the tool executes. Physically impossible calls are short-circuited with an
evidence-bearing ToolMessage instead of running.

No LLM in the judging path, no network, no external API: verdicts are
recomputable by anyone from the rule table.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

__all__ = ["HARD_LIMITS", "check_tool_args", "phm_tool_guard"]

HARD_LIMITS: Dict[str, float] = {
    "speed": 299_792_458.0,
    "velocity": 299_792_458.0,
    "rpm": 500_000.0,
    "force_n": 10_000_000.0,
    "power_w": 1_000_000_000.0,
    "altitude_m": 40_000_000.0,
    "depth_m": 12_000.0,
}

# quantities that must be >= 0
NONNEGATIVE = {
    "mass", "distance", "time", "age", "count", "quantity", "depth",
    "altitude", "power", "force", "pressure", "radius", "length",
    "width", "height", "energy", "temperature", "duration",
}


def check_tool_args(args: Dict[str, Any], limits: Optional[Dict[str, float]] = None) -> List[str]:
    """Return the violation list for a tool-arguments dict (empty = clean)."""
    limits = dict(HARD_LIMITS) if limits is None else dict(limits)
    out: List[str] = []
    for k, v in args.items():
        if isinstance(v, bool) or isinstance(v, str):
            continue
        if isinstance(v, (int, float)):
            name = k.lower()
            f = float(v)
            if not math.isfinite(f):
                out.append(f"{k}={v} is not a finite number")
                continue
            lim = limits.get(name) or limits.get(k)
            if lim is not None and abs(f) > lim:
                out.append(f"{k}={v} exceeds hard limit {lim}")
                continue
            if name in NONNEGATIVE and f < 0:
                out.append(f"{k}={v} is negative (forbidden for {name})")
                continue
            if "probability" in name and not (0.0 <= f <= 1.0):
                out.append(f"{k}={v} outside [0, 1] for probability")
                continue
            if "percent" in name and not (0.0 <= f <= 100.0):
                out.append(f"{k}={v} outside [0, 100] for percent")
    return out


def phm_tool_guard(mode: str = "deny", hard_limits: Optional[Dict[str, float]] = None) -> Callable:
    """Factory for a ToolNode `wrap_` interceptor.

    mode="deny": failing calls are short-circuited with a PHM_VETO ToolMessage.
    mode="guide": failing calls return PHM_GUIDE so the model can self-correct.
    """
    if mode not in ("deny", "guide"):
        raise ValueError("mode must be 'deny' or 'guide'")
    limits = dict(HARD_LIMITS)
    limits.update(hard_limits or {})

    from langchain_core.messages import ToolMessage

    def wrapper(request: Any, execute: Callable) -> Any:
        tc = request.tool_call
        violations = check_tool_args(tc.get("input") or tc.get("args") or {}, limits)
        if not violations:
            return execute(request)
        detail = "; ".join(violations[:4])
        if mode == "deny":
            return ToolMessage(
                content=f"PHM_VETO | {detail}",
                tool_call_id=tc.get("id") or "",
                name=tc.get("name") or "",
            )
        return ToolMessage(
            content=f"PHM_GUIDE: recheck arguments - {detail}",
            tool_call_id=tc.get("id") or "",
            name=tc.get("name") or "",
        )

    wrapper.__name__ = "phm_tool_guard"
    return wrapper
