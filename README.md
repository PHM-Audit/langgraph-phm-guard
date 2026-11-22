# langgraph-phm-guard

Deterministic physics/bounds tool-call guard for [LangGraph](https://github.com/langchain-ai/langgraph) `ToolNode` — a `wrap_tool_call` interceptor that denies physically impossible tool arguments **before** the tool executes.

Where content-safety guardrails check what the model says, this checks whether what the agent is about to do is physically possible: faster-than-light speeds, negative masses, NaN leaking into measurements, impossible depths — short-circuited with an evidence-bearing `ToolMessage` (`PHM_VETO | speed=400000000.0 exceeds hard limit 299792458.0`).

## Features

- **Deterministic**: fixed rule table, no LLM in the judging path, no network; every verdict is recomputable
- **Three-state exits**: clean calls run (`Proceed`), fixable violations guide the model (`PHM_GUIDE`), impossible calls hard-veto (`PHM_VETO`)
- **Operator-declared bounds**: extend/override the default hard limits per deployment via `hard_limits=...`
- **Real-dispatch tested**: tests drive langgraph's actual `ToolNode` execution machinery (`_arun_one` → `wrap_tool_call`), not mocked wrappers

## Install

```bash
pip install langgraph-phm-guard
```

## Usage

```python
from langgraph.prebuilt import ToolNode
from langgraph_phm_guard import phm_tool_guard

node = ToolNode(
    [launch_probe, read_sensor],
    wrap_tool_call=phm_tool_guard(mode="deny"),   # or mode="guide"
)
```

Custom bounds (operator-declared caps override the default table):

```python
node = ToolNode(
    tools,
    wrap_tool_call=phm_tool_guard(hard_limits={"speed": 300.0}),  # 300 m/s fleet cap
)
```

Default rule table:

| Quantity | Hard limit |
|---|---|
| speed / velocity | 299,792,458 m/s |
| rpm | 500,000 |
| force_n | 10,000,000 N |
| power_w | 1,000,000,000 W |
| altitude_m | 40,000,000 m |
| depth_m | 12,000 m |

Negative values are forbidden for mass-like quantities; all numeric values must be finite (NaN/inf denied).

## Origin

Proposed in langgraph#9428; maintainer guidance was that a validator belongs at the existing `wrap_tool_call` hook — this package is that validator. The physical-checking core ships in `phm-lite-check` (PyPI) and the same rule table is merged into microsoft/PyRIT as `PhysicsConsistencyScorer`.

## License

MIT
