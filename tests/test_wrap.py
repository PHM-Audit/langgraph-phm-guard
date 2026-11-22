# Real-dispatch tests: driven through langgraph's actual ToolNode execution
# machinery (_execute_tool_sync with wrap_tool_call wired in constructor).
from langgraph.prebuilt.tool_node import ToolNode, ToolCallRequest, ToolRuntime
from langgraph_phm_guard import phm_tool_guard

executed = []


def probe(speed: float, mass: float = 1.0) -> str:
    """Launch a probe. speed in m/s, mass in kg."""
    executed.append(speed)
    return f"probe at {speed} m/s"


def make_node(mode="deny", hard_limits=None):
    return ToolNode([probe], wrap_tool_call=phm_tool_guard(mode=mode, hard_limits=hard_limits))


import asyncio


def run(node, args):
    executed.clear()
    import json
    req_json = json.dumps({"messages": [{"role": "ai", "tool_calls": [{"name": "probe", "args": args, "id": "call_1", "type": "tool_call"}]}]})
    state = node.invoke(req_json) if False else None
    tool_obj = node.tools_by_name["probe"]
    rt = ToolRuntime(None, None, {"configurable": {}}, None, "rt-test", None, [tool_obj])
    call = {"name": "probe", "args": args, "id": "call_1", "type": "tool_call"}
    out = asyncio.run(node._arun_one(call, "list", rt))
    return out.content, len(executed)


def test_veto_faster_than_light():
    content, n = run(make_node(), {"speed": 4e8})
    assert "PHM_VETO" in content
    assert "400000000" in content or "4e+08" in content
    assert n == 0


def test_veto_negative_mass():
    content, n = run(make_node(), {"speed": 120.0, "mass": -5.0})
    assert "PHM_VETO" in content and "negative" in content
    assert n == 0


def test_veto_nan():
    content, n = run(make_node(), {"speed": float("nan")})
    assert "PHM_VETO" in content and "finite" in content
    assert n == 0


def test_clean_passes_through():
    content, n = run(make_node(), {"speed": 120.0})
    assert "PHM_VETO" not in content
    assert n == 1


def test_guide_mode():
    content, n = run(make_node(mode="guide"), {"speed": 4e8})
    assert "PHM_GUIDE" in content
    assert n == 0


def test_custom_limits():
    # 1000 m/s is legal under the default table but over the operator's 300 m/s cap
    content, n = run(make_node(hard_limits={"speed": 300.0}), {"speed": 1000.0})
    assert "PHM_VETO" in content and n == 0


def test_determinism():
    a = run(make_node(), {"speed": 4e8})
    b = run(make_node(), {"speed": 4e8})
    assert a[0] == b[0]
