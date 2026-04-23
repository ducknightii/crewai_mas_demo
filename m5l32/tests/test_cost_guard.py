"""T13-T18: CostGuard 单元测试。"""

import pytest

from hook_framework.registry import EventType, GuardrailDeny, HookContext

from reliability.cost_guard import CostGuard


def _turn_ctx(input_tokens: int = 100, output_tokens: int = 50, turn: int = 1):
    return HookContext(
        event_type=EventType.AFTER_TURN,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        turn_number=turn,
        session_id="test",
    )


def _tool_ctx(tool_name: str = "search"):
    return HookContext(
        event_type=EventType.BEFORE_TOOL_CALL,
        tool_name=tool_name,
        session_id="test",
    )


# T13: token 累加正确
def test_token_accumulation():
    cg = CostGuard(budget_usd=100.0, model="qwen-plus")
    cg.after_turn_handler(_turn_ctx(100, 50, 1))
    cg.after_turn_handler(_turn_ctx(100, 50, 2))
    m = cg.get_metrics()
    assert m["total_input_tokens"] == 200
    assert m["total_output_tokens"] == 100


# T14: 预算超出时 before_tool_handler 抛出 GuardrailDeny
def test_budget_exceeded_denies_tool():
    cg = CostGuard(budget_usd=0.0001, model="qwen-plus")
    # after_turn 累加后立即 deny（二次检查）
    with pytest.raises(GuardrailDeny, match="Budget exceeded"):
        cg.after_turn_handler(_turn_ctx(10000, 5000, 1))
    # 成本已累加，before_tool 也应 deny
    with pytest.raises(GuardrailDeny, match="Budget exceeded"):
        cg.before_tool_handler(_tool_ctx("search"))


# T15: 预算内时 before_tool_handler 正常通过
def test_within_budget_allows_tool():
    cg = CostGuard(budget_usd=100.0, model="qwen-plus")
    cg.after_turn_handler(_turn_ctx(100, 50, 1))
    cg.before_tool_handler(_tool_ctx("search"))


# T16: metrics 准确
def test_metrics_accuracy():
    cg = CostGuard(budget_usd=1.0, model="qwen-plus")
    cg.after_turn_handler(_turn_ctx(100000, 50000, 1))
    m = cg.get_metrics()
    assert m["model"] == "qwen-plus"
    assert m["total_input_tokens"] == 100000
    assert m["total_output_tokens"] == 50000
    assert m["budget_usd"] == 1.0
    assert m["estimated_cost_usd"] > 0
    assert m["remaining_usd"] > 0
    assert 0 < m["budget_utilization"] < 1
    assert m["deny_count"] == 0


# T17: 精确边界测试（>= 触发 deny）
def test_exact_budget_boundary_triggers_deny():
    cg = CostGuard(budget_usd=0.0, model="qwen-plus")
    with pytest.raises(GuardrailDeny):
        cg.before_tool_handler(_tool_ctx())


# T18: after_turn_handler 超预算也触发 deny（二次检查）
def test_after_turn_budget_check():
    cg = CostGuard(budget_usd=0.0001, model="qwen-plus")
    with pytest.raises(GuardrailDeny, match="Budget exceeded"):
        cg.after_turn_handler(_turn_ctx(100000, 50000, 1))


# T_extra: deny_count 递增
def test_deny_count_increments():
    cg = CostGuard(budget_usd=0.0, model="qwen-plus")
    for _ in range(3):
        with pytest.raises(GuardrailDeny):
            cg.before_tool_handler(_tool_ctx())
    assert cg.get_metrics()["deny_count"] == 3
