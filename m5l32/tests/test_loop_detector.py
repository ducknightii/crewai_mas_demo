"""T9-T12: LoopDetector 单元测试。"""

import pytest

from hook_framework.registry import EventType, GuardrailDeny, HookContext

from reliability.loop_detector import LoopDetector


def _turn_ctx(tool_name: str, output: str, turn: int = 1):
    return HookContext(
        event_type=EventType.AFTER_TURN,
        tool_name=tool_name,
        turn_number=turn,
        session_id="test",
        metadata={"output": output},
    )


# T9: 不同状态不触发
def test_different_states_no_trigger():
    ld = LoopDetector(threshold=3)
    ld.after_turn_handler(_turn_ctx("search", "result A", 1))
    ld.after_turn_handler(_turn_ctx("search", "result B", 2))
    ld.after_turn_handler(_turn_ctx("search", "result C", 3))
    assert ld.get_metrics()["loop_detections"] == 0


# T10: 连续相同状态触发 GuardrailDeny
def test_consecutive_identical_triggers_deny():
    ld = LoopDetector(threshold=3)
    ld.after_turn_handler(_turn_ctx("search", "same result", 1))
    ld.after_turn_handler(_turn_ctx("search", "same result", 2))
    with pytest.raises(GuardrailDeny, match="Loop detected"):
        ld.after_turn_handler(_turn_ctx("search", "same result", 3))
    assert ld.get_metrics()["loop_detections"] == 1


# T11: 阈值参数生效
def test_threshold_parameter():
    ld2 = LoopDetector(threshold=2)
    ld2.after_turn_handler(_turn_ctx("search", "same", 1))
    with pytest.raises(GuardrailDeny):
        ld2.after_turn_handler(_turn_ctx("search", "same", 2))

    ld5 = LoopDetector(threshold=5)
    for i in range(4):
        ld5.after_turn_handler(_turn_ctx("search", "same", i + 1))
    assert ld5.get_metrics()["loop_detections"] == 0


# T12: 重复但不连续不触发 (AABA pattern)
def test_non_consecutive_repeats_no_trigger():
    ld = LoopDetector(threshold=3)
    ld.after_turn_handler(_turn_ctx("search", "A", 1))
    ld.after_turn_handler(_turn_ctx("search", "A", 2))
    ld.after_turn_handler(_turn_ctx("search", "B", 3))
    ld.after_turn_handler(_turn_ctx("search", "A", 4))
    assert ld.get_metrics()["loop_detections"] == 0


# T_extra: metrics 统计正确
def test_metrics_tracking():
    ld = LoopDetector(threshold=3)
    ld.after_turn_handler(_turn_ctx("t1", "x", 1))
    ld.after_turn_handler(_turn_ctx("t2", "y", 2))
    ld.after_turn_handler(_turn_ctx("t3", "z", 3))
    m = ld.get_metrics()
    assert m["total_turns"] == 3
    assert m["unique_states"] == 3


# T_extra2: after_tool_handler 独立检测循环
def test_tool_path_loop_detection():
    ld = LoopDetector(threshold=2)
    tool_ctx = HookContext(
        event_type=EventType.AFTER_TOOL_CALL,
        tool_name="search",
        turn_number=1,
        session_id="test",
        metadata={"output": "same output"},
    )
    ld.after_tool_handler(tool_ctx)
    with pytest.raises(GuardrailDeny, match="Loop detected"):
        ld.after_tool_handler(tool_ctx)


# T_extra3: tool path 和 turn path 互不干扰
def test_dual_path_independent():
    ld = LoopDetector(threshold=3)
    tool_ctx = HookContext(
        event_type=EventType.AFTER_TOOL_CALL,
        tool_name="search",
        turn_number=1,
        session_id="test",
        metadata={"output": "same"},
    )
    # 2 次 tool + 2 次 turn（各不足 threshold=3）
    ld.after_tool_handler(tool_ctx)
    ld.after_turn_handler(_turn_ctx("search", "same", 1))
    ld.after_tool_handler(tool_ctx)
    ld.after_turn_handler(_turn_ctx("search", "same", 2))
    assert ld.get_metrics()["loop_detections"] == 0
