"""可靠性策略：重试追踪 + 循环检测 + 成本围栏。"""

from hook_framework.registry import EventType

from .cost_guard import CostGuard
from .loop_detector import LoopDetector
from .retry_tracker import RetryTracker


def install_reliability_hooks(
    registry,
    config: dict | None = None,
) -> dict:
    """在 HookRegistry 上注册所有可靠性策略。

    Returns:
        dict with "retry", "loop", "cost" keys -> strategy instances
    """
    config = config or {}

    retry = RetryTracker(max_retries=config.get("max_retries", 3))
    loop = LoopDetector(threshold=config.get("loop_threshold", 3))
    cost = CostGuard(
        budget_usd=config.get("budget_usd", 1.0),
        model=config.get("model", ""),
    )

    registry.register(
        EventType.AFTER_TOOL_CALL,
        retry.after_tool_handler,
        name="retry_tracker",
    )
    # 循环检测也挂在 AFTER_TOOL_CALL：覆盖 native function calling 路径
    # （该路径下 step_callback 仅在最终回答时触发）
    registry.register(
        EventType.AFTER_TOOL_CALL,
        loop.after_tool_handler,
        name="loop_detector.tool",
    )

    # cost_guard.accumulate 先于 loop_detector：
    # cost 已发生（LLM 调用已完成），即使循环检测 deny，成本也应记录
    registry.register(
        EventType.AFTER_TURN,
        cost.after_turn_handler,
        name="cost_guard.accumulate",
    )
    registry.register(
        EventType.AFTER_TURN,
        loop.after_turn_handler,
        name="loop_detector.turn",
    )

    registry.register(
        EventType.BEFORE_TOOL_CALL,
        cost.before_tool_handler,
        name="cost_guard.gate",
    )

    return {"retry": retry, "loop": loop, "cost": cost}
