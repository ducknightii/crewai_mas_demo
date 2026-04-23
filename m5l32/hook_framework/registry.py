"""F1-F2: EventType 枚举 + HookContext 数据类 + HookRegistry 核心分发。

31课新增：GuardrailDeny 异常类 + dispatch_gate 方法。
"""

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable


class EventType(Enum):
    BEFORE_TURN = "before_turn"
    BEFORE_LLM = "before_llm"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    AFTER_TURN = "after_turn"
    TASK_COMPLETE = "task_complete"
    SESSION_END = "session_end"


class GuardrailDeny(Exception):
    """可靠性策略拒绝当前操作时抛出。"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class HookContext:
    event_type: EventType
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    agent_id: str = ""
    task_name: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0
    success: bool = True
    session_id: str = ""
    turn_number: int = 0
    metadata: dict = field(default_factory=dict)


class HookRegistry:
    def __init__(self):
        self._handlers: dict[EventType, list[Callable]] = defaultdict(list)
        self._handler_names: dict[EventType, list[str]] = defaultdict(list)

    def register(self, event_type: EventType, handler: Callable, name: str = ""):
        self._handlers[event_type].append(handler)
        self._handler_names[event_type].append(name or getattr(handler, "__name__", repr(handler)))

    def dispatch(self, event_type: EventType, context: HookContext):
        for handler in self._handlers[event_type]:
            try:
                handler(context)
            except Exception as e:
                print(
                    f"[HookRegistry] {event_type.value} handler error: {e}",
                    file=sys.stderr,
                )

    def dispatch_gate(self, event_type: EventType, context: HookContext):
        """带拦截能力的分发——GuardrailDeny 向上传播，其他异常吞掉。"""
        for handler in self._handlers[event_type]:
            try:
                handler(context)
            except GuardrailDeny:
                raise
            except Exception as e:
                print(
                    f"[HookRegistry] {event_type.value} handler error: {e}",
                    file=sys.stderr,
                )

    def handler_count(self, event_type: EventType) -> int:
        return len(self._handlers[event_type])

    def summary(self) -> dict[str, list[str]]:
        return {
            et.value: list(names)
            for et, names in self._handler_names.items()
            if names
        }
