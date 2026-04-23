"""F5: CrewAI 机制 → HookRegistry 事件映射。

31课升级：dispatch_gate + pending_deny + success 检测 + token 估算。

映射关系：
┌──────────────────────────┬───────────────────────────┐
│ @before_llm_call         │ BEFORE_TURN（首次）       │
│                          │ BEFORE_LLM（每次）        │
│ @before_tool_call        │ BEFORE_TOOL_CALL (gate)   │
│ @after_tool_call         │ AFTER_TOOL_CALL (gate)    │
│ step_callback            │ AFTER_TURN (gate)         │
│ task_callback            │ TASK_COMPLETE             │
└──────────────────────────┴───────────────────────────┘
"""

from typing import Callable

from crewai.hooks import (
    after_tool_call,
    before_llm_call,
    before_tool_call,
    clear_after_tool_call_hooks,
    clear_before_llm_call_hooks,
    clear_before_tool_call_hooks,
)

from .registry import EventType, GuardrailDeny, HookContext, HookRegistry


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) * 2 // 3)


class CrewObservabilityAdapter:
    def __init__(self, registry: HookRegistry, session_id: str = ""):
        self._registry = registry
        self._session_id = session_id
        self._turn_count = 0
        self._current_turn_has_llm = False
        self._cleaned = False
        self._pending_input_tokens = 0
        self._pending_deny: GuardrailDeny | None = None

    def install_global_hooks(self):
        registry = self._registry
        sid = self._session_id

        @before_llm_call
        def _before_llm(context):
            agent_id = getattr(getattr(context, "agent", None), "role", "")
            if not self._current_turn_has_llm:
                self._turn_count += 1
                self._current_turn_has_llm = True
                registry.dispatch(
                    EventType.BEFORE_TURN,
                    HookContext(
                        event_type=EventType.BEFORE_TURN,
                        agent_id=agent_id,
                        session_id=sid,
                        turn_number=self._turn_count,
                    ),
                )
            messages = getattr(context, "messages", None)
            if messages:
                text_len = sum(len(str(m)) for m in messages)
                self._pending_input_tokens = max(1, text_len * 2 // 3)
            else:
                self._pending_input_tokens = 0
            registry.dispatch(
                EventType.BEFORE_LLM,
                HookContext(
                    event_type=EventType.BEFORE_LLM,
                    agent_id=agent_id,
                    session_id=sid,
                    turn_number=self._turn_count,
                    input_tokens=self._pending_input_tokens,
                ),
            )
            return None

        @before_tool_call
        def _before_tool(context):
            if self._pending_deny:
                return False
            try:
                registry.dispatch_gate(
                    EventType.BEFORE_TOOL_CALL,
                    HookContext(
                        event_type=EventType.BEFORE_TOOL_CALL,
                        tool_name=context.tool_name,
                        tool_input=dict(context.tool_input),
                        session_id=sid,
                        turn_number=self._turn_count,
                    ),
                )
            except GuardrailDeny as e:
                self._pending_deny = e
                return False
            return None

        @after_tool_call
        def _after_tool(context):
            output = str(getattr(context, "tool_output", ""))
            is_error = any(
                kw in output.lower()
                for kw in ["error", "exception", "traceback", "failed"]
            )
            try:
                registry.dispatch_gate(
                    EventType.AFTER_TOOL_CALL,
                    HookContext(
                        event_type=EventType.AFTER_TOOL_CALL,
                        tool_name=context.tool_name,
                        success=not is_error,
                        session_id=sid,
                        turn_number=self._turn_count,
                        metadata={"output": output[:500]},
                    ),
                )
            except GuardrailDeny as e:
                self._pending_deny = e

    def make_step_callback(self) -> Callable:
        registry = self._registry
        sid = self._session_id

        def callback(step):
            from crewai.agents.parser import AgentAction

            output_str = str(getattr(step, "output", ""))
            est_output_tokens = _estimate_tokens(output_str)

            try:
                registry.dispatch_gate(
                    EventType.AFTER_TURN,
                    HookContext(
                        event_type=EventType.AFTER_TURN,
                        session_id=sid,
                        turn_number=self._turn_count,
                        tool_name=step.tool if isinstance(step, AgentAction) else "",
                        input_tokens=self._pending_input_tokens,
                        output_tokens=est_output_tokens,
                        metadata={"output": output_str[:500]},
                    ),
                )
            finally:
                self._current_turn_has_llm = False
                self._pending_input_tokens = 0

            pending = self._pending_deny
            self._pending_deny = None
            if pending:
                raise pending

        return callback

    def make_task_callback(self) -> Callable:
        registry = self._registry
        sid = self._session_id

        def callback(task_output):
            registry.dispatch(
                EventType.TASK_COMPLETE,
                HookContext(
                    event_type=EventType.TASK_COMPLETE,
                    session_id=sid,
                    metadata={"raw_output": str(task_output)[:500]},
                ),
            )

        return callback

    def cleanup(self):
        if self._cleaned:
            return
        self._cleaned = True
        self._registry.dispatch(
            EventType.SESSION_END,
            HookContext(
                event_type=EventType.SESSION_END,
                session_id=self._session_id,
            ),
        )
        clear_before_llm_call_hooks()
        clear_before_tool_call_hooks()
        clear_after_tool_call_hooks()
