"""F7: Langfuse 追踪 handler（全局）——Langfuse v4 SDK API。

架构要点：
- span 栈实现 Sub-Crew 工具调用的父子嵌套（树状 trace）
- TOOL 使用 pending_spans 追踪真实耗时
- GENERATION / TASK_COMPLETE 始终挂在 root span 下
- flush_and_close 清理孤儿 span
"""

import atexit
import os
import threading

from langfuse import Langfuse
from langfuse.types import TraceContext

_client = None
_trace_id = None
_trace_context = None
_root_span = None
_root_span_id = None
_trace_name = None
_session_id = None
_pending_spans: dict[str, object] = {}
_span_stack: list = []
_span_stack_lock = threading.Lock()
_task_description = ""


def _ensure_client():
    global _client
    if _client is None:
        _client = Langfuse()
        atexit.register(lambda: _client.flush() if _client else None)
    return _client


def _get_otel_span_id(span):
    otel = getattr(span, "_otel_span", None)
    if otel is None:
        return None
    return format(otel.get_span_context().span_id, "016x")


def _set_trace_attrs(span):
    otel = getattr(span, "_otel_span", None)
    if otel is None:
        return
    otel.set_attribute("langfuse.trace.name", _trace_name)
    otel.set_attribute("session.id", _session_id)
    otel.set_attribute("langfuse.trace.metadata", '{"source":"m5l32-hook-framework"}')


def _ensure_trace(ctx):
    global _trace_id, _trace_context, _root_span, _root_span_id, _trace_name, _session_id
    client = _ensure_client()
    if _trace_id is None:
        _session_id = ctx.session_id
        _trace_name = f"agent-run-{ctx.session_id}"
        _trace_id = client.create_trace_id(seed=ctx.session_id)
        _trace_context = TraceContext(trace_id=_trace_id)
        _root_span = client.start_observation(
            trace_context=_trace_context,
            name=f"session-{ctx.session_id}",
            as_type="chain",
            metadata={"session_id": ctx.session_id},
        )
        _set_trace_attrs(_root_span)
        _root_span_id = _get_otel_span_id(_root_span)
    return _trace_context


def _get_parent_context():
    """栈顶 span 为父；栈空则 root span 为父。"""
    parent_id = None
    if _span_stack:
        parent_id = _get_otel_span_id(_span_stack[-1])
    if not parent_id:
        parent_id = _root_span_id
    if parent_id:
        return TraceContext(trace_id=_trace_id, parent_span_id=parent_id)
    return _trace_context


def _get_root_context():
    """始终以 root span 为父（用于 generation / task_complete）。"""
    if _root_span_id:
        return TraceContext(trace_id=_trace_id, parent_span_id=_root_span_id)
    return _trace_context


def _tool_span_key(ctx):
    return f"{ctx.tool_name}:{ctx.turn_number}"


def before_llm_handler(ctx):
    _ensure_trace(ctx)


def before_tool_handler(ctx):
    """BEFORE_TOOL_CALL: 开启 TOOL span，以栈顶为父节点。"""
    _ensure_trace(ctx)
    client = _ensure_client()
    tc = _get_parent_context()
    span = client.start_observation(
        trace_context=tc,
        name=f"tool-{ctx.tool_name}",
        as_type="tool",
        input=ctx.tool_input or None,
        metadata={"tool": ctx.tool_name, "turn": ctx.turn_number},
    )
    _set_trace_attrs(span)
    _pending_spans[_tool_span_key(ctx)] = span
    with _span_stack_lock:
        _span_stack.append(span)


def after_tool_handler(ctx):
    """AFTER_TOOL_CALL: 关闭 TOOL span，出栈。"""
    key = _tool_span_key(ctx)
    span = _pending_spans.pop(key, None)
    if span:
        tool_output = ctx.metadata.get("tool_output", "")
        span.update(output=tool_output or None)
        span.end()
        with _span_stack_lock:
            if _span_stack and _span_stack[-1] is span:
                _span_stack.pop()
    else:
        _ensure_trace(ctx)
        client = _ensure_client()
        tool_output = ctx.metadata.get("tool_output", "")
        tc = _get_parent_context()
        fallback = client.start_observation(
            trace_context=tc,
            name=f"tool-{ctx.tool_name}",
            as_type="tool",
            input=ctx.tool_input or None,
            output=tool_output or None,
            metadata={"tool": ctx.tool_name, "turn": ctx.turn_number},
        )
        _set_trace_attrs(fallback)
        fallback.end()


def after_turn_handler(ctx):
    """AFTER_TURN: 创建 GENERATION，始终挂在 root span 下。"""
    _ensure_trace(ctx)
    client = _ensure_client()
    model = os.environ.get("AGENT_MODEL", "qwen-plus")

    prompt_preview = ctx.metadata.get("prompt_preview", "")
    llm_response = ctx.metadata.get("llm_response", "")
    step_output = ctx.metadata.get("output", "")

    gen_input = prompt_preview or None
    gen_output = llm_response or step_output or None

    tc = _get_root_context()
    gen = client.start_observation(
        trace_context=tc,
        name=f"turn-{ctx.turn_number}",
        as_type="generation",
        model=model,
        input=gen_input,
        output=gen_output,
        metadata={
            "agent": ctx.agent_id,
            "turn": ctx.turn_number,
        },
    )
    _set_trace_attrs(gen)
    gen.end()


def task_complete_handler(ctx):
    """TASK_COMPLETE: 始终挂在 root span 下。"""
    global _task_description
    _ensure_trace(ctx)
    client = _ensure_client()

    task_desc = ctx.metadata.get("task_description", ctx.task_name)
    raw_output = ctx.metadata.get("raw_output", "")
    _task_description = task_desc

    tc = _get_root_context()
    span = client.start_observation(
        trace_context=tc,
        name="task-complete",
        as_type="span",
        input=task_desc or None,
        output=raw_output or None,
        metadata={
            "agent": ctx.agent_id,
        },
    )
    _set_trace_attrs(span)
    span.end()

    if _root_span and (task_desc or raw_output):
        _root_span.update(
            input=task_desc or None,
            output=raw_output or None,
        )


def flush_and_close(ctx):
    global _trace_id, _trace_context, _root_span, _root_span_id
    global _trace_name, _session_id, _task_description

    with _span_stack_lock:
        _span_stack.clear()

    for key, span in list(_pending_spans.items()):
        span.update(level="WARNING", status_message="orphaned-span-auto-closed")
        span.end()
    _pending_spans.clear()

    if _root_span:
        _root_span.end()
    if _client:
        _client.flush()
    _trace_id = None
    _trace_context = None
    _root_span = None
    _root_span_id = None
    _trace_name = None
    _session_id = None
    _task_description = ""
