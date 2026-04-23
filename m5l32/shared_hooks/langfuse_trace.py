"""F7: Langfuse 追踪 handler（全局）——Langfuse v4 SDK API。"""

import atexit
import os

from langfuse import Langfuse
from langfuse.types import TraceContext

_client = None
_trace_id = None
_trace_context = None
_root_span = None


def _ensure_client():
    global _client
    if _client is None:
        _client = Langfuse()
        atexit.register(lambda: _client.flush() if _client else None)
    return _client


def _ensure_trace(ctx):
    global _trace_id, _trace_context, _root_span
    client = _ensure_client()
    if _trace_id is None:
        _trace_id = client.create_trace_id(seed=ctx.session_id)
        _trace_context = TraceContext(trace_id=_trace_id)
        _root_span = client.start_observation(
            trace_context=_trace_context,
            name=f"session-{ctx.session_id}",
            as_type="chain",
            metadata={"session_id": ctx.session_id},
        )
    return _trace_context


def before_llm_handler(ctx):
    _ensure_trace(ctx)


def after_tool_handler(ctx):
    tc = _ensure_trace(ctx)
    client = _ensure_client()
    span = client.start_observation(
        trace_context=tc,
        name=f"tool-{ctx.tool_name}",
        as_type="tool",
        metadata={"tool": ctx.tool_name, "turn": ctx.turn_number},
    )
    span.end()


def after_turn_handler(ctx):
    tc = _ensure_trace(ctx)
    client = _ensure_client()
    model = os.environ.get("AGENT_MODEL", "qwen-plus")
    gen = client.start_observation(
        trace_context=tc,
        name=f"turn-{ctx.turn_number}",
        as_type="generation",
        model=model,
        metadata=ctx.metadata,
    )
    gen.end()


def task_complete_handler(ctx):
    tc = _ensure_trace(ctx)
    client = _ensure_client()
    span = client.start_observation(
        trace_context=tc,
        name="task-complete",
        as_type="span",
        metadata=ctx.metadata,
    )
    span.end()


def flush_and_close(ctx):
    global _trace_id, _trace_context, _root_span
    if _root_span:
        _root_span.end()
    if _client:
        _client.flush()
    _trace_id = None
    _trace_context = None
    _root_span = None
