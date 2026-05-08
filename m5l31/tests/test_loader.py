"""T6-T8 + T_extra3: HookLoader 单元测试。"""

import textwrap
from pathlib import Path

from hook_framework.registry import EventType, HookContext, HookRegistry
from hook_framework.loader import HookLoader


def _write_hook_dir(tmp_path: Path, yaml_content: str, handler_code: str, module_name: str = "my_handler"):
    hooks_dir = tmp_path / "hooks_dir"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.yaml").write_text(yaml_content)
    (hooks_dir / f"{module_name}.py").write_text(handler_code)
    return hooks_dir


# T6: 加载 yaml + handler
def test_load_from_directory(tmp_path):
    yaml_content = textwrap.dedent("""\
        hooks:
          BEFORE_TURN:
            - handler: my_handler.on_turn
    """)
    handler_code = textwrap.dedent("""\
        calls = []
        def on_turn(ctx):
            calls.append(ctx)
    """)
    hooks_dir = _write_hook_dir(tmp_path, yaml_content, handler_code)

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(hooks_dir, layer_name="test")
    assert registry.handler_count(EventType.BEFORE_TURN) == 1


# T7: 两层合并
def test_two_layer_merge(tmp_path):
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    (global_dir / "hooks.yaml").write_text(textwrap.dedent("""\
        hooks:
          TASK_COMPLETE:
            - handler: g_handler.on_complete
    """))
    (global_dir / "g_handler.py").write_text("def on_complete(ctx): pass")

    ws_dir = tmp_path / "workspace"
    ws_hooks = ws_dir / "hooks"
    ws_hooks.mkdir(parents=True)
    (ws_hooks / "hooks.yaml").write_text(textwrap.dedent("""\
        hooks:
          TASK_COMPLETE:
            - handler: w_handler.on_complete
    """))
    (ws_hooks / "w_handler.py").write_text("def on_complete(ctx): pass")

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_two_layers(global_dir, ws_dir)
    assert registry.handler_count(EventType.TASK_COMPLETE) == 2


# T8: 缺 yaml 不报错
def test_missing_yaml(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(empty_dir)
    assert registry.handler_count(EventType.BEFORE_TURN) == 0


# T_extra3: yaml 引用不存在的模块
def test_missing_module_skipped(tmp_path):
    hooks_dir = tmp_path / "hooks_dir"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.yaml").write_text(textwrap.dedent("""\
        hooks:
          BEFORE_TURN:
            - handler: nonexistent_module.do_stuff
    """))

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(hooks_dir, layer_name="test")
    assert registry.handler_count(EventType.BEFORE_TURN) == 0


# ── 31课新增：strategies 段测试 ──────────────────────────────────

_STRATEGY_CLASS = textwrap.dedent("""\
    class MyStrategy:
        def __init__(self, threshold=5):
            self.threshold = threshold
            self.calls = []

        def on_turn(self, ctx):
            self.calls.append(("turn", ctx))

        def on_tool(self, ctx):
            self.calls.append(("tool", ctx))

        def get_metrics(self):
            return {"threshold": self.threshold, "call_count": len(self.calls)}
""")


def test_strategy_class_instantiation(tmp_path):
    hooks_dir = tmp_path / "hooks_dir"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.yaml").write_text(textwrap.dedent("""\
        strategies:
          - class: my_strategy.MyStrategy
            config:
              threshold: 10
            hooks:
              AFTER_TURN: on_turn
    """))
    (hooks_dir / "my_strategy.py").write_text(_STRATEGY_CLASS)

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(hooks_dir, layer_name="test")

    assert registry.handler_count(EventType.AFTER_TURN) == 1
    assert "my_strategy" in loader.strategies
    assert loader.strategies["my_strategy"].threshold == 10


def test_strategy_multiple_events(tmp_path):
    hooks_dir = tmp_path / "hooks_dir"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.yaml").write_text(textwrap.dedent("""\
        strategies:
          - class: my_strategy.MyStrategy
            config:
              threshold: 3
            hooks:
              AFTER_TURN: on_turn
              AFTER_TOOL_CALL: on_tool
    """))
    (hooks_dir / "my_strategy.py").write_text(_STRATEGY_CLASS)

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(hooks_dir, layer_name="test")

    assert registry.handler_count(EventType.AFTER_TURN) == 1
    assert registry.handler_count(EventType.AFTER_TOOL_CALL) == 1

    ctx = HookContext(event_type=EventType.AFTER_TURN, session_id="t")
    registry.dispatch(EventType.AFTER_TURN, ctx)

    instance = loader.strategies["my_strategy"]
    assert len(instance.calls) == 1
    assert instance.calls[0][0] == "turn"


def test_strategy_shared_instance(tmp_path):
    """Same class instance is shared across all its registered events."""
    hooks_dir = tmp_path / "hooks_dir"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.yaml").write_text(textwrap.dedent("""\
        strategies:
          - class: my_strategy.MyStrategy
            config:
              threshold: 1
            hooks:
              AFTER_TURN: on_turn
              AFTER_TOOL_CALL: on_tool
    """))
    (hooks_dir / "my_strategy.py").write_text(_STRATEGY_CLASS)

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(hooks_dir, layer_name="test")

    registry.dispatch(EventType.AFTER_TURN, HookContext(event_type=EventType.AFTER_TURN))
    registry.dispatch(EventType.AFTER_TOOL_CALL, HookContext(event_type=EventType.AFTER_TOOL_CALL))

    instance = loader.strategies["my_strategy"]
    assert len(instance.calls) == 2


def test_strategy_bad_class_ref_skipped(tmp_path):
    hooks_dir = tmp_path / "hooks_dir"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.yaml").write_text(textwrap.dedent("""\
        strategies:
          - class: nonexistent_module.BadClass
            hooks:
              AFTER_TURN: handler
    """))

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(hooks_dir, layer_name="test")

    assert registry.handler_count(EventType.AFTER_TURN) == 0
    assert len(loader.strategies) == 0


def test_strategy_bad_method_skipped(tmp_path):
    hooks_dir = tmp_path / "hooks_dir"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.yaml").write_text(textwrap.dedent("""\
        strategies:
          - class: my_strategy.MyStrategy
            hooks:
              AFTER_TURN: nonexistent_method
    """))
    (hooks_dir / "my_strategy.py").write_text(_STRATEGY_CLASS)

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(hooks_dir, layer_name="test")

    assert registry.handler_count(EventType.AFTER_TURN) == 0
    assert "my_strategy" in loader.strategies


def test_strategies_property_returns_copy(tmp_path):
    hooks_dir = tmp_path / "hooks_dir"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.yaml").write_text(textwrap.dedent("""\
        strategies:
          - class: my_strategy.MyStrategy
            hooks:
              AFTER_TURN: on_turn
    """))
    (hooks_dir / "my_strategy.py").write_text(_STRATEGY_CLASS)

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(hooks_dir, layer_name="test")

    s1 = loader.strategies
    s2 = loader.strategies
    assert s1 is not s2
    assert s1.keys() == s2.keys()
