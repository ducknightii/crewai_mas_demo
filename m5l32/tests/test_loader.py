"""T6-T8 + strategies + deps: HookLoader 单元测试。"""

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


# --- strategies 段测试（31课） ---

def test_strategies_basic(tmp_path):
    hooks_dir = tmp_path / "hooks_dir"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.yaml").write_text(textwrap.dedent("""\
        strategies:
          - class: my_strategy.Counter
            config:
              start: 10
            hooks:
              AFTER_TURN: on_turn
    """))
    (hooks_dir / "my_strategy.py").write_text(textwrap.dedent("""\
        class Counter:
            def __init__(self, start=0):
                self.count = start
            def on_turn(self, ctx):
                self.count += 1
    """))

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(hooks_dir, layer_name="test")

    assert registry.handler_count(EventType.AFTER_TURN) == 1
    assert "my_strategy" in loader.strategies
    assert loader.strategies["my_strategy"].count == 10

    ctx = HookContext(event_type=EventType.AFTER_TURN, session_id="t")
    registry.dispatch(EventType.AFTER_TURN, ctx)
    assert loader.strategies["my_strategy"].count == 11


def test_strategies_missing_class_skipped(tmp_path):
    hooks_dir = tmp_path / "hooks_dir"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.yaml").write_text(textwrap.dedent("""\
        strategies:
          - class: missing_mod.Foo
            hooks:
              AFTER_TURN: bar
    """))

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(hooks_dir, layer_name="test")
    assert registry.handler_count(EventType.AFTER_TURN) == 0


# --- deps 段测试（32课） ---

def test_deps_injection(tmp_path):
    hooks_dir = tmp_path / "hooks_dir"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.yaml").write_text(textwrap.dedent("""\
        strategies:
          - class: logger_mod.Logger
            config: {}
            hooks:
              SESSION_END: on_end
          - class: gate_mod.Gate
            config:
              level: strict
            deps:
              logger: logger_mod
            hooks:
              BEFORE_TOOL_CALL: check
    """))
    (hooks_dir / "logger_mod.py").write_text(textwrap.dedent("""\
        class Logger:
            def __init__(self):
                self.events = []
            def on_end(self, ctx):
                self.events.append("end")
    """))
    (hooks_dir / "gate_mod.py").write_text(textwrap.dedent("""\
        class Gate:
            def __init__(self, level="default", logger=None):
                self.level = level
                self.logger = logger
            def check(self, ctx):
                if self.logger:
                    self.logger.events.append("checked")
    """))

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(hooks_dir, layer_name="test")

    assert "logger_mod" in loader.strategies
    assert "gate_mod" in loader.strategies

    gate = loader.strategies["gate_mod"]
    logger = loader.strategies["logger_mod"]
    assert gate.logger is logger
    assert gate.level == "strict"

    ctx = HookContext(event_type=EventType.BEFORE_TOOL_CALL, tool_name="t", session_id="s")
    registry.dispatch(EventType.BEFORE_TOOL_CALL, ctx)
    assert logger.events == ["checked"]


def test_deps_missing_ref_skipped(tmp_path):
    """deps 引用不存在的 strategy → 跳过该 dep，不阻塞实例化。"""
    hooks_dir = tmp_path / "hooks_dir"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.yaml").write_text(textwrap.dedent("""\
        strategies:
          - class: gate_mod.Gate
            config: {}
            deps:
              logger: nonexistent_strategy
            hooks:
              BEFORE_TOOL_CALL: check
    """))
    (hooks_dir / "gate_mod.py").write_text(textwrap.dedent("""\
        class Gate:
            def __init__(self, logger=None):
                self.logger = logger
            def check(self, ctx):
                pass
    """))

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(hooks_dir, layer_name="test")

    assert "gate_mod" in loader.strategies
    assert loader.strategies["gate_mod"].logger is None
