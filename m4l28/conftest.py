"""
m4l28 conftest.py

解决 tools 包命名冲突（与 m4l27 策略相同）：
  crewai_mas_demo/tools/ 不含 log_ops / m4l28 专属模块，
  但 pytest 在 m4l28/ 下运行时需确保 tools 子模块解析到 m4l28/tools/。

修复策略：
  - 保留父包 crewai_mas_demo/tools/ 为 sys.modules["tools"]
  - 将 tools.mailbox_ops / tools.log_ops 显式指向 m4l28/tools/ 下对应文件
  - schemas 指向 m4l28/schemas.py

Hook 清理：
  - CrewAI @before_llm_call 全局累积，每次测试前必须清除
"""

import importlib.util as _util
import sys
from pathlib import Path

_here         = Path(__file__).resolve().parent          # m4l28/
_project_root = _here.parent                             # crewai_mas_demo/

for _p in [str(_project_root), str(_here)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_from(module_name: str, file_path: Path) -> None:
    spec = _util.spec_from_file_location(module_name, str(file_path))
    if spec is None:
        raise ImportError(f"Cannot find {file_path}")
    mod = _util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)


# 确保父包 tools 已加载（crewai_mas_demo/tools/）
if "tools" not in sys.modules:
    _load_from("tools", _project_root / "tools" / "__init__.py")

# 将 m4l28 专属模块指向 m4l28/tools/
_load_from("tools.mailbox_ops",  _here / "tools" / "mailbox_ops.py")
_load_from("tools.log_ops",      _here / "tools" / "log_ops.py")
_load_from("tools.proposal_ops", _here / "tools" / "proposal_ops.py")

# schemas 指向 m4l28/schemas.py
_load_from("schemas", _here / "schemas.py")

# hooks 指向 m4l28/hooks/
if "hooks" not in sys.modules:
    _load_from("hooks", _here / "hooks" / "__init__.py")
_load_from("hooks.l2_task_callback", _here / "hooks" / "l2_task_callback.py")

# scheduler
_load_from("scheduler", _here / "scheduler.py")

# seed_logs
_load_from("seed_logs", _here / "seed_logs.py")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

import pytest


@pytest.fixture
def clean_crewai_hooks():
    """集成测试使用：清除残留 hooks，测试结束后再次清除。"""
    from crewai.hooks import clear_before_llm_call_hooks
    clear_before_llm_call_hooks()
    yield
    clear_before_llm_call_hooks()
