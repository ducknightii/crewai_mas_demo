"""
m4l27 conftest.py（v3）

v3 不再使用 tools/mailbox_ops.py，所有邮箱操作均通过
workspace/manager/skills/mailbox/scripts/mailbox_cli.py 完成。
"""

import sys
from pathlib import Path

_here         = Path(__file__).resolve().parent   # m4l27/
_project_root = _here.parent                      # crewai_mas_demo/

for _p in [str(_here), str(_project_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
