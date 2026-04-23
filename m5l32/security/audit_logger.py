"""安全审计日志——汇总安全事件，写入可追溯的审计文件。"""

import json
from datetime import datetime, timezone
from pathlib import Path


class SecurityAuditLogger:
    def __init__(self, audit_file: Path | None = None):
        self._audit_file = audit_file
        self._events: list[dict] = []

    def record_event(self, event_type: str, details: dict):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "security_event": event_type,
            **details,
        }
        self._events.append(entry)
        if self._audit_file:
            with open(self._audit_file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def session_end_handler(self, ctx):
        """SESSION_END: 输出安全审计摘要。"""
        summary = self.get_metrics()
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "security_event": "session_summary",
            "session_id": ctx.session_id,
            **summary,
        }
        if self._audit_file:
            with open(self._audit_file, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_metrics(self) -> dict:
        by_type: dict[str, int] = {}
        for e in self._events:
            t = e["security_event"]
            by_type[t] = by_type.get(t, 0) + 1
        return {
            "total_security_events": len(self._events),
            "events_by_type": by_type,
        }
