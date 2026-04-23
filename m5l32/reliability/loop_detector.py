"""循环检测——状态哈希去重，连续重复时终止。"""

import hashlib
import json
import sys

from hook_framework.registry import GuardrailDeny


class LoopDetector:
    def __init__(self, threshold: int = 3):
        self._threshold = threshold
        self._tool_hashes: list[str] = []
        self._turn_hashes: list[str] = []
        self._loop_detections = 0

    def _check_loop(self, hashes: list[str], state: str, ctx) -> None:
        h = hashlib.md5(state.encode()).hexdigest()[:16]
        hashes.append(h)

        if len(hashes) >= self._threshold:
            recent = hashes[-self._threshold:]
            if len(set(recent)) == 1:
                self._loop_detections += 1
                self._emit_detection(ctx)
                raise GuardrailDeny(
                    f"Loop detected: identical state repeated "
                    f"{self._threshold} consecutive times "
                    f"(turn {ctx.turn_number}, tool: {ctx.tool_name})"
                )

    def after_turn_handler(self, ctx):
        state = f"{ctx.tool_name}:{ctx.metadata.get('output', '')[:200]}"
        self._check_loop(self._turn_hashes, state, ctx)

    def after_tool_handler(self, ctx):
        """AFTER_TOOL_CALL: 检测工具调用循环（覆盖 native function calling 路径）。"""
        output = ctx.metadata.get("output", "")[:200]
        state = f"{ctx.tool_name}:{output}"
        self._check_loop(self._tool_hashes, state, ctx)

    def _emit_detection(self, ctx):
        record = {
            "level": "CRITICAL",
            "guardrail": "loop_detector",
            "message": "Loop detected — terminating",
            "turn": ctx.turn_number,
            "tool": ctx.tool_name,
            "threshold": self._threshold,
        }
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr)

    def get_metrics(self) -> dict:
        all_hashes = self._tool_hashes + self._turn_hashes
        return {
            "total_turns": len(self._turn_hashes),
            "total_tool_calls": len(self._tool_hashes),
            "unique_states": len(set(all_hashes)),
            "loop_detections": self._loop_detections,
        }
