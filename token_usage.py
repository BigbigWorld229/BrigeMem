from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def approximate_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(str(text)))


def _to_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def normalize_usage(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {}

    if isinstance(usage, dict):
        return {
            "prompt_tokens": _to_int(usage.get("prompt_tokens")),
            "completion_tokens": _to_int(usage.get("completion_tokens")),
            "total_tokens": _to_int(usage.get("total_tokens")),
        }

    return {
        "prompt_tokens": _to_int(getattr(usage, "prompt_tokens", 0)),
        "completion_tokens": _to_int(getattr(usage, "completion_tokens", 0)),
        "total_tokens": _to_int(getattr(usage, "total_tokens", 0)),
    }


@dataclass
class OperationUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    runtime_seconds: float = 0.0
    api_usage_calls: int = 0
    approximate_usage_calls: int = 0

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        runtime_seconds: float,
        source: str,
    ) -> None:
        self.calls += 1
        self.prompt_tokens += max(0, int(prompt_tokens))
        self.completion_tokens += max(0, int(completion_tokens))
        self.total_tokens += max(0, int(total_tokens))
        self.runtime_seconds += max(0.0, float(runtime_seconds))
        if source == "api_usage":
            self.api_usage_calls += 1
        else:
            self.approximate_usage_calls += 1

    def to_dict(self) -> Dict[str, Any]:
        avg_prompt = self.prompt_tokens / self.calls if self.calls else 0.0
        avg_completion = self.completion_tokens / self.calls if self.calls else 0.0
        avg_total = self.total_tokens / self.calls if self.calls else 0.0
        avg_runtime = self.runtime_seconds / self.calls if self.calls else 0.0
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_tokens_k": self.total_tokens / 1000.0,
            "runtime_seconds": self.runtime_seconds,
            "avg_prompt_tokens_per_call": avg_prompt,
            "avg_completion_tokens_per_call": avg_completion,
            "avg_total_tokens_per_call": avg_total,
            "avg_runtime_seconds_per_call": avg_runtime,
            "api_usage_calls": self.api_usage_calls,
            "approximate_usage_calls": self.approximate_usage_calls,
        }


class TokenUsageTracker:
    def __init__(self) -> None:
        self._ops: Dict[str, OperationUsage] = {}

    def record(
        self,
        op_name: str,
        *,
        prompt_text: str,
        completion_text: str,
        usage: Any = None,
        runtime_seconds: float = 0.0,
    ) -> None:
        usage_dict = normalize_usage(usage)
        prompt_tokens = usage_dict.get("prompt_tokens", 0)
        completion_tokens = usage_dict.get("completion_tokens", 0)
        total_tokens = usage_dict.get("total_tokens", 0)

        source = "api_usage"
        if total_tokens <= 0 and prompt_tokens <= 0 and completion_tokens <= 0:
            source = "approximate"
            prompt_tokens = approximate_token_count(prompt_text)
            completion_tokens = approximate_token_count(completion_text)
            total_tokens = prompt_tokens + completion_tokens
        elif total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens

        bucket = self._ops.setdefault(op_name, OperationUsage())
        bucket.record(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            runtime_seconds=runtime_seconds,
            source=source,
        )

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {name: stats.to_dict() for name, stats in sorted(self._ops.items())}

    def aggregate(self, op_names: Iterable[str]) -> Dict[str, Any]:
        total = OperationUsage()
        names = list(op_names)
        for name in names:
            op = self._ops.get(name)
            if op is None:
                continue
            total.calls += op.calls
            total.prompt_tokens += op.prompt_tokens
            total.completion_tokens += op.completion_tokens
            total.total_tokens += op.total_tokens
            total.runtime_seconds += op.runtime_seconds
            total.api_usage_calls += op.api_usage_calls
            total.approximate_usage_calls += op.approximate_usage_calls
        payload = total.to_dict()
        payload["operation_names"] = names
        return payload

    def operation_names(self) -> list[str]:
        return sorted(self._ops.keys())
