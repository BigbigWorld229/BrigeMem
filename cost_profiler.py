from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

from token_usage import TokenUsageTracker, normalize_usage


LOGGER = logging.getLogger("memweave_cost_profiler")

STAGE_PREMEM_STEP1 = "premem_step1_information_extraction"
STAGE_QUERY_REWRITE = "query_rewrite"
STAGE_RETRIEVAL_ANSWER = "retrieval_and_answer_generation"

STAGES = (
    STAGE_PREMEM_STEP1,
    STAGE_QUERY_REWRITE,
    STAGE_RETRIEVAL_ANSWER,
)

OPERATION_STAGE_FALLBACK = {
    "prestorage_step1_metadata": STAGE_PREMEM_STEP1,
    "retrieval_keyword_query": STAGE_QUERY_REWRITE,
    "retrieval_relevance_filter": STAGE_RETRIEVAL_ANSWER,
    "qa_answer_generation": STAGE_RETRIEVAL_ANSWER,
}

MEMORY_BUILD_OPERATIONS = ("prestorage_step1_metadata",)
SUMMARY_OPERATIONS = ("prestorage_step1_metadata",)
RETRIEVAL_OPERATIONS = (
    "retrieval_keyword_query",
    "retrieval_relevance_filter",
)
QUERY_ANSWER_OPERATIONS = (*RETRIEVAL_OPERATIONS, "qa_answer_generation")
MEMORY_BUILD_STAGES = (STAGE_PREMEM_STEP1,)

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _empty_stage_summary() -> Dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "llm_latency_seconds": 0.0,
        "auxiliary_latency_seconds": 0.0,
        "total_latency_seconds": 0.0,
        "stage_wall_seconds": 0.0,
        "llm_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
    }


class CostProfiler:
    """Optional, side-channel LLM cost and latency recorder.

    The profiler never invokes an LLM and never modifies prompts or responses.
    Context variables attach the current LoCoMo conversation, pipeline stage,
    and item identifiers to the single controller-level record for each call.
    """

    def __init__(self, *, enabled: bool = False, model_name: str = "") -> None:
        self.enabled = bool(enabled)
        self.model_name = str(model_name)
        self._lock = threading.Lock()
        self._next_call_id = 1
        self._calls: list[Dict[str, Any]] = []
        self._timings: list[Dict[str, Any]] = []
        self._conversation_started: Dict[str, float] = {}
        self._conversation_elapsed: Dict[str, float] = {}
        # The former default TokenUsageTracker now lives inside the profiler so
        # one enable flag controls both operation totals and stage profiling.
        self._usage_tracker = TokenUsageTracker()

        self._conversation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
            "memweave_cost_conversation_id", default=None
        )
        self._stage: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
            "memweave_cost_stage", default=None
        )
        self._item_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
            "memweave_cost_item_context", default={}
        )

    def begin_conversation(self, conversation_id: Any) -> None:
        if not self.enabled:
            return
        conversation_id = str(conversation_id)
        self._conversation_id.set(conversation_id)
        self._conversation_started[conversation_id] = time.perf_counter()

    def end_conversation(self, conversation_id: Any) -> None:
        if not self.enabled:
            return
        conversation_id = str(conversation_id)
        started = self._conversation_started.pop(conversation_id, None)
        if started is not None:
            self._conversation_elapsed[conversation_id] = time.perf_counter() - started
        self._conversation_id.set(None)
        self._stage.set(None)
        self._item_context.set({})

    @contextlib.contextmanager
    def scope(
        self,
        *,
        stage: Optional[str] = None,
        session_id: Any = None,
        turn_id: Any = None,
        qa_id: Any = None,
        item_type: Optional[str] = None,
        **metadata: Any,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        stage_token = None
        if stage is not None:
            stage_token = self._stage.set(str(stage))

        merged = dict(self._item_context.get())
        for key, value in {
            "session_id": session_id,
            "turn_id": turn_id,
            "qa_id": qa_id,
            "item_type": item_type,
            **metadata,
        }.items():
            if value is not None:
                merged[key] = value
        item_token = self._item_context.set(merged)
        try:
            yield
        finally:
            self._item_context.reset(item_token)
            if stage_token is not None:
                self._stage.reset(stage_token)

    @contextlib.contextmanager
    def stage_timer(
        self,
        stage: str,
        operation: str,
        *,
        contributes_to_stage_latency: bool = False,
        **metadata: Any,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        started = time.perf_counter()
        status = "success"
        error_type = None
        try:
            yield
        except BaseException as exc:
            status = "failed"
            error_type = type(exc).__name__
            raise
        finally:
            self.record_timing(
                stage=stage,
                operation=operation,
                latency_seconds=time.perf_counter() - started,
                status=status,
                error_type=error_type,
                contributes_to_stage_latency=contributes_to_stage_latency,
                **metadata,
            )

    @staticmethod
    def _usage_has_tokens(usage: Any) -> bool:
        usage_dict = normalize_usage(usage)
        return any(
            int(usage_dict.get(key, 0) or 0) > 0
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        )

    def _count_with_tokenizer(
        self,
        prompt: str,
        completion: str,
        tokenizer_factory: Optional[Callable[[], Any]],
        system_message: str,
    ) -> tuple[int, int, str]:
        tokenizer = None
        if tokenizer_factory is not None:
            try:
                tokenizer = tokenizer_factory()
            except Exception as exc:
                LOGGER.warning("Cost profiler tokenizer unavailable; using regex estimate: %s", exc)

        if tokenizer is not None:
            try:
                messages = [
                    {"role": "system", "content": str(system_message)},
                    {"role": "user", "content": str(prompt)},
                ]
                if hasattr(tokenizer, "apply_chat_template"):
                    encoded_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                else:
                    encoded_prompt = tokenizer.encode(
                        f"{system_message}\n{prompt}", add_special_tokens=True
                    )
                encoded_completion = tokenizer.encode(
                    str(completion), add_special_tokens=False
                )
                tokenizer_name = (
                    getattr(tokenizer, "name_or_path", None)
                    or getattr(tokenizer, "model_name", None)
                    or tokenizer.__class__.__name__
                )
                return (
                    len(encoded_prompt),
                    len(encoded_completion),
                    f"tokenizer:{tokenizer_name}",
                )
            except Exception as exc:
                LOGGER.warning("Cost profiler tokenizer count failed; using regex estimate: %s", exc)

        input_tokens = len(TOKEN_RE.findall(f"{system_message}\n{prompt}"))
        output_tokens = len(TOKEN_RE.findall(str(completion)))
        return input_tokens, output_tokens, "approximate_regex"

    def record_llm_call(
        self,
        *,
        operation: str,
        prompt: str,
        completion: str,
        usage: Any,
        latency_seconds: float,
        status: str,
        error: Optional[BaseException] = None,
        tokenizer_factory: Optional[Callable[[], Any]] = None,
        system_message: str = "",
        model_name: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        usage_dict = normalize_usage(usage)
        if self._usage_has_tokens(usage):
            input_tokens = int(usage_dict.get("prompt_tokens", 0) or 0)
            output_tokens = int(usage_dict.get("completion_tokens", 0) or 0)
            total_tokens = int(usage_dict.get("total_tokens", 0) or 0)
            if total_tokens <= 0:
                total_tokens = input_tokens + output_tokens
            token_count_method = "api_usage"
        else:
            input_tokens, output_tokens, token_count_method = self._count_with_tokenizer(
                prompt,
                completion,
                tokenizer_factory,
                system_message,
            )
            total_tokens = input_tokens + output_tokens

        stage = self._stage.get() or OPERATION_STAGE_FALLBACK.get(
            str(operation), "unclassified"
        )
        item_context = dict(self._item_context.get())
        conversation_id = self._conversation_id.get()

        with self._lock:
            call_id = self._next_call_id
            self._next_call_id += 1
            self._calls.append(
                {
                    "call_id": call_id,
                    "conversation_id": conversation_id,
                    "stage": stage,
                    "operation": str(operation),
                    "session_id": item_context.pop("session_id", None),
                    "turn_id": item_context.pop("turn_id", None),
                    "qa_id": item_context.pop("qa_id", None),
                    "item_type": item_context.pop("item_type", None),
                    "item_metadata": item_context,
                    "model": str(model_name or self.model_name),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "latency_seconds": max(0.0, float(latency_seconds)),
                    "call_count": 1,
                    "status": str(status),
                    "error_type": type(error).__name__ if error is not None else None,
                    "error_message": str(error) if error is not None else None,
                    "token_count_method": token_count_method,
                }
            )
            # Preserve the former token_statistics semantics: only successful
            # logical LLM calls contribute to operation-level totals.
            if str(status) == "success":
                self._usage_tracker.record(
                    str(operation),
                    prompt_text=prompt,
                    completion_text=completion,
                    usage=usage,
                    runtime_seconds=latency_seconds,
                )

    def record_timing(
        self,
        *,
        stage: str,
        operation: str,
        latency_seconds: float,
        status: str = "success",
        error_type: Optional[str] = None,
        contributes_to_stage_latency: bool = False,
        **metadata: Any,
    ) -> None:
        if not self.enabled:
            return
        item_context = dict(self._item_context.get())
        item_context.update(metadata)
        with self._lock:
            self._timings.append(
                {
                    "conversation_id": self._conversation_id.get(),
                    "stage": str(stage),
                    "operation": str(operation),
                    "latency_seconds": max(0.0, float(latency_seconds)),
                    "status": str(status),
                    "error_type": error_type,
                    "contributes_to_stage_latency": bool(contributes_to_stage_latency),
                    "metadata": item_context,
                }
            )

    @property
    def calls(self) -> list[Dict[str, Any]]:
        return [dict(record) for record in self._calls]

    @property
    def timings(self) -> list[Dict[str, Any]]:
        return [dict(record) for record in self._timings]

    def _summarize_records(
        self,
        calls: list[Dict[str, Any]],
        timings: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        stages = {stage: _empty_stage_summary() for stage in STAGES}
        for call in calls:
            stage = str(call.get("stage") or "unclassified")
            bucket = stages.setdefault(stage, _empty_stage_summary())
            bucket["input_tokens"] += int(call.get("input_tokens", 0))
            bucket["output_tokens"] += int(call.get("output_tokens", 0))
            bucket["total_tokens"] += int(call.get("total_tokens", 0))
            bucket["llm_latency_seconds"] += float(call.get("latency_seconds", 0.0))
            bucket["llm_calls"] += 1
            if call.get("status") == "success":
                bucket["successful_calls"] += 1
            else:
                bucket["failed_calls"] += 1

        for timing in timings:
            stage = str(timing.get("stage") or "unclassified")
            bucket = stages.setdefault(stage, _empty_stage_summary())
            elapsed = float(timing.get("latency_seconds", 0.0))
            bucket["stage_wall_seconds"] += elapsed
            if timing.get("contributes_to_stage_latency"):
                bucket["auxiliary_latency_seconds"] += elapsed

        for bucket in stages.values():
            bucket["total_latency_seconds"] = (
                bucket["llm_latency_seconds"] + bucket["auxiliary_latency_seconds"]
            )

        return {
            "stages": stages,
            "input_tokens": sum(int(call.get("input_tokens", 0)) for call in calls),
            "output_tokens": sum(int(call.get("output_tokens", 0)) for call in calls),
            "total_tokens": sum(int(call.get("total_tokens", 0)) for call in calls),
            "llm_latency_seconds": sum(
                float(call.get("latency_seconds", 0.0)) for call in calls
            ),
            "auxiliary_latency_seconds": sum(
                float(timing.get("latency_seconds", 0.0))
                for timing in timings
                if timing.get("contributes_to_stage_latency")
            ),
            "llm_calls": len(calls),
            "successful_calls": sum(call.get("status") == "success" for call in calls),
            "failed_calls": sum(call.get("status") != "success" for call in calls),
        }

    def snapshot(self) -> Dict[str, Any]:
        conversation_ids = sorted(
            {
                str(record["conversation_id"])
                for record in [*self._calls, *self._timings]
                if record.get("conversation_id") is not None
            }
            | set(self._conversation_elapsed)
        )
        conversations: Dict[str, Any] = {}
        for conversation_id in conversation_ids:
            calls = [
                call for call in self._calls
                if str(call.get("conversation_id")) == conversation_id
            ]
            timings = [
                timing for timing in self._timings
                if str(timing.get("conversation_id")) == conversation_id
            ]
            summary = self._summarize_records(calls, timings)
            summary["conversation_id"] = conversation_id
            summary["end_to_end_seconds"] = float(
                self._conversation_elapsed.get(conversation_id, 0.0)
            )
            conversations[conversation_id] = summary

        dataset_summary = self._summarize_records(self._calls, self._timings)
        dataset_summary["conversation_count"] = len(conversations)
        dataset_summary["end_to_end_seconds"] = sum(
            float(summary["end_to_end_seconds"])
            for summary in conversations.values()
        )
        return {
            "enabled": self.enabled,
            "model": self.model_name,
            "stage_order": list(STAGES),
            "dataset": dataset_summary,
            "conversations": conversations,
            "call_count": len(self._calls),
            "timing_event_count": len(self._timings),
        }

    def token_statistics(self) -> Dict[str, Any]:
        """Return the former default token_statistics payload from this profiler."""
        memory_build = self._usage_tracker.aggregate(MEMORY_BUILD_OPERATIONS)
        query_answer = self._usage_tracker.aggregate(QUERY_ANSWER_OPERATIONS)
        summary_stage = self._usage_tracker.aggregate(SUMMARY_OPERATIONS)
        total_tokens = memory_build["total_tokens"] + query_answer["total_tokens"]
        total_calls = memory_build["calls"] + query_answer["calls"]
        return {
            "summary": {
                "total_llm_calls": total_calls,
                "total_llm_tokens": total_tokens,
            },
            "lightmem_style": {
                "summary_tokens": summary_stage["total_tokens"],
                "summary_tokens_k": summary_stage["total_tokens_k"],
                "memory_build_total_tokens": memory_build["total_tokens"],
                "memory_build_total_tokens_k": memory_build["total_tokens_k"],
                "qa_total_tokens": query_answer["total_tokens"],
                "qa_total_tokens_k": query_answer["total_tokens_k"],
                "total_tokens": total_tokens,
                "total_tokens_k": total_tokens / 1000.0,
                "calls": total_calls,
            },
            "llm": {
                "summary_stage": summary_stage,
                "memory_build": memory_build,
                "query_answer": query_answer,
                "operations": self._usage_tracker.snapshot(),
            },
        }

    def conversation_memory_build_summary(
        self,
        conversation_id: Any,
        *,
        construction_time_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Summarize completed memory-build work for one LoCoMo conversation."""
        conversation_id = str(conversation_id)
        calls = [
            call for call in self._calls
            if str(call.get("conversation_id")) == conversation_id
            and call.get("stage") in MEMORY_BUILD_STAGES
        ]
        timings = [
            timing for timing in self._timings
            if str(timing.get("conversation_id")) == conversation_id
            and timing.get("stage") in MEMORY_BUILD_STAGES
        ]
        summary = self._summarize_records(calls, timings)
        summary["conversation_id"] = conversation_id
        summary["stage_order"] = list(MEMORY_BUILD_STAGES)
        if construction_time_seconds is not None:
            summary["construction_time_seconds"] = float(construction_time_seconds)
        return summary

    def qa_cost_summary(self, conversation_id: Any, qa_id: Any) -> Dict[str, Any]:
        """Return durable per-QA retrieval and answer costs from recorded events."""
        conversation_id = str(conversation_id)
        qa_id_text = str(qa_id)
        calls = [
            call for call in self._calls
            if str(call.get("conversation_id")) == conversation_id
            and str(call.get("qa_id")) == qa_id_text
        ]
        timings = [
            timing for timing in self._timings
            if str(timing.get("conversation_id")) == conversation_id
            and str(timing.get("metadata", {}).get("qa_id")) == qa_id_text
        ]
        retrieval_calls = [
            call for call in calls
            if call.get("operation") in RETRIEVAL_OPERATIONS
        ]
        answer_calls = [
            call for call in calls
            if call.get("operation") == "qa_answer_generation"
        ]
        vector_retrieval_seconds = sum(
            float(timing.get("latency_seconds", 0.0))
            for timing in timings
            if timing.get("operation") == "vector_retrieval_and_link_expansion"
        )
        retrieval_llm_seconds = sum(
            float(call.get("latency_seconds", 0.0)) for call in retrieval_calls
        )
        retrieval_total_events = [
            timing for timing in timings
            if timing.get("operation") == "qa_retrieval_total"
        ]
        answer_total_events = [
            timing for timing in timings
            if timing.get("operation") == "qa_answer_total"
        ]

        def _call_totals(records: list[Dict[str, Any]]) -> Dict[str, Any]:
            return {
                "llm_calls": len(records),
                "successful_calls": sum(r.get("status") == "success" for r in records),
                "failed_calls": sum(r.get("status") != "success" for r in records),
                "input_tokens": sum(int(r.get("input_tokens", 0)) for r in records),
                "output_tokens": sum(int(r.get("output_tokens", 0)) for r in records),
                "total_tokens": sum(int(r.get("total_tokens", 0)) for r in records),
                "llm_latency_seconds": sum(
                    float(r.get("latency_seconds", 0.0)) for r in records
                ),
            }

        retrieval_time_seconds = (
            sum(float(item.get("latency_seconds", 0.0)) for item in retrieval_total_events)
            if retrieval_total_events
            else retrieval_llm_seconds + vector_retrieval_seconds
        )
        answer_time_seconds = (
            sum(float(item.get("latency_seconds", 0.0)) for item in answer_total_events)
            if answer_total_events
            else sum(float(call.get("latency_seconds", 0.0)) for call in answer_calls)
        )

        return {
            "conversation_id": conversation_id,
            "qa_id": qa_id,
            "retrieval_time_seconds": retrieval_time_seconds,
            "answer_time_seconds": answer_time_seconds,
            "total_time_seconds": retrieval_time_seconds + answer_time_seconds,
            "vector_retrieval_time_seconds": vector_retrieval_seconds,
            "retrieval_llm": _call_totals(retrieval_calls),
            "answer_llm": _call_totals(answer_calls),
        }

    def terminal_summary(self) -> str:
        dataset = self.snapshot()["dataset"]
        parts = []
        for stage in STAGES:
            stats = dataset["stages"][stage]
            parts.append(
                f"{stage}: calls={stats['llm_calls']} "
                f"tokens={stats['total_tokens']} "
                f"latency={stats['total_latency_seconds']:.2f}s"
            )
        return "Cost profiling: " + " | ".join(parts)


class IncrementalCostWriter:
    """Append-only, fsync-backed writer for one job-local cost JSONL file."""

    def __init__(self, output_path: str | Path, *, enabled: bool = True) -> None:
        self.output_path = Path(output_path)
        self.enabled = bool(enabled)
        self._lock = threading.Lock()

    def start(self, **metadata: Any) -> None:
        if not self.enabled:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.output_path.open("w", encoding="utf-8") as f:
                self._write_line(f, {
                    "event": "run_started",
                    "timestamp_unix": time.time(),
                    **metadata,
                })

    def append(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return
        record = {
            "event": str(event),
            "timestamp_unix": time.time(),
            **payload,
        }
        with self._lock:
            with self.output_path.open("a", encoding="utf-8") as f:
                self._write_line(f, record)

    @staticmethod
    def _write_line(file_obj, record: Dict[str, Any]) -> None:
        # One serialized write plus flush/fsync keeps every previously completed
        # line durable if the process or node terminates before the next event.
        file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
