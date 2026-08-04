from __future__ import annotations

import importlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


LOGGER = logging.getLogger("amem_precompress")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(str(text)))


@dataclass
class CompressionStats:
    turns_seen: int = 0
    turns_compressed: int = 0
    turns_skipped_short: int = 0
    turns_skipped_image: int = 0
    turns_skipped_disabled: int = 0
    turns_failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        ratio = 1.0
        if self.input_tokens > 0:
            ratio = self.output_tokens / self.input_tokens
        return {
            "turns_seen": self.turns_seen,
            "turns_compressed": self.turns_compressed,
            "turns_skipped_short": self.turns_skipped_short,
            "turns_skipped_image": self.turns_skipped_image,
            "turns_skipped_disabled": self.turns_skipped_disabled,
            "turns_failed": self.turns_failed,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "effective_token_ratio": ratio,
        }


@dataclass
class CompressionDecision:
    compress_text: str
    was_compressed: bool
    skip_reason: Optional[str]
    input_tokens: int
    output_tokens: int


@dataclass
class Step1MessageRecord:
    """PREMem Step 1 style message-level preconstruction output."""

    speaker: str
    original_text: str
    compress_text: str
    memory_content: str
    session_id: Optional[int | str] = None
    session_date: Optional[str] = None
    message_id: Optional[str] = None
    was_compressed: bool = False
    skip_reason: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0



class LLMLinguaTurnCompressor:
    """
    PREMem Step 1 style history preconstructor.

    The original PREMem Step 1 extracts structured personal information from the
    dialogue history. In this project we intentionally replace that extraction
    step with single-message LLMLingua compression, so the Step 1 output for
    each message is simply `compress_text`.
    """

    def __init__(
        self,
        enabled: bool = False,
        model_name_or_path: Optional[str] = None,
        device_map: str = "cpu",
        rate: float = 0.9,
        target_token: int = -1,
        min_tokens: int = 24,
        instruction: str = "",
        skip_image_turns: bool = True,
    ) -> None:
        self.enabled = enabled
        self.model_name_or_path = model_name_or_path
        self.device_map = device_map
        self.rate = rate
        self.target_token = target_token
        self.min_tokens = min_tokens
        self.instruction = instruction
        self.skip_image_turns = skip_image_turns
        self.stats = CompressionStats()
        self._compressor = None
        self._disabled_reason: Optional[str] = None

    @property
    def effective_enabled(self) -> bool:
        return self.enabled and bool(self.model_name_or_path) and self._disabled_reason is None

    def cache_tag(self) -> str:
        if not self.effective_enabled:
            return "plain"
        rate_tag = f"{int(self.rate * 100):03d}"
        return f"llmlingua_r{rate_tag}_min{self.min_tokens}"

    def config_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "effective_enabled": self.effective_enabled,
            "step1_mode": "premem_single_message_compression",
            "output_field": "compress_text",
            "model_name_or_path": self.model_name_or_path,
            "device_map": self.device_map,
            "rate": self.rate,
            "target_token": self.target_token,
            "min_tokens": self.min_tokens,
            "instruction": self.instruction,
            "skip_image_turns": self.skip_image_turns,
            "disabled_reason": self._disabled_reason,
            "cache_tag": self.cache_tag(),
            "stats": self.stats.to_dict(),
        }



    def _should_skip(self, text: str) -> Optional[str]:
        if not self.enabled:
            return "disabled"
        if not self.model_name_or_path:
            return "missing_model_path"
        if self.skip_image_turns and "[Image:" in text:
            return "image_turn"
        if count_tokens(text) < self.min_tokens:
            return "short_turn"
        return None

    def _register_skip(self, text: str, skip_reason: str, input_tokens: int) -> CompressionDecision:
        if skip_reason in {"disabled", "missing_model_path"} or skip_reason.startswith("init_failed"):
            self.stats.turns_skipped_disabled += 1
        elif skip_reason == "image_turn":
            self.stats.turns_skipped_image += 1
        elif skip_reason == "short_turn":
            self.stats.turns_skipped_short += 1
        elif skip_reason == "compression_failed":
            self.stats.turns_failed += 1
        self.stats.output_tokens += input_tokens
        return CompressionDecision(
            compress_text=text,
            was_compressed=False,
            skip_reason=skip_reason,
            input_tokens=input_tokens,
            output_tokens=input_tokens,
        )

    def _compress_to_decision(self, text: str) -> CompressionDecision:
        text = str(text)
        self.stats.turns_seen += 1
        input_tokens = count_tokens(text)
        self.stats.input_tokens += input_tokens

        skip_reason = self._should_skip(text)
        if skip_reason is not None:
            return self._register_skip(text, skip_reason, input_tokens)

        compress_config = {
            "context": [text],
            "instruction": self.instruction,
            "rate": self.rate,
            "target_token": self.target_token,
        }
        try:
            compressed = self._compressor.compress_prompt(**compress_config)["compressed_prompt"]
            compressed = str(compressed).strip()
        except Exception as exc:
            LOGGER.warning("LLMLingua compression failed, using original text: %s", exc)
            return self._register_skip(text, "compression_failed", input_tokens)

        if not compressed:
            return self._register_skip(text, "compression_failed", input_tokens)

        output_tokens = count_tokens(compressed)
        if output_tokens <= 0 or output_tokens >= input_tokens:
            self.stats.output_tokens += input_tokens
            return CompressionDecision(
                compress_text=text,
                was_compressed=False,
                skip_reason="not_shorter",
                input_tokens=input_tokens,
                output_tokens=input_tokens,
            )

        self.stats.turns_compressed += 1
        self.stats.output_tokens += output_tokens
        return CompressionDecision(
            compress_text=compressed,
            was_compressed=True,
            skip_reason=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


    def build_step1_record(
        self,
        *,
        speaker: str,
        turn_text: str,
        session_id: Optional[int | str] = None,
        session_date: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> Step1MessageRecord:
        decision = self._compress_to_decision(turn_text)
        memory_content = f"Speaker {speaker}says : {decision.compress_text}"
        return Step1MessageRecord(
            speaker=str(speaker),
            original_text=str(turn_text),
            compress_text=decision.compress_text,
            memory_content=memory_content,
            session_id=session_id,
            session_date=session_date,
            message_id=message_id,
            was_compressed=decision.was_compressed,
            skip_reason=decision.skip_reason,
            input_tokens=decision.input_tokens,
            output_tokens=decision.output_tokens,
        )


def add_precompression_args(parser) -> None:
    parser.add_argument(
        "--enable-precompress",
        action="store_true",
        help=(
            "Enable PREMem Step 1 style single-message preconstruction with "
            "LLMLingua before A-MEM note construction."
        ),
    )
    parser.add_argument(
        "--llmlingua-model-dir",
        type=str,
        default=None,
        help="Local path to microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank.",
    )
    parser.add_argument(
        "--llmlingua-device",
        type=str,
        default="cpu",
        help="Device map passed to LLMLingua PromptCompressor.",
    )
    parser.add_argument(
        "--llmlingua-rate",
        type=float,
        default=0.9,
        help="Conservative compression rate. Higher keeps more text.",
    )
    parser.add_argument(
        "--llmlingua-target-token",
        type=int,
        default=-1,
        help="Optional LLMLingua target token budget per turn. -1 disables it.",
    )
    parser.add_argument(
        "--llmlingua-min-tokens",
        type=int,
        default=24,
        help="Skip compression for short turns below this token count.",
    )


def build_turn_compressor_from_args(args) -> LLMLinguaTurnCompressor:
    model_dir = getattr(args, "llmlingua_model_dir", None)
    if model_dir is not None:
        model_dir = str(Path(model_dir).expanduser())
    return LLMLinguaTurnCompressor(
        enabled=bool(getattr(args, "enable_precompress", False)),
        model_name_or_path=model_dir,
        device_map=getattr(args, "llmlingua_device", "cpu"),
        rate=float(getattr(args, "llmlingua_rate", 0.9)),
        target_token=int(getattr(args, "llmlingua_target_token", -1)),
        min_tokens=int(getattr(args, "llmlingua_min_tokens", 24)),
    )


def build_step1_record(
    speaker: str,
    turn_text: str,
    compressor: Optional[LLMLinguaTurnCompressor],
    *,
    session_id: Optional[int | str] = None,
    session_date: Optional[str] = None,
    message_id: Optional[str] = None,
) -> Step1MessageRecord:
    if compressor is None:
        text = str(turn_text)
        token_count = count_tokens(text)
        return Step1MessageRecord(
            speaker=str(speaker),
            original_text=text,
            compress_text=text,
            memory_content=f"Speaker {speaker}says : {text}",
            session_id=session_id,
            session_date=session_date,
            message_id=message_id,
            was_compressed=False,
            skip_reason="no_compressor_object",
            input_tokens=token_count,
            output_tokens=token_count,
        )
    return compressor.build_step1_record(
        speaker=speaker,
        turn_text=turn_text,
        session_id=session_id,
        session_date=session_date,
        message_id=message_id,
    )


