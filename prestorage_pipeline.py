from __future__ import annotations

import contextlib
import json
import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from precompression import build_step1_record
from cost_profiler import (
    CostProfiler,
    STAGE_PREMEM_STEP1,
)


LOGGER = logging.getLogger("amem_prestorage")
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

MEMORY_BUILD_MODE_STEP1_PROMPT_ONLY = "step1_prompt_only"



STEP1_NOTE_PROMPT="""
Generate a concise, retrieval-oriented memory representation for one dialogue message.

[Input]
Conversation-time anchor (reference only; not the event time):
{{$conversation_time}}
Speaker: {{$speaker}}
Message: {{$message}}



[Core Requirements]

1. Use only information explicitly supported by the provided input.
   * Do not invent missing people, events, objects, locations, relationships, intentions, or causes.
   * Do not infer a fact merely because it is plausible.

2. Preserve the correct subject and event roles.

   * A question asked by the speaker is not evidence that the speaker has the queried preference, experience, possession, or intention.
   * A compliment given by the speaker is not evidence that the praised action or attribute belongs to the speaker.
   * Do not interpret "you" as the speaker.
   * Do not interpret "we", "us", "they", or another pronoun unless its referent is clearly established within the current message.

3. Extract the most retrieval-useful concrete information.
   Prioritize:

   * specific objects, activities, works, organizations, places, and topics;
   * preferences, goals, plans, experiences, relationships, causes, and outcomes when explicitly stated;
   * details that distinguish this message from other messages about the same person or topic.

4. Preserve event status accurately.
   Distinguish between:

   * completed events;
   * current states or ongoing activities;
   * habitual actions;
   * future plans or intentions;
   * possibilities, suggestions, and uncertainty.

   Do not convert "might", "thinking about", "planning", "want to", or a suggestion into a completed or certain event.

5. Normalize time only when the required temporal anchor is available.

   * Use the provided Conversation time as the reference point.
   * Convert clear relative expressions such as "yesterday", "tomorrow", "last Saturday", "two days ago", "last month", "last year", and "next month" into an absolute date, month, year, or supported date range.
   * Perform ordinary calendar calculation rather than copying the conversation date.
   * If the expression cannot be resolved reliably, preserve the original relative expression and do not guess.

6. Handle non-factual messages conservatively.

   * For a pure greeting, acknowledgement, generic praise, or question that introduces no new factual information, do not invent a surrounding event or answer.

7. Optimize fields for retrieval without unnecessary repetition.

   * `keywords` should normally contain 3 to 6 specific event-discriminating terms.
   * Do not include the speaker name in `keywords` unless multiple people are explicitly involved and the name is necessary for disambiguation.
   * Do not use generic terms such as "personal experience", "completed event", "future plan", "conversation", or "information" as keywords.
   * Put broad classifications and event-status labels in `tags`, not in `keywords`.
   * Do not include unresolved relative time expressions in `keywords`.
   * `context` and `summary` must state facts directly, not describe what the conversation discusses.
   * Avoid phrases such as "the conversation explores", "the speaker reflects on", "an inquiry is made", or "the content is about".
8. Use the conversation-time anchor only when Message itself contains an
explicit relative-time expression. Otherwise, do not mention or copy the
conversation time into keywords, context, summary, or tags.


[Output Format]

Return exactly one valid JSON object with the following fields, names, hierarchy, and data types:

{
"keywords": [
"specific event, object, place, organization, activity, or distinguishing concept",
"ordered from most to least useful for retrieval"
],
"context": "one complete factual sentence preserving the explicit subject, action, object, status, and supported time",
"tags": [
"two to five broad domain, information-type, or event-status categories"
],
"summary": "one short standalone factual sentence optimized for retrieval"
}

[Field Rules]

`keywords`:

* Prefer specific nouns and short noun phrases.
* Prefer the event object and distinguishing details over generic themes.
* Avoid duplicating the same concept with minor wording changes.
* Do not include unsupported facts.

`context`:

* Use one sentence.
* Explicitly name the speaker when the speaker is the subject of the fact.
* Include the central fact and only the most useful supporting detail.
* Prefer approximately 12 to 28 words when possible.
* Do not combine unrelated facts merely to maximize coverage.

`tags`:

* Use broad categories such as "art", "family", "career", "mental health", "travel", "preference", "completed event", "ongoing activity", or "future plan".
* Keep tags general and consistent.
* Do not copy the full keywords list into tags.

`summary`:

* Use a complete, independent factual sentence.
* Explicitly identify the relevant person when needed.
* Preserve a normalized date or approximate period when time is central.
* Prefer approximately 8 to 18 words.
* Do not repeat `context` verbatim unless the message contains only one very short fact.

[Few-Shot Example]

Input:
Conversation time: 25 May 2023, 1:14 pm
Speaker: Melanie
Message: My kids are so excited about summer break! We're thinking about going camping next month.

Output:
{
"keywords": [
"family camping trip",
"summer break",
"June 2023"
],
"context": "Melanie and her children are considering a family camping trip in June 2023 during summer break.",
"tags": [
"family",
"camping",
"future plan"
],
"summary": "Melanie is considering a family camping trip with her children in June 2023."
}

Now process the provided input. Return only the JSON object.


"""


# STEP1_NOTE_PROMPT = """Generate a structured analysis of the following content by:
# 1. Identifying the most salient keywords.
# 2. Extracting core themes and contextual elements.
# 3. Creating relevant categorical tags.
# 4. Writing a concise retrieval-oriented summary sentence.

# Format the response as a single JSON object:
# {
#   "keywords": [
#     "several specific, distinct keywords that capture key concepts and terminology",
#     "order from most to least important",
#     "do not include speaker names or time references"
#   ],
#   "context": "one sentence summarizing the main topic, key points, and purpose",
#   "tags": [
#     "several broad categories/themes for classification",
#     "include domain, format, and type tags where appropriate"
#   ],
#   "summary": "one concise retrieval-oriented summary sentence"
# }

# Summary rules:
# - Focus on concrete details useful for later memory retrieval.
# - Prefer 18 words or fewer when possible.
# - Avoid repeating the context sentence verbatim unless necessary.

# Content for analysis:
# {{$message}}"""


STEP1_NOTE_PROMPT_NO_PRONOUNS_RELATIVE_TIME = """Generate a structured analysis of the following content by:
1. Identifying the most salient keywords.
2. Extracting core themes and contextual elements.
3. Creating relevant categorical tags.
4. Writing a concise retrieval-oriented summary sentence.
5. Absolutely PROHIBIT using pronouns (he, she, it, they, this, that) and relative time (yesterday, today, last week, tomorrow)
6. Lossless Information**: Each entry's lossless_restatement must be a complete, independent, understandable sentence

Format the response as a single JSON object:
{
  "keywords": [
    "several specific, distinct keywords that capture key concepts and terminology",
    "order from most to least important",
    "do not include speaker names or time references"
  ],
  "context": "one sentence summarizing the main topic, key points, and purpose",
  "tags": [
    "several broad categories/themes for classification",
    "include domain, format, and type tags where appropriate"
  ],
  "summary": "one concise retrieval-oriented summary sentence"
}

Summary rules:
- Focus on concrete details useful for later memory retrieval.
- Prefer 18 words or fewer when possible.
- Avoid repeating the context sentence verbatim unless necessary.

Content for analysis:
{{$message}}"""







def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")






def _replace_rows_for_sample(
    path: Path,
    sample_id: int | str,
    new_rows: list[Dict[str, Any]],
) -> None:
    sample_id_str = str(sample_id)
    rows = [
        row for row in _read_jsonl(path) if str(row.get("sample_id")) != sample_id_str
    ]
    rows.extend(new_rows)
    _write_jsonl(path, rows)


def _extract_json_object(text: str) -> dict:
    text = str(text).strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        last_brace = text.rfind("}")
        if last_brace != -1:
            candidate = text[: last_brace + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
    return {}


def _truncate_words(text: str, max_words: int) -> str:
    words = str(text).split()
    if len(words) <= max_words:
        return str(text).strip()
    return " ".join(words[:max_words]).strip()


def _heuristic_keywords(text: str, limit: int = 5) -> list[str]:
    tokens = []
    seen = set()
    for token in WORD_RE.findall(str(text).lower()):
        if len(token) <= 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens or ["memory"]


def _heuristic_summary(text: str, max_words: int = 18) -> str:
    return _truncate_words(" ".join(str(text).split()), max_words=max_words) or "General memory."


def _heuristic_context(text: str, max_words: int = 24) -> str:
    return _truncate_words(" ".join(str(text).split()), max_words=max_words) or "General memory."


def _infer_tags(keywords: list[str]) -> list[str]:
    tags = []
    for keyword in keywords[:3]:
        normalized = keyword.strip().lower()
        if normalized and normalized not in tags:
            tags.append(normalized)
    return tags or ["memory"]






@dataclass
class Step1NoteRecord:
    sample_id: int | str
    session_id: int | str
    session_date: str
    role: str
    message_id: Optional[str]
    original_text: str
    compress_text: str
    keywords: list[str]
    summary: str
    context: str
    tags: list[str]
    note_content: str
    response: dict
    was_compressed: bool
    skip_reason: Optional[str]
    input_tokens: int
    output_tokens: int



    @property
    def embed_text(self) -> str:
        keyword_text = ", ".join(self.keywords[:5]) if self.keywords else "memory"
        tag_text = ", ".join(self.tags[:5]) if self.tags else "memory"
        return (
            f"Keywords: {keyword_text}. "
            f"Context: {self.context}. "
            f"Summary: {self.summary}. "
            f"Tags: {tag_text}."
        )

    def to_cache_row(self) -> Dict[str, Any]:
        return {
            "cache_stage": "prestorage_step1_note",
            "sample_id": str(self.sample_id),
            "session_id": self.session_id,
            "session_date": self.session_date,
            "role": self.role,
            "message_id": self.message_id,
            "conversation": self.original_text,
            "compress_text": self.compress_text,
            "note_content": self.note_content,
            "was_compressed": self.was_compressed,
            "skip_reason": self.skip_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "response": json.dumps(self.response, ensure_ascii=False),
        }


@dataclass
class DirectStep1Memory:
    """Passive Step 1 record used when A-Mem is ablated."""

    id: str
    content: str
    timestamp: str
    keywords: list[str]
    context: str
    tags: list[str]
    retrieval_text: str
    category: str = "PreStorageStep1"
    links: list[int] = field(default_factory=list)


class PreStoragePipeline:
    def __init__(
        self,
        *,
        backend: str,
        model: str,
        model_type: str = "qwen25",
        cache_dir: str | Path,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
        usage_tracker=None,
        cost_profiler: Optional[CostProfiler] = None,
        use_existing_step1_note_prompt: bool = True,
    ) -> None:
        from memory_layer_robust import RobustLLMController

        self.cache_dir = Path(cache_dir)
        self.usage_tracker = usage_tracker
        self.cost_profiler = cost_profiler
        self.step1_note_prompt = (
            STEP1_NOTE_PROMPT
            if use_existing_step1_note_prompt
            else STEP1_NOTE_PROMPT_NO_PRONOUNS_RELATIVE_TIME
        )
        self.controller = RobustLLMController(
            backend=backend,
            model=model,
            model_type=model_type,
            api_key=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
            usage_tracker=usage_tracker,
            usage_op_name="prestorage_step1_metadata",
            cost_profiler=cost_profiler,
        )
        self.step1_jsonl = self.cache_dir / "premem_step1_ours.jsonl"
        self._force_rebuild_prepared = False
        print(
            "[DEBUG_QWEN_COMPARE] PRESTORAGE_INIT "
            f"backend={backend} model={model} model_type={model_type} cache_dir={self.cache_dir} "
            f"host={sglang_host} port={sglang_port} "
            f"use_existing_step1_note_prompt={use_existing_step1_note_prompt}"
        )

    def set_usage_tracker(self, usage_tracker) -> None:
        self.usage_tracker = usage_tracker
        self.controller.usage_tracker = usage_tracker

    def _profile_scope(self, **kwargs):
        if self.cost_profiler is None or not self.cost_profiler.enabled:
            return contextlib.nullcontext()
        return self.cost_profiler.scope(**kwargs)

    def prepare_force_rebuild(self) -> None:
        """Remove Step 1 pre-storage artifacts once before a forced run."""
        if self._force_rebuild_prepared:
            return

        removed_paths = []
        for path in (self.step1_jsonl,):
            if path.exists():
                path.unlink()
                removed_paths.append(str(path))
        self._force_rebuild_prepared = True
        print(
            "[DEBUG_QWEN_COMPARE] PRESTORAGE_FORCE_REBUILD_PREPARED "
            f"cache_dir={self.cache_dir} "
            f"memory_build_mode={MEMORY_BUILD_MODE_STEP1_PROMPT_ONLY} "
            f"removed_count={len(removed_paths)} "
            f"removed_files={removed_paths}"
        )





    def _call_llm(self, op_name: str, prompt: str, temperature: float = 0.0) -> str:
        self.controller.usage_op_name = op_name
        try:
            return self.controller.get_completion(prompt, temperature=temperature)
        finally:
            self.controller.usage_op_name = op_name

    def _build_step1_metadata_input(self, *, role: str, compress_text: str) -> str:
        role = str(role).strip()
        compress_text = str(compress_text).strip()
        if not role:
            return compress_text
        return f"Speaker {role} says: {compress_text}"

    def _extract_step1_metadata(
        self,
        message_text: str,
        *,
        conversation_time: str,
        speaker: str,
    ) -> dict:
        prompt = (
            self.step1_note_prompt
            .replace("{{$conversation_time}}", str(conversation_time).strip())
            .replace("{{$speaker}}", str(speaker).strip())
            .replace("{{$message}}", message_text)
        )
        response = self._call_llm("prestorage_step1_metadata", prompt, temperature=0.0)
        parsed = _extract_json_object(response)
        keywords = parsed.get("keywords")
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(item).strip() for item in keywords if str(item).strip()]
        context = str(parsed.get("context", "")).strip()
        summary = str(parsed.get("summary", "")).strip()
        tags = parsed.get("tags")
        if not isinstance(tags, list):
            tags = []
        tags = [str(item).strip() for item in tags if str(item).strip()]

        if not keywords:
            keywords = _heuristic_keywords(message_text)
        if not context:
            context = _heuristic_context(message_text)
        if not summary:
            summary = _heuristic_summary(message_text)
        if not tags:
            tags = _infer_tags(keywords)

        return {
            "keywords": keywords,
            "context": context,
            "summary": summary,
            "tags": tags,
        }

    def _build_step1_note_content(
        self,
        *,
        role: str,
        compress_text: str,
        context: str,
        summary: str,
        keywords: list[str],
    ) -> str:
        keyword_text = ", ".join(keywords[:5]) if keywords else "memory"
        return (
            f"Speaker {role} says: {compress_text}\n"
            f"Context: {context}\n"
            f"Summary: {summary}\n"
            f"Keywords: {keyword_text}"
        )

    def _upgrade_cached_step1_rows(
        self,
        sample_idx: int,
        cached_rows: list[dict],
    ) -> list[Step1NoteRecord]:
        records: list[Step1NoteRecord] = []
        refreshed = False
        for row in cached_rows:
            compress_text = str(
                row.get("compress_text")
                or row.get("memory_content")
                or row.get("conversation", "")
            ).strip()
            if not compress_text:
                return []
            metadata_input = self._build_step1_metadata_input(
                role=str(row.get("role", "")),
                compress_text=compress_text,
            )
            conversation_time = str(row.get("session_date", "")).strip()
            speaker = str(row.get("role", "")).strip()
            response = _extract_json_object(row.get("response", "{}"))
            keywords = response.get("keywords")
            context = response.get("context")
            summary = response.get("summary")
            tags = response.get("tags")
            note_content = response.get("note_content", row.get("note_content", ""))
            cached_metadata_input = str(response.get("metadata_input", "")).strip()
            cached_conversation_time = str(
                response.get("conversation_time", "")
            ).strip()
            cached_speaker = str(response.get("speaker", "")).strip()

            if (
                not isinstance(keywords, list)
                or not keywords
                or not str(context or "").strip()
                or not str(summary or "").strip()
                or cached_metadata_input != metadata_input
                or cached_conversation_time != conversation_time
                or cached_speaker != speaker
            ):
                with self._profile_scope(
                    stage=STAGE_PREMEM_STEP1,
                    session_id=row.get("session_id"),
                    turn_id=row.get("message_id"),
                    item_type="turn",
                    cache_refresh=True,
                ):
                    metadata = self._extract_step1_metadata(
                        metadata_input,
                        conversation_time=conversation_time,
                        speaker=speaker,
                    )
                keywords = metadata["keywords"]
                context = metadata["context"]
                summary = metadata["summary"]
                tags = metadata["tags"]
                note_content = self._build_step1_note_content(
                    role=str(row.get("role", "")),
                    compress_text=compress_text,
                    context=context,
                    summary=summary,
                    keywords=keywords,
                )
                response = {
                    "metadata_input": metadata_input,
                    "conversation_time": conversation_time,
                    "speaker": speaker,
                    "keywords": keywords,
                    "context": context,
                    "summary": summary,
                    "tags": tags,
                    "note_content": note_content,
                }
                refreshed = True
            else:
                keywords = [str(item).strip() for item in keywords if str(item).strip()]
                context = str(context).strip()
                tags = tags if isinstance(tags, list) else []
                tags = [str(item).strip() for item in tags if str(item).strip()]
                summary = str(summary).strip()
                if not note_content:
                    note_content = self._build_step1_note_content(
                        role=str(row.get("role", "")),
                        compress_text=compress_text,
                        context=context,
                        summary=summary,
                        keywords=keywords,
                    )
                    response["note_content"] = note_content
                    refreshed = True
                if cached_metadata_input != metadata_input:
                    response["metadata_input"] = metadata_input
                    refreshed = True

            if "prestorage_step1" not in tags:
                tags = tags + ["prestorage_step1"]
                response["tags"] = tags
                refreshed = True

            records.append(
                Step1NoteRecord(
                    sample_id=sample_idx,
                    session_id=row.get("session_id"),
                    session_date=row.get("session_date", ""),
                    role=row.get("role", ""),
                    message_id=row.get("message_id"),
                    original_text=row.get("conversation", ""),
                    compress_text=compress_text,
                    keywords=keywords,
                    summary=summary,
                    context=context,
                    tags=tags,
                    note_content=str(note_content),
                    response=response,
                    was_compressed=bool(row.get("was_compressed", False)),
                    skip_reason=row.get("skip_reason"),
                    input_tokens=int(row.get("input_tokens", 0)),
                    output_tokens=int(row.get("output_tokens", 0)),
                )
            )

        if refreshed and records:
            _replace_rows_for_sample(
                self.step1_jsonl,
                sample_idx,
                [record.to_cache_row() for record in records],
            )
        return records

    def _build_step1_records(
        self,
        sample_idx: int,
        sample,
        compressor,
        *,
        force_rebuild: bool = False,
    ) -> list[Step1NoteRecord]:
        cached_rows = []
        if not force_rebuild:
            cached_rows = [
                row for row in _read_jsonl(self.step1_jsonl)
                if str(row.get("sample_id")) == str(sample_idx)
            ]
        print(
            "[DEBUG_QWEN_COMPARE] STEP1_START "
            f"sample_id={sample_idx} cached_rows={len(cached_rows)} "
            f"session_count={len(sample.conversation.sessions)}"
        )
        if cached_rows:
            records = self._upgrade_cached_step1_rows(sample_idx, cached_rows)
            if records:
                print(
                    "[DEBUG_QWEN_COMPARE] STEP1_CACHE_HIT "
                    f"sample_id={sample_idx} records={len(records)} cache_file={self.step1_jsonl}"
                )
                return records

        records: list[Step1NoteRecord] = []
        session_items = sorted(sample.conversation.sessions.items(), key=lambda item: item[0])
        for _, session in session_items:
            for turn in session.turns:
                compressed = build_step1_record(
                    turn.speaker,
                    turn.text,
                    compressor,
                    session_id=session.session_id,
                    session_date=session.date_time,
                    message_id=turn.dia_id,
                )
                metadata_input = self._build_step1_metadata_input(
                    role=turn.speaker,
                    compress_text=compressed.compress_text,
                )
                with self._profile_scope(
                    stage=STAGE_PREMEM_STEP1,
                    session_id=session.session_id,
                    turn_id=turn.dia_id,
                    item_type="turn",
                    speaker=turn.speaker,
                ):
                    metadata = self._extract_step1_metadata(
                        metadata_input,
                        conversation_time=session.date_time,
                        speaker=turn.speaker,
                    )
                note_content = self._build_step1_note_content(
                    role=turn.speaker,
                    compress_text=compressed.compress_text,
                    context=metadata["context"],
                    summary=metadata["summary"],
                    keywords=metadata["keywords"],
                )
                response = {
                    "metadata_input": metadata_input,
                    "conversation_time": str(session.date_time).strip(),
                    "speaker": str(turn.speaker).strip(),
                    "keywords": metadata["keywords"],
                    "context": metadata["context"],
                    "summary": metadata["summary"],
                    "tags": metadata["tags"],
                    "note_content": note_content,
                }
                # Field lengths expose speaker loss and model-dependent metadata expansion safely.
                print(
                    "[DEBUG_QWEN_COMPARE] STEP1_RECORD "
                    f"sample_id={sample_idx} session_id={session.session_id} message_id={turn.dia_id} "
                    f"speaker_present={bool(str(turn.speaker).strip())} speaker_chars={len(str(turn.speaker))} "
                    f"original_chars={len(compressed.original_text)} compress_chars={len(compressed.compress_text)} "
                    f"metadata_input_chars={len(metadata_input)} summary_chars={len(metadata['summary'])} "
                    f"context_chars={len(metadata['context'])} keywords_count={len(metadata['keywords'])} "
                    f"keywords_chars={len(str(metadata['keywords']))} tags_count={len(metadata['tags'])} "
                    f"tags_chars={len(str(metadata['tags']))} note_content_chars={len(note_content)} "
                    f"speaker_prefix_preserved={note_content.startswith(f'Speaker {turn.speaker} says:')} "
                    f"was_compressed={compressed.was_compressed} input_tokens={compressed.input_tokens} "
                    f"output_tokens={compressed.output_tokens}"
                )
                records.append(
                    Step1NoteRecord(
                        sample_id=sample_idx,
                        session_id=session.session_id,
                        session_date=session.date_time,
                        role=turn.speaker,
                        message_id=turn.dia_id,
                        original_text=compressed.original_text,
                        compress_text=compressed.compress_text,
                        keywords=metadata["keywords"],
                        summary=metadata["summary"],
                        context=metadata["context"],
                        tags=list(dict.fromkeys(metadata["tags"] + ["prestorage_step1"])),
                        note_content=note_content,
                        response=response,
                        was_compressed=compressed.was_compressed,
                        skip_reason=compressed.skip_reason,
                        input_tokens=compressed.input_tokens,
                        output_tokens=compressed.output_tokens,
                    )
                )

        _replace_rows_for_sample(
            self.step1_jsonl,
            sample_idx,
            [record.to_cache_row() for record in records],
        )
        print(
            "[DEBUG_QWEN_COMPARE] STEP1_END "
            f"sample_id={sample_idx} records={len(records)} cache_hit=False cache_file={self.step1_jsonl}"
        )
        return records









def populate_direct_step1_memory(agent, step1_records: list[Step1NoteRecord]) -> None:
    """Build a passive retrieval corpus directly from Step 1 outputs."""
    memories: dict[str, DirectStep1Memory] = {}
    retrieval_documents: list[str] = []
    for index, record in enumerate(step1_records):
        memory_id = f"direct-step1-{index}"
        memories[memory_id] = DirectStep1Memory(
            id=memory_id,
            content=record.note_content,
            timestamp=record.session_date,
            keywords=list(record.keywords),
            context=record.context,
            tags=list(record.tags),
            retrieval_text=record.embed_text,
        )
        retrieval_documents.append(record.embed_text)

    agent.memory_system.memories = memories
    if retrieval_documents:
        agent.memory_system.retriever.add_documents(retrieval_documents)


def find_direct_step1_memories_raw(memory_system, query: str, k: int = 5) -> str:
    """Retrieve passive records directly, without A-Mem link expansion."""
    if not memory_system.memories:
        return ""

    indices = memory_system.retriever.search(query, k)
    all_memories = list(memory_system.memories.values())
    blocks = []
    for raw_index in indices:
        memory = all_memories[int(raw_index)]
        blocks.append(
            "talk start time:"
            + str(memory.timestamp)
            + "memory content: "
            + memory.content
            + "memory context: "
            + memory.context
            + "memory keywords: "
            + str(memory.keywords)
            + "memory tags: "
            + str(memory.tags)
        )
    return "\n".join(blocks) + ("\n" if blocks else "")




def _retrieval_documents_from_cached_memories(memories: dict) -> list[str]:
    """Rebuild mixed caches while preserving passive records' exact text."""
    documents = []
    for memory in memories.values():
        retrieval_text = getattr(memory, "retrieval_text", None)
        if retrieval_text is not None:
            documents.append(retrieval_text)
            continue
        metadata_text = (
            f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
        )
        documents.append(f"{memory.content} , {metadata_text}")
    return documents




def populate_agent_memory_from_prestorage(
    *,
    agent,
    sample,
    sample_idx: int,
    cache_dir: str | Path,
    compressor,
    pipeline: PreStoragePipeline,
    force_rebuild_cache: bool = False,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_suffix = "_step1_prompt_only"
    memory_cache_file = cache_dir / f"memory_cache{cache_suffix}_sample_{sample_idx}.pkl"  # 记忆原文和元信息
    memory_cache_meta_file = cache_dir / f"memory_cache{cache_suffix}_sample_{sample_idx}.meta.json"
    retriever_cache_file = cache_dir / f"retriever_cache{cache_suffix}_sample_{sample_idx}.pkl"  # 检索器配置、索引结构、memory id 映射等
    retriever_cache_embeddings_file = cache_dir / f"retriever_cache_embeddings{cache_suffix}_sample_{sample_idx}.npy"  # 每条记忆对应的向量矩阵

    cache_metadata_loaded = False
    if not force_rebuild_cache and memory_cache_meta_file.exists():
        try:
            with memory_cache_meta_file.open("r", encoding="utf-8") as f:
                cache_metadata = json.load(f)
            cache_metadata_loaded = (
                cache_metadata.get("memory_build_mode")
                == MEMORY_BUILD_MODE_STEP1_PROMPT_ONLY
            )
        except Exception:
            cache_metadata_loaded = False

    if force_rebuild_cache:
        pipeline.prepare_force_rebuild()
        for path in (
            memory_cache_file,
            memory_cache_meta_file,
            retriever_cache_file,
            retriever_cache_embeddings_file,
        ):
            path.unlink(missing_ok=True)

    cache_ready = (
        memory_cache_file.exists()
        and cache_metadata_loaded
        and not force_rebuild_cache
    )
    print(
        "[DEBUG_QWEN_COMPARE] MEMORY_POPULATE_START "
        f"sample_id={sample_idx} force_rebuild_cache={force_rebuild_cache} "
        f"memory_build_mode={MEMORY_BUILD_MODE_STEP1_PROMPT_ONLY} "
        f"memory_cache_exists={memory_cache_file.exists()} "
        f"cache_metadata_loaded={cache_metadata_loaded} "
        f"retriever_cache_exists={retriever_cache_file.exists()} "
        f"cache_ready={cache_ready} cache_dir={cache_dir}"
    )
    if cache_ready:
        with memory_cache_file.open("rb") as f:
            cached_memories = pickle.load(f)
        agent.memory_system.memories = cached_memories
        if retriever_cache_file.exists():
            agent.memory_system.retriever = agent.memory_system.retriever.load(
                str(retriever_cache_file),
                str(retriever_cache_embeddings_file),
            )
        else:
            retrieval_documents = _retrieval_documents_from_cached_memories(
                cached_memories
            )
            if retrieval_documents:
                agent.memory_system.retriever.add_documents(retrieval_documents)
        print(
            "[DEBUG_QWEN_COMPARE] MEMORY_POPULATE_END "
            f"sample_id={sample_idx} source=cache memories={len(agent.memory_system.memories)} "
            f"retriever_corpus={len(agent.memory_system.retriever.corpus)}"
        )
        return

    step1_records = pipeline._build_step1_records(
        sample_idx,
        sample,
        compressor,
        force_rebuild=force_rebuild_cache,
    )
    populate_direct_step1_memory(agent, step1_records)
    print(
        "[DEBUG_QWEN_COMPARE] STEP1_MEMORY_DIRECT "
        f"sample_id={sample_idx} records={len(step1_records)} "
        f"memories={len(agent.memory_system.memories)} "
        f"retriever_corpus={len(agent.memory_system.retriever.corpus)}"
    )
    with memory_cache_file.open("wb") as f:
        pickle.dump(agent.memory_system.memories, f)
    with memory_cache_meta_file.open("w", encoding="utf-8") as f:
        json.dump(
            {"memory_build_mode": MEMORY_BUILD_MODE_STEP1_PROMPT_ONLY},
            f,
        )
    agent.memory_system.retriever.save(
        str(retriever_cache_file),
        str(retriever_cache_embeddings_file),
    )
    print(
        "[DEBUG_QWEN_COMPARE] MEMORY_POPULATE_END "
        f"sample_id={sample_idx} source=rebuild "
        f"memory_build_mode={MEMORY_BUILD_MODE_STEP1_PROMPT_ONLY} "
        f"step1_records={len(step1_records)} "
        f"memories={len(agent.memory_system.memories)} "
        f"retriever_corpus={len(agent.memory_system.retriever.corpus)}"
    )
