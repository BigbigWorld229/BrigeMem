from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from dateutil import parser as date_parser

LOGGER = logging.getLogger("amem_prestorage")
RELATION_MEMORY_CACHE_SCHEMA = 6
STEP1_CROSS_SESSION_PROMPT_VERSION = 2
STEP1_CROSS_SESSION_SCHEMA_VERSION = 2
RELATION_ROUTER_MEMORY_TYPE = "cross_session_relation_router"
SIMILARITY_SKIP_THRESHOLD = 0.85
MAX_PATCH_TURNS = 4
PATCH_TURN_OVERLAP = 0
RELATION_TYPES = frozenset({
    "SET_LIST_UNION", "COUNT_FREQUENCY", "TRAJECTORY_RECURRENCE",
    "ATTRIBUTE_CAUSAL_SYNTHESIS",
})

RELATION_INFERENCE_PROMPT = """You are given several dialogue turns about the same subject.
Determine whether some turns jointly support a useful multi-turn relation.

Do NOT merge turns only because they have the same subject or facet.
First identify a narrow shared scope, then aggregate only the turns that support that scope.

Use only these relation types:

1. SET_LIST_UNION
   Combine distinct values of the same relation into a list.
   Example: Melanie camped at the beach, mountains, and forest.

2. COUNT_FREQUENCY
   Count distinct real-world occurrences of the same event or action.
   Repeated mentions of the same occurrence count once.
   Example: two distinct beach trips → count 2.

3. TRAJECTORY_RECURRENCE
   Combine evidence showing change, progression, or a recurring pattern over time.
   Preserve temporal order when relevant.
   Example: Sam gradually adopts healthy eating, exercise, running, and hiking.

4. ATTRIBUTE_CAUSAL_SYNTHESIS
   Combine complementary evidence that jointly describes an attribute, reason, motivation, or cause.
   Do not infer causality from mere co-occurrence.

Rules:

* A relation must use at least two distinct source turns.
* Do not force a relation when the turns are unrelated.
* Do not merge different scopes.
* Merge aliases or repeated descriptions of the same fact, but keep genuinely different facts.
* For counts, count distinct event instances, not mentions.
* A planned event is not an attended event unless another turn confirms it happened.
* Do not add facts that are not supported by the turns.
* `relation_text` is only a retrieval-oriented synthesis and must preserve the meaning of its sources.
* `source_turn_ids` must contain all and only the turns supporting that relation.
* One patch may produce multiple relations.
* If no valid relation exists, return an empty `relations` list.

Output only JSON:

```json
{
  "relations": [
    {
      "relation_type": "SET_LIST_UNION | COUNT_FREQUENCY | TRAJECTORY_RECURRENCE | ATTRIBUTE_CAUSAL_SYNTHESIS",
      "scope": "short description of the exact shared relation",
      "relation_text": "concise synthesis supported by the source turns",
      "source_turn_ids": ["turn_id_1", "turn_id_2"]
    }
  ]
}
```

Input:

Subject: {{$subject}}

Facet candidates: {{$facets}}

Turns:
{{$turn_patch}}
"""


@dataclass
class CrossSessionRelationMemory:
    id: str
    content: str
    timestamp: Any
    tags: list[str]
    relation_type: str
    scope: str
    relation_text: str
    subject: str
    facet_type: str
    facet_types: list[str]
    source_turn_ids: list[str]
    source_session_ids: list[Any]
    source_memory_ids: list[str]
    mention_times: list[dict]
    memory_type: str = RELATION_ROUTER_MEMORY_TYPE
    category: str = "CrossSessionRelationRouter"
    links: list[int] = field(default_factory=list)


def normalize_subject(value: Any) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def _record_subject(record: Any) -> str:
    response = getattr(record, "response", None)
    value = response.get("subject") if isinstance(response, dict) else None
    return value.strip() if isinstance(value, str) else ""


def _record_hard_entities(record: Any) -> list[dict]:
    response = getattr(record, "response", None)
    value = response.get("hard_entities") if isinstance(response, dict) else None
    return value if isinstance(value, list) else []


def _entity_types(entity: Any) -> list[str]:
    value = entity.get("types") if isinstance(entity, dict) else None
    if not isinstance(value, list):
        return []
    return [str(item).strip().upper() for item in value if str(item).strip()]


def build_subject_type_postings(step1_records: Sequence[Any]) -> OrderedDict:
    """Build full-turn postings grouped only by normalized subject and type."""
    grouped: OrderedDict[tuple[str, str], dict] = OrderedDict()
    for record_index, record in enumerate(step1_records):
        subject = _record_subject(record)
        normalized_subject = normalize_subject(subject)
        if not normalized_subject:
            continue
        hard_entities = [dict(e) for e in _record_hard_entities(record) if isinstance(e, dict)]
        types = list(dict.fromkeys(t for entity in hard_entities for t in _entity_types(entity)))
        session_id = getattr(record, "session_id", "")
        turn_id = str(getattr(record, "message_id", "") or "").strip() or f"__record_{record_index}"
        for facet_type in types:
            key = (normalized_subject, facet_type)
            group = grouped.setdefault(key, {
                "normalized_subject": normalized_subject,
                "subject": subject,
                "facet_type": facet_type,
                "postings": [],
                "_turn_keys": set(),
            })
            turn_key = (str(session_id), turn_id)
            if turn_key in group["_turn_keys"]:
                continue
            group["_turn_keys"].add(turn_key)
            group["postings"].append({
                "session_id": session_id,
                "session_time": str(getattr(record, "session_date", "") or "").strip(),
                "turn_id": turn_id,
                "record_index": record_index,
                "speaker": str(getattr(record, "role", "") or ""),
                "subject": subject,
                "hard_entities": hard_entities,
                "keywords": [str(k) for k in getattr(record, "keywords", [])],
                "original_text": str(getattr(record, "original_text", "") or ""),
                "embedding_ref": record_index,
            })
    for group in grouped.values():
        group.pop("_turn_keys", None)
        group["postings"].sort(key=lambda p: p["record_index"])
    return grouped


def _has_cross_session_support(postings: Sequence[dict]) -> bool:
    return (
        len(postings) >= 2
        and len({str(p.get("turn_id", "")) for p in postings}) >= 2
        and len({str(p.get("session_id", "")) for p in postings}) >= 2
    )


def discover_subject_type_batches(step1_records: Sequence[Any]) -> tuple[list[dict], dict]:
    grouped = build_subject_type_postings(step1_records)
    batches, groups = [], []
    for key, group in grouped.items():
        eligible = _has_cross_session_support(group["postings"])
        groups.append({
            "group_key": [key[0], key[1]],
            "posting_count": len(group["postings"]),
            "session_count": len({str(p["session_id"]) for p in group["postings"]}),
            "turn_count": len({str(p["turn_id"]) for p in group["postings"]}),
            "eligible": eligible,
            "postings": [dict(p) for p in group["postings"]],
        })
        if eligible:
            batches.append(dict(group))
    return batches, {
        "scope": "conversation_global_subject_type_postings",
        "record_count": len(step1_records),
        "group_count": len(grouped),
        "eligible_group_count": len(batches),
        "groups": groups,
    }


def _as_vector(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    return vector if vector.size and np.all(np.isfinite(vector)) else None


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape:
        return None
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else None


OCCURRENCE_MARKER_RE = re.compile(
    r"\b(?:again|another|new|second|third|once more|this time|today|yesterday|"
    r"tomorrow|last|next|ago|earlier|later)\b", re.IGNORECASE,
)
EVENT_INSTANCE_TYPES = frozenset({
    "ACTIVITY", "EVENT_PARTICIPATION", "PLACE_RELATION", "CREATED_WORK",
    "PURCHASED_ITEM", "HEALTH_STATE", "RECURRING_ISSUE",
})


def _canonical_values(posting: Mapping[str, Any], facet_type: str) -> set[str]:
    return {
        str(e.get("canonical", "")).strip().casefold()
        for e in posting.get("hard_entities", [])
        if isinstance(e, dict) and facet_type in _entity_types(e)
        and str(e.get("canonical", "")).strip()
    }


def clearly_distinct_event_instances(left: Mapping, right: Mapping, facet_type: str) -> bool:
    """Protect explicitly distinct occurrences from embedding-only deletion."""
    if (
        str(left.get("turn_id")) == str(right.get("turn_id"))
        or not str(left.get("session_time", "")).strip()
        or not str(right.get("session_time", "")).strip()
        or str(left.get("session_time")) == str(right.get("session_time"))
    ):
        return False
    left_values, right_values = _canonical_values(left, facet_type), _canonical_values(right, facet_type)
    if left_values and right_values and left_values != right_values:
        return True
    return facet_type in EVENT_INSTANCE_TYPES and bool(
        OCCURRENCE_MARKER_RE.search(str(left.get("original_text", "")))
        or OCCURRENCE_MARKER_RE.search(str(right.get("original_text", "")))
    )


def filter_posting_batches_by_embedding(
    step1_records: Sequence[Any], batches: Sequence[dict], *,
    precomputed_embeddings: Mapping[int, Any] | None = None,
    embedding_model: Any = None,
    retriever_corpus: Sequence[str] | None = None,
    threshold: float = SIMILARITY_SKIP_THRESHOLD,
) -> tuple[list[dict], dict]:
    """Greedily compare each posting with every retained posting in record order."""
    required = sorted({int(p["embedding_ref"]) for b in batches for p in b["postings"]})
    vectors, reused, alignment_failures = {}, [], []
    expected_dimension = None
    dimension_getter = getattr(embedding_model, "get_sentence_embedding_dimension", None)
    if callable(dimension_getter):
        try:
            expected_dimension = int(dimension_getter())
        except (TypeError, ValueError):
            expected_dimension = None
    if expected_dimension is None and precomputed_embeddings is not None:
        dimension_counts = Counter(
            vector.size
            for index in required
            for vector in [_as_vector(precomputed_embeddings.get(index))]
            if vector is not None
        )
        if dimension_counts:
            expected_dimension = dimension_counts.most_common(1)[0][0]
    for index in required:
        if index < 0 or index >= len(step1_records):
            alignment_failures.append(index)
            continue
        if retriever_corpus is not None and (
            index >= len(retriever_corpus)
            or str(retriever_corpus[index]) != str(step1_records[index].embed_text)
        ):
            alignment_failures.append(index)
            continue
        raw = precomputed_embeddings.get(index) if precomputed_embeddings is not None else None
        vector = _as_vector(raw)
        if vector is None:
            continue
        if expected_dimension is None:
            expected_dimension = vector.size
        if vector.size != expected_dimension:
            alignment_failures.append(index)
            continue
        vectors[index], reused = vector, reused + [index]

    missing = [i for i in required if i not in vectors]
    encoded = []
    if missing and embedding_model is not None:
        rows = embedding_model.encode([step1_records[i].embed_text for i in missing])
        rows = list(rows) if rows is not None else []
        if len(rows) == len(missing):
            for index, raw in zip(missing, rows):
                vector = _as_vector(raw)
                if vector is None:
                    continue
                if expected_dimension is None:
                    expected_dimension = vector.size
                if vector.size == expected_dimension:
                    vectors[index], encoded = vector, encoded + [index]

    output, group_diags = [], []
    dropped_total = protected_total = missing_total = 0
    for batch in batches:
        retained, dropped = [], []
        for posting in sorted(batch["postings"], key=lambda p: p["record_index"]):
            vector = vectors.get(int(posting["embedding_ref"]))
            redundant = None
            if vector is None:
                missing_total += 1
            else:
                for kept in retained:
                    similarity = _cosine(vector, vectors.get(int(kept["embedding_ref"]))) \
                        if int(kept["embedding_ref"]) in vectors else None
                    if similarity is None or similarity <= threshold:
                        continue
                    if clearly_distinct_event_instances(kept, posting, batch["facet_type"]):
                        protected_total += 1
                        continue
                    redundant = (kept["turn_id"], similarity)
                    break
            if redundant:
                dropped_total += 1
                dropped.append({
                    "turn_id": posting["turn_id"],
                    "redundant_with_turn_id": redundant[0],
                    "cosine_similarity": redundant[1],
                })
            else:
                retained.append(dict(posting))
        eligible = _has_cross_session_support(retained)
        filtered = dict(batch)
        filtered["postings"] = retained
        if eligible:
            output.append(filtered)
        group_diags.append({
            "group_key": [batch["normalized_subject"], batch["facet_type"]],
            "input_posting_count": len(batch["postings"]),
            "retained_posting_count": len(retained),
            "eligible_after_embedding_redundancy": eligible,
            "dropped": dropped,
            "retained_postings": retained,
        })
    return output, {
        "algorithm": "record_order_greedy_against_any_retained",
        "cosine_drop_rule": f"> {threshold}",
        "similarity_skip_threshold": threshold,
        "required_turn_embedding_count": len(required),
        "reused_turn_embedding_count": len(reused),
        "batch_encoded_turn_count": len(encoded),
        "embedding_alignment_failure_indices": alignment_failures,
        "missing_embedding_posting_count": missing_total,
        "dropped_high_similarity_posting_count": dropped_total,
        "protected_distinct_event_instance_comparison_count": protected_total,
        "eligible_group_count_after_filter": len(output),
        "groups": group_diags,
    }


def serialize_turn_patch(batch: Mapping, postings: Sequence[dict]) -> list[dict]:
    return [{
        "turn_id": str(p["turn_id"]),
        "session_id": p["session_id"],
        "session_time": str(p["session_time"]),
        "speaker": str(p["speaker"]),
        "subject": str(p["subject"]),
        "facet_types": [str(batch["facet_type"])],
        "hard_entities": [dict(e) for e in p.get("hard_entities", [])],
        "keywords": [str(k) for k in p["keywords"]],
        "original_text": str(p["original_text"]),
    } for p in postings]


def chunk_posting_batch(batch: Mapping, *, max_turns: int = MAX_PATCH_TURNS,
                        overlap: int = PATCH_TURN_OVERLAP) -> list[list[dict]]:
    postings = list(batch["postings"])
    if len(postings) <= max_turns:
        return [postings] if _has_cross_session_support(postings) else []
    step = max_turns - overlap
    if step <= 0:
        raise ValueError("patch overlap must be smaller than max turns")
    chunks = []
    for start in range(0, len(postings), step):
        chunk = postings[start:start + max_turns]
        if len(chunk) < 2:
            break
        if _has_cross_session_support(chunk):
            chunks.append(chunk)
        if start + max_turns >= len(postings):
            break
    return chunks


def build_relation_inference_prompt(batch: Mapping, postings: Sequence[dict]) -> tuple[str, dict]:
    turn_patch = serialize_turn_patch(batch, postings)
    facets = batch.get("facet_candidates")
    if not isinstance(facets, list):
        # Preserve the legacy subject/type prompt byte-for-byte when the slot
        # ablation has not supplied its richer candidate description.
        facets = [str(batch["facet_type"])]
    prompt = (RELATION_INFERENCE_PROMPT
              .replace("{{$subject}}", str(batch["subject"]))
              .replace("{{$facets}}", json.dumps(facets, ensure_ascii=False))
              .replace("{{$turn_patch}}", json.dumps(turn_patch, ensure_ascii=False, indent=2)))
    return prompt, {"subject": str(batch["subject"]), "facets": facets,
                    "turn_patch": turn_patch, "prompt": prompt}


def validate_relation_output(item: Any, batch: Mapping, postings: Sequence[dict]) -> dict | None:
    if not isinstance(item, dict):
        return None
    relation_type = str(item.get("relation_type", "")).strip().upper()
    scope, relation_text = str(item.get("scope", "")).strip(), str(item.get("relation_text", "")).strip()
    source_ids = item.get("source_turn_ids")
    if (
        relation_type not in RELATION_TYPES
        or not scope
        or not relation_text
        or not isinstance(source_ids, list)
        or not source_ids
        or any(
            not isinstance(value, str) or not value.strip()
            for value in source_ids
        )
    ):
        return None
    source_ids = [value.strip() for value in source_ids]
    if len(source_ids) < 2 or len(source_ids) != len(set(source_ids)):
        return None
    by_turn = {str(p["turn_id"]): p for p in postings}
    if any(turn_id not in by_turn for turn_id in source_ids):
        return None
    sources = [by_turn[turn_id] for turn_id in source_ids]
    if len({str(p["session_id"]) for p in sources}) < 2:
        return None
    sources.sort(key=lambda p: p["record_index"])
    source_records = [{
        "turn_id": str(p["turn_id"]),
        "session_id": p["session_id"],
        "session_time": str(p["session_time"]),
        "speaker": str(p["speaker"]),
        "record_index": int(p["record_index"]),
    } for p in sources]
    return {
        "valid": True, "memory_type": RELATION_ROUTER_MEMORY_TYPE,
        "relation_type": relation_type, "scope": scope,
        "relation_text": relation_text,
        "subject": str(batch["subject"]), "facet_type": str(batch["facet_type"]),
        "facet_types": [str(batch["facet_type"])],
        "source_turn_ids": [str(p["turn_id"]) for p in sources],
        "source_session_ids": [p["session_id"] for p in sources],
        "source_memory_ids": [f"direct-step1-{p['record_index']}" for p in sources],
        "mention_times": source_records,
        "source_records": source_records,
    }


def infer_relation_batches(batches: Sequence[dict], *, completion_fn: Callable[[str, dict], str],
                           json_parser: Callable[[str], dict]) -> tuple[list[dict], list[dict]]:
    relations, diagnostics, seen = [], [], set()
    for batch_number, batch in enumerate(batches):
        chunks = chunk_posting_batch(batch)
        if not chunks:
            diagnostics.append({
                "group_key": [batch["normalized_subject"], batch["facet_type"]],
                "llm_called": False,
                "reason": "fewer_than_two_cross_session_turns_after_filter",
            })
            continue
        for patch_number, postings in enumerate(chunks):
            prompt, llm_input = build_relation_inference_prompt(batch, postings)
            patch = {"group_key": [batch["normalized_subject"], batch["facet_type"]],
                     "batch_number": batch_number, "patch_number": patch_number,
                     "postings": [dict(p) for p in postings]}
            diagnostic = {**patch, "llm_called": True, "llm_input": llm_input, "valid_relations": []}
            try:
                response = completion_fn(prompt, patch)
                diagnostic["llm_response"] = str(response)
                parsed = json_parser(response)
                items = parsed.get("relations") if isinstance(parsed, dict) else None
                if not isinstance(items, list):
                    raise ValueError("relation response is missing a relations list")
                for item in items:
                    relation = validate_relation_output(item, batch, postings)
                    if relation is None:
                        continue
                    key = (
                        relation["relation_type"].strip().upper(),
                        relation["scope"].strip().casefold(),
                        relation["relation_text"].casefold(),
                        tuple(sorted(relation["source_turn_ids"])),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    relations.append(relation)
                    diagnostic["valid_relations"].append(relation)
            except Exception as exc:
                diagnostic["error"] = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("Cross-session relation inference failed for %r patch %d",
                                 diagnostic["group_key"], patch_number)
            diagnostics.append(diagnostic)
    return relations, diagnostics


def compute_step1_relation_signature(step1_records: Iterable[Any]) -> str:
    rows = []
    for record in step1_records:
        row = {
            "session_id": getattr(record, "session_id", ""),
            "session_time": str(getattr(record, "session_date", "") or ""),
            "turn_id": str(getattr(record, "message_id", "") or ""),
            "speaker": str(getattr(record, "role", "") or ""),
            "original_text": str(getattr(record, "original_text", "") or ""),
            "subject": _record_subject(record),
            "hard_entities": [
                {
                    "canonical": str(entity.get("canonical", "")).strip(),
                    "types": _entity_types(entity),
                }
                for entity in _record_hard_entities(record)
                if isinstance(entity, dict)
            ],
            "keywords": [
                str(keyword) for keyword in getattr(record, "keywords", [])
            ],
        }
        response = getattr(record, "response", None)
        metadata_fields = (
            response.get("metadata_fields")
            if isinstance(response, dict)
            else None
        )
        if isinstance(metadata_fields, list):
            # Ablations alter Step1 embedding text and therefore the relation
            # redundancy filter. Keep legacy signatures byte-identical when
            # all fields are enabled, while isolating every ablated cache.
            row["metadata_fields"] = list(metadata_fields)
            row["embed_text"] = str(getattr(record, "embed_text", ""))
        rows.append(row)
    payload = json.dumps({
        "relation_cache_schema": RELATION_MEMORY_CACHE_SCHEMA,
        "step1_prompt_version": STEP1_CROSS_SESSION_PROMPT_VERSION,
        "step1_schema_version": STEP1_CROSS_SESSION_SCHEMA_VERSION,
        "records": rows,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_relation_timestamp(session_time: Any) -> Any:
    """Parse LoCoMo's natural-language session time, preserving invalid input."""
    text = str(session_time or "").strip()
    if not text:
        return text
    try:
        return date_parser.parse(text, fuzzy=True)
    except (TypeError, ValueError, OverflowError):
        LOGGER.warning("Could not parse relation source session time: %r", text)
        return text


def relation_memory_content(relation: Mapping[str, Any]) -> str:
    mentions = "; ".join(f"{m['turn_id']} at {m['session_time']}" for m in relation["mention_times"])
    return (f"Relation type: {relation['relation_type']}\nSubject: {relation['subject']}\n"
            f"Facet: {relation['facet_type']}\nScope: {relation['scope']}\n"
            f"Relation: {relation['relation_text']}\n"
            f"Source turns: {', '.join(relation['source_turn_ids'])}\nMention times: {mentions}")


def append_relation_memories(memory_system: Any, relations: Sequence[dict]) -> int:
    memories: OrderedDict[str, CrossSessionRelationMemory] = OrderedDict()
    documents = []
    start = sum(getattr(m, "memory_type", "") == RELATION_ROUTER_MEMORY_TYPE
                for m in memory_system.memories.values())
    for relation in relations:
        source_ids = relation.get("source_memory_ids")
        if not isinstance(source_ids, list) or len(source_ids) < 2 or any(
            not isinstance(mid, str) or mid not in memory_system.memories
            or getattr(memory_system.memories[mid], "memory_type", "") == RELATION_ROUTER_MEMORY_TYPE
            for mid in source_ids
        ):
            LOGGER.warning("Dropping relation without resolvable source memories: %s", source_ids)
            continue
        memory_id = f"cross-session-relation-{start + len(memories)}"
        text = str(relation["relation_text"])
        memories[memory_id] = CrossSessionRelationMemory(
            id=memory_id, content=relation_memory_content(relation),
            timestamp=parse_relation_timestamp(
                relation["mention_times"][-1]["session_time"]
            ),
            tags=["cross_session_relation", relation["relation_type"].lower(), relation["facet_type"].lower()],
            relation_type=relation["relation_type"], scope=relation["scope"],
            relation_text=text,
            subject=relation["subject"], facet_type=relation["facet_type"],
            facet_types=list(relation.get("facet_types") or [relation["facet_type"]]),
            source_turn_ids=list(relation["source_turn_ids"]),
            source_session_ids=list(relation["source_session_ids"]),
            source_memory_ids=list(source_ids), mention_times=[dict(m) for m in relation["mention_times"]],
        )
        documents.append(text)
    if not memories:
        return 0
    retriever = memory_system.retriever
    before = (list(getattr(retriever, "corpus", [])), getattr(retriever, "embeddings", None),
              dict(getattr(retriever, "document_ids", {})))
    try:
        retriever.add_documents(documents)
        memory_system.memories.update(memories)
    except Exception:
        retriever.corpus, retriever.embeddings, retriever.document_ids = before
        raise
    return len(memories)
