from __future__ import annotations

import hashlib
import json
from collections import OrderedDict, Counter
from typing import Any, Iterable, Sequence

from cross_session_relation_memory import normalize_subject


PREDICATE_SLOT_SCHEMA_VERSION = "predicate_slots_v1"
PREDICATE_SLOT_PROMPT_VERSION = 1

# These definitions are also presented to the extraction model by
# STEP1_PREDICATE_SLOT_PROMPT.  Keeping the executable schema here makes exact
# ID validation and the relation-prompt facet description share one source of
# truth.
PREDICATE_SLOT_DEFINITIONS = OrderedDict(
    [
        (
            "activity_participation",
            "personal participation in an activity or event, including intended, started, stopped, or resumed participation",
        ),
        ("place_visit", "visiting, travelling to, or temporarily staying at a place"),
        ("residence", "where the subject lives or how that residence changes"),
        ("work", "employment, occupational role, work duties, or work status"),
        ("education_training", "study, courses, training, qualifications, or learning progress"),
        ("volunteering", "providing or intending to provide volunteer service"),
        ("giving_support", "giving money, goods, care, or other explicit support to a recipient"),
        ("social_relationship", "a relationship with another person or group, including its formation, maintenance, or change"),
        ("possession", "owning, acquiring, keeping, or relinquishing an object or animal"),
        ("preference", "an explicit like, dislike, preference, or interest"),
        ("goal_pursuit", "an explicitly expressed or pursued goal, including explicitly connected steps and outcomes"),
        ("health_wellbeing", "physical or mental condition, or actions explicitly presented as managing health or wellbeing"),
        ("emotional_response", "an expressed emotion or emotional reaction to something"),
        ("creation_project", "creating, developing, working on, or completing an identifiable creation or project"),
        ("belief_value", "an explicitly expressed belief, principle, or value"),
        ("self_attribute", "an explicitly stated characteristic, ability, identity, or background"),
    ]
)

PREDICATE_SLOT_IDS = frozenset(PREDICATE_SLOT_DEFINITIONS)
PREDICATE_ASSERTIONS = frozenset(
    {"affirmed", "negated", "uncertain", "intended", "hypothetical"}
)


def resolve_subject_id(value: Any, allowed_subjects: Sequence[str]) -> str:
    """Resolve a model subject to the conversation's canonical subject ID."""
    normalized = normalize_subject(value)
    if not normalized:
        return ""
    by_normalized = {
        normalize_subject(subject): str(subject).strip()
        for subject in allowed_subjects
        if normalize_subject(subject)
    }
    return by_normalized.get(normalized, "")

#事实校验
#
def validate_slot_facts(
    value: Any,
    *,
    allowed_subjects: Sequence[str],
    original_text: str,
) -> tuple[list[dict], dict]:
    """Validate facts independently so one bad fact never discards its peers."""#对于每个事实要独立验证，防止一条记忆的所有事实都被认为是不正确的
    diagnostics = {
        "schema_version": PREDICATE_SLOT_SCHEMA_VERSION,
        "input_is_list": isinstance(value, list),
        "invalid_output_count": 0,
        "input_fact_count": len(value) if isinstance(value, list) else 0,
        "valid_fact_count": 0,
        "indexed_fact_count": 0,
        "empty_slot_fact_count": 0,
        "validation_failure_count": 0,
        "failures": [],
    }
    if not isinstance(value, list):
        diagnostics["invalid_output_count"] = 1
        diagnostics["validation_failure_count"] = 1
        diagnostics["failures"].append(
            {"fact_index": None, "reasons": ["slot_facts_not_list"]}
        )
        return [], diagnostics

    valid_facts: list[dict] = []
    source_text = str(original_text)
    for fact_index, item in enumerate(value):
        if not isinstance(item, dict):
            diagnostics["validation_failure_count"] += 1
            diagnostics["failures"].append(
                {"fact_index": fact_index, "reasons": ["fact_not_object"]}
            )
            continue

        fatal_reasons: list[str] = []
        nonfatal_reasons: list[str] = []
        subject = resolve_subject_id(item.get("subject"), allowed_subjects)
        if not subject:
            fatal_reasons.append("invalid_subject")

        raw_slot_ids = item.get("slot_ids")
        slot_ids: list[str] = []
        if not isinstance(raw_slot_ids, list):
            fatal_reasons.append("slot_ids_not_list")
        else:
            unknown_slot_ids: list[str] = []
            for raw_slot_id in raw_slot_ids:
                slot_id = str(raw_slot_id).strip()
                if slot_id not in PREDICATE_SLOT_IDS:
                    unknown_slot_ids.append(slot_id)
                elif slot_id not in slot_ids:
                    slot_ids.append(slot_id)#slot id允许为空
            if unknown_slot_ids:
                nonfatal_reasons.append("unknown_slot_ids")

        predicate = item.get("predicate")
        if not isinstance(predicate, str) or not predicate.strip():
            fatal_reasons.append("invalid_predicate")
        else:
            predicate = predicate.strip()

        arguments = item.get("arguments")
        if not isinstance(arguments, dict):
            fatal_reasons.append("arguments_not_object")

        assertion = item.get("assertion")
        if not isinstance(assertion, str) or assertion.strip().lower() not in PREDICATE_ASSERTIONS:
            fatal_reasons.append("invalid_assertion")
        else:
            assertion = assertion.strip().lower()

        fact_time = item.get("time")
        if fact_time is not None and not isinstance(fact_time, str):
            fatal_reasons.append("invalid_time")

        evidence = item.get("evidence")
        if not isinstance(evidence, str) or not evidence or evidence not in source_text:
            fatal_reasons.append("evidence_not_in_current_turn")

        reasons = fatal_reasons + nonfatal_reasons
        if reasons:
            failure = {"fact_index": fact_index, "reasons": reasons}
            if "unknown_slot_ids" in nonfatal_reasons:
                failure["unknown_slot_ids"] = [
                    str(slot_id).strip()
                    for slot_id in raw_slot_ids
                    if str(slot_id).strip() not in PREDICATE_SLOT_IDS
                ]
            diagnostics["validation_failure_count"] += 1
            diagnostics["failures"].append(failure)
        if fatal_reasons:
            continue

        fact = {
            "subject": subject,
            "slot_ids": slot_ids,
            "predicate": predicate,
            "arguments": dict(arguments),
            "assertion": assertion,
            "time": fact_time.strip() if isinstance(fact_time, str) else None,
            "evidence": evidence,
        }
        valid_facts.append(fact)
        if slot_ids:
            diagnostics["indexed_fact_count"] += 1
        else:
            diagnostics["empty_slot_fact_count"] += 1

    diagnostics["valid_fact_count"] = len(valid_facts)
    return valid_facts, diagnostics


def _record_slot_facts(record: Any) -> list[dict]:
    response = getattr(record, "response", None)
    facts = response.get("slot_facts") if isinstance(response, dict) else None
    return facts if isinstance(facts, list) else []


def build_predicate_slot_postings(
    step1_records: Sequence[Any],
) -> OrderedDict[tuple[str, str], dict]:
    """Build a conversation-local exact (subject, slot ID) inverted index."""
    grouped: OrderedDict[tuple[str, str], dict] = OrderedDict()
    for record_index, record in enumerate(step1_records):
        turn_id = str(getattr(record, "message_id", "") or "").strip()
        if not turn_id:
            turn_id = f"__record_{record_index}"
        session_id = getattr(record, "session_id", "")
        facts_by_bucket: OrderedDict[tuple[str, str], list[dict]] = OrderedDict()
        #先建立“当前 turn 内”的临时 buckect
        for fact in _record_slot_facts(record):
            if not isinstance(fact, dict):
                continue
            subject = str(fact.get("subject", "")).strip()
            normalized_subject = normalize_subject(subject)
            if not normalized_subject:
                continue
            slot_ids = fact.get("slot_ids")
            if not isinstance(slot_ids, list):
                continue
            for slot_id in slot_ids:#一个 fact 可以进入多个 bucket 因为一个turn可能有多个fact
                slot_id = str(slot_id).strip()
                if slot_id not in PREDICATE_SLOT_IDS:
                    continue
                facts_by_bucket.setdefault(
                    (normalized_subject, slot_id), []
                ).append(dict(fact))

        #把 turn-level bucket 合并进全局 group
        for (normalized_subject, slot_id), slot_facts in facts_by_bucket.items():
            subject = str(slot_facts[0]["subject"]).strip()
            group = grouped.setdefault(
                (normalized_subject, slot_id),
                {
                    "normalized_subject": normalized_subject,
                    "subject": subject,
                    "facet_type": slot_id,
                    "slot_id": slot_id,
                    "slot_definition": PREDICATE_SLOT_DEFINITIONS[slot_id],
                    "facet_candidates": [
                        {
                            "slot_id": slot_id,
                            "definition": PREDICATE_SLOT_DEFINITIONS[slot_id],
                        }
                    ],
                    "postings": [],
                    "_turn_ids": set(),
                },
            )
            if turn_id in group["_turn_ids"]:
                continue
            group["_turn_ids"].add(turn_id)
            group["postings"].append(
                {
                    "session_id": session_id,
                    "session_time": str(
                        getattr(record, "session_date", "") or ""
                    ).strip(),
                    "turn_id": turn_id,
                    "record_index": record_index,
                    "speaker": str(getattr(record, "role", "") or ""),
                    "subject": subject,
                    "hard_entities": [],
                    "keywords": [
                        str(keyword)
                        for keyword in getattr(record, "keywords", [])
                    ],
                    "original_text": str(
                        getattr(record, "original_text", "") or ""
                    ),
                    "embedding_ref": record_index,
                    "slot_facts": slot_facts,
                }
            )
    for group in grouped.values():
        group.pop("_turn_ids", None)
        group["postings"].sort(key=lambda posting: posting["record_index"])
    return grouped


def discover_predicate_slot_batches(
    step1_records: Sequence[Any],
    *,
    conversation_id: Any,
) -> tuple[list[dict], dict]:
    """Return exact slot buckets eligible for existing cross-session windows."""
    grouped = build_predicate_slot_postings(step1_records)
    batches: list[dict] = []
    groups: list[dict] = []
    for (normalized_subject, slot_id), group in grouped.items():
        postings = group["postings"]
        turn_count = len({str(posting["turn_id"]) for posting in postings})
        session_count = len(
            {str(posting["session_id"]) for posting in postings}
        )
        eligible = turn_count >= 2 and session_count >= 2
        diagnostic = {
            "conversation_id": str(conversation_id),
            "group_key": [normalized_subject, slot_id],
            "posting_count": len(postings),
            "session_count": session_count,
            "turn_count": turn_count,
            "eligible": eligible,
            "postings": [dict(posting) for posting in postings],
        }
        groups.append(diagnostic)
        if eligible:
            batch = dict(group)
            batch["conversation_id"] = str(conversation_id)
            batches.append(batch)
    return batches, {
        "scope": "conversation_local_subject_predicate_slot_postings",
        "predicate_slot_schema_version": PREDICATE_SLOT_SCHEMA_VERSION,
        "conversation_id": str(conversation_id),
        "record_count": len(step1_records),
        "group_count": len(grouped),
        "eligible_group_count": len(batches),
        "groups": groups,
    }


def summarize_slot_extraction(step1_records: Iterable[Any]) -> dict:
    counts = Counter()
    for record in step1_records:
        response = getattr(record, "response", None)
        response = response if isinstance(response, dict) else {}
        facts = response.get("slot_facts")
        facts = facts if isinstance(facts, list) else []
        diagnostics = response.get("slot_fact_diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        counts["turns"] += 1
        counts["valid_facts"] += len(facts)
        counts["indexed_facts"] += sum(
            bool(fact.get("slot_ids")) for fact in facts if isinstance(fact, dict)
        )
        counts["empty_slot_facts"] += sum(
            not fact.get("slot_ids") for fact in facts if isinstance(fact, dict)
        )
        counts["validation_failures"] += int(
            diagnostics.get("validation_failure_count", 0) or 0
        )
        if not facts:
            counts["turns_without_valid_facts"] += 1
        if not any(
            isinstance(fact, dict) and fact.get("slot_ids") for fact in facts
        ):
            counts["turns_without_slot_facts"] += 1
        counts["invalid_outputs"] += int(
            diagnostics.get("invalid_output_count", 0) or 0
        )
    return dict(counts)


def compute_predicate_slot_signature(step1_records: Iterable[Any]) -> str:
    rows = []
    for record in step1_records:
        response = getattr(record, "response", None)
        response = response if isinstance(response, dict) else {}
        rows.append(
            {
                "session_id": getattr(record, "session_id", ""),
                "session_time": str(
                    getattr(record, "session_date", "") or ""
                ),
                "turn_id": str(getattr(record, "message_id", "") or ""),
                "speaker": str(getattr(record, "role", "") or ""),
                "original_text": str(
                    getattr(record, "original_text", "") or ""
                ),
                "slot_facts": response.get("slot_facts", []),
            }
        )
    payload = json.dumps(
        {
            "predicate_slot_ablation": True,
            "predicate_slot_schema_version": PREDICATE_SLOT_SCHEMA_VERSION,
            "predicate_slot_prompt_version": PREDICATE_SLOT_PROMPT_VERSION,
            "records": rows,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
