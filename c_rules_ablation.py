"""Opt-in context / answer policies from job 139515 C.RULES.

Reference: experiments/priority3_full/outputs/p3_full_rules_t07_s42_43_44_v1/
launch_code/engine/{priority01/core.py,priority23/protocols.py,runtime.py}.
No experiment-directory imports, cache mutations, or generation/scorer changes.
"""
from __future__ import annotations

import re

CONTEXT_BUDGET = 4096
CONTEXT_MARKER = "__P23_FROZEN_CONTEXT_b92392__"
POLICY_VERSION = "139515_c_rules_v1"


def question_features(question):
    """Only question text is accepted: no category, answer, or gold evidence."""
    text = question.casefold()
    patterns = {
        "time": r"\b(when|what date|which year|what year|how long|how many (?:days|weeks|months|years)|before|after|earlier|later)\b",
        "count": r"\b(how many|how often|number of|times did)\b",
        "enumeration": r"\b(list|all|both)\b|\b(?:what|which)\b.*\b(activities|places|countries|books|hobbies|sports|instruments|events|jobs|pets|fields)\b",
        "hypothetical": r"\b(would|likely|might|could|suitable|recommend)\b",
    }
    return [key for key, expression in patterns.items() if re.search(expression, text)]


RULES = {
    "time": "For time questions, anchor relative times to the source turn's session date, not today's date. Preserve supported precision: a year stays a year; a month stays a month; give an exact day only when supported. Distinguish mention time from event time; do not invent dates to satisfy formatting.",
    "count": "For counts/frequency, count distinct supported event instances, not repeated mentions. Separate actual events from plans. State uncertainty rather than inventing an exact count from incomplete evidence.",
    "enumeration": "For explicit list or plural-entity questions, include all distinct supported requested items, deduplicate aliases, and exclude irrelevant extras. Concision must not omit supported items required by the question.",
    "hypothetical": "For suitability or hypothetical questions, synthesize a cautious inference from explicitly stated preferences, experience or constraints. Mark it as an inference, not a new biographical fact; do not import unsupported personal facts or external claims.",
}


def apply_answer_rules(prompt, context, question):
    """The frozen answer_prompt RULES branch, never FORMAT.

Protect the context while editing the base template, including when dialogue
itself contains the date-format instruction. Empty contexts are supported.
"""
    if context:
        if prompt.count(context) != 1:
            raise ValueError("QA prompt must contain exactly one context occurrence")
        prompt = prompt.replace(context, CONTEXT_MARKER, 1)
        if prompt.count(CONTEXT_MARKER) != 1:
            raise ValueError("Context marker collision")
    features = question_features(question)
    if "time" in features:
        prompt = prompt.replace("All dates in the response must be formatted as 'DD Month YYYY' but you can output more or less details if needed",
                                "Preserve the date precision actually supported by the evidence")
    prompt += ("\nAdditional evidence rules: Attribute each fact to its actual subject, not automatically to the speaker. "
               "Keep ownership, negation and intention separate from confirmed action.\n")
    prompt += "\n".join(RULES[f] for f in features) + "\n"
    return (prompt.replace(CONTEXT_MARKER, context, 1) if context else prompt), features


def raw_block(record):
    from dateutil import parser
    date = parser.parse(record["session_time"], fuzzy=True)
    label = f"{date.day} {date.strftime('%B %Y')}"
    return f"[source {record['turn_id']}] [{label}] {record['speaker']}: {record['original_text']}"


def build_c_context(
    direct,
    relations,
    direct_ranking,
    relation_ranking,
    *,
    direct_k,
    relation_k,
    token_count,
):
    """C text with the original, cumulative D-side atomic budget check."""
    by_id = {d["id"]: d for d in direct}
    rel_by_id = {r["id"]: r for r in relations}
    direct_ids = [h["id"] for h in direct_ranking[:direct_k]]
    c = [raw_block(by_id[i]) for i in direct_ids]#真正给模型的context内容
    d = list(c)

    def join(blocks):
        return "\n".join(blocks) + ("\n" if blocks else "")

    def count(blocks):
        return token_count(join(blocks))

    direct_tokens = count(c)
    if direct_tokens > CONTEXT_BUDGET:
        raise ValueError(
            f"C.RULES reserved direct top{direct_k} needs {direct_tokens} tokens "
            f"> context budget {CONTEXT_BUDGET}; refusing to drop or truncate direct evidence"
        )
    emitted, ordered_ids = set(direct_ids), list(direct_ids)
    selected, skipped = [], []
    for hit in relation_ranking[:relation_k]:
        relation = rel_by_id[hit["id"]]
        source_ids = list(dict.fromkeys(relation["source_memory_ids"]))
        if len(source_ids) < 2 or any(i not in by_id for i in source_ids):
            raise ValueError("Cannot expose an incomplete relation source group")
        new_ids = [i for i in source_ids if i not in emitted]
        additions = [raw_block(by_id[i]) for i in new_ids]
        relation_block = f"[Relation; sources {', '.join(by_id[i]['turn_id'] for i in source_ids)}] {relation['relation_text']}"
        if count(d + additions + [relation_block]) > CONTEXT_BUDGET:
            skipped.append({**hit, "reason": "atomic_group_exceeds_shared_budget"})
            continue
        c.extend(additions)
        d.extend(additions + [relation_block])
        emitted.update(source_ids)
        ordered_ids.extend(new_ids)
        selected.append({**hit, "source_memory_ids": source_ids,
                         "source_turn_ids": [by_id[i]["turn_id"] for i in source_ids]})
    text = join(c)
    return text, {
        "policy": POLICY_VERSION, "direct_k": direct_k, "relation_k": relation_k,
        "shared_context_token_limit": CONTEXT_BUDGET,
        "direct_tokens": direct_tokens, "context_tokens": token_count(text),
        "d_budget_tokens": count(d), "reserved_direct_ids": direct_ids,
        "source_memory_ids": ordered_ids,
        "source_turn_ids": [by_id[i]["turn_id"] for i in ordered_ids],
        "direct_ranking": direct_ranking[:direct_k],
        "relation_ranking": relation_ranking[:relation_k],
        "selected_relations": selected, "skipped_relations": skipped,
    }


def rank_channel(vector, vectors, ids):
    """Same cosine ranking and tie order as frozen priority23.runtime.rank."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    if not ids:
        return []
    if len(vectors) != len(ids) or not np.isfinite(vectors).all() or not np.isfinite(vector).all():
        raise ValueError("Invalid/misaligned embeddings")
    scores = cosine_similarity(vector, vectors)[0]
    return [{"id": ids[int(i)], "score": float(scores[i]), "corpus_index": int(i)}
            for i in np.argsort(scores)[::-1]]


def retrieve_c_context(memory_system, query, tokenizer, *, direct_k, relation_k):
    """Rank existing index vectors by channel; encode only the query once."""
    import numpy as np
    memories = list(memory_system.memories.items())
    retriever = memory_system.retriever
    if len(retriever.corpus) != len(memories):
        raise ValueError("C.RULES memory/corpus alignment mismatch")
    direct, relations, direct_indices, relation_indices = [], [], [], []
    for index, (memory_id, memory) in enumerate(memories):
        if memory.id != memory_id:
            raise ValueError("C.RULES memory ID alignment mismatch")
        if getattr(memory, "memory_type", "") == "cross_session_relation_router":
            expected = memory.relation_text
            relations.append({"id": memory_id, "relation_text": memory.relation_text,
                              "source_memory_ids": list(memory.source_memory_ids)})
            relation_indices.append(index)
        else:
            fields = ("turn_id", "session_time", "speaker", "original_text", "retrieval_text")
            if any(not hasattr(memory, name) for name in fields):
                raise ValueError("C.RULES needs original source metadata; enable cross-session relation memory and use a compatible cache")
            expected = memory.retrieval_text
            direct.append({"id": memory_id, **{name: getattr(memory, name) for name in fields}})
            direct_indices.append(index)
        if retriever.corpus[index] != expected:
            raise ValueError("C.RULES memory/corpus order or index text mismatch")
    if memories:
        vectors = np.asarray(retriever.embeddings)
        if vectors.ndim != 2 or len(vectors) != len(memories):
            raise ValueError("C.RULES memory/embedding alignment mismatch")
        vector = retriever.model.encode([query])
        direct_ranking = rank_channel(vector, vectors[direct_indices], [r["id"] for r in direct])
        relation_ranking = rank_channel(vector, vectors[relation_indices], [r["id"] for r in relations])
    else:
        direct_ranking, relation_ranking = [], []
    return build_c_context(
        direct,
        relations,
        direct_ranking,
        relation_ranking,
        direct_k=direct_k,
        relation_k=relation_k,
        token_count=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
    )
