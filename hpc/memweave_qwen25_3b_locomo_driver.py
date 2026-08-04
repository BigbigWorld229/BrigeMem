#!/usr/bin/env python
"""Canonical MemWeaveMine HPC driver for Qwen2.5-3B LoCoMo evaluation."""
import argparse
import contextlib
import json
import logging
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

REPO_ROOT_ON_DISK = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_ON_DISK) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_ON_DISK))

from precompression import (
    add_precompression_args,
    build_turn_compressor_from_args,
)
from prestorage_pipeline import (
    MEMORY_BUILD_MODE_STEP1_PROMPT_ONLY,
    PreStoragePipeline,
    find_direct_step1_memories_raw,
    populate_agent_memory_from_prestorage,
)
from cost_profiler import (
    CostProfiler,
    IncrementalCostWriter,
    STAGE_QUERY_REWRITE,
    STAGE_RETRIEVAL_ANSWER,
)


CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

CATEGORY_K = {
    1: 10,
    2: 10,
    3: 50,
    4: 10,
    5: 10,
}

PAPER_TABLE1_TARGET = {
    "multi_hop": {"f1": 12.57, "bleu1": 9.01},
    "temporal": {"f1": 27.59, "bleu1": 25.07},
    "open_domain": {"f1": 7.12, "bleu1": 7.28},
    "single_hop": {"f1": 17.23, "bleu1": 13.12},
    "adversarial": {"f1": 27.91, "bleu1": 25.15},
}

WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
QA_CHECKPOINT_INTERVAL = 10


def parse_boolean(value: str) -> bool:
    """Parse an explicit true/false CLI value from the HPC shell scripts."""
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MemWeaveMine Qwen2.5-3B LoCoMo evaluation driver."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="MemWeaveMine repository root.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/locomo10.json",
        help="Dataset path relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Served model name exposed by vLLM.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="qwen25",
        choices=["qwen25", "qwen3"],
        help="Qwen model type used to load the official tokenizer chat template.",
    )
    parser.add_argument(
        "--embedding-model-dir",
        type=Path,
        required=True,
        help="Local embedding model directory.",
    )
    parser.add_argument(
        "--llmlingua-pre-step1-compression",
        type=parse_boolean,
        required=True,
        help="Enable LLMLingua compression before Step 1 metadata extraction.",
    )
    parser.add_argument(
        "--memory-build-mode",
        choices=[MEMORY_BUILD_MODE_STEP1_PROMPT_ONLY],
        default=MEMORY_BUILD_MODE_STEP1_PROMPT_ONLY,
        help="Use Step 1 prompt extraction and direct retrieval memory.",
    )

    parser.add_argument(
        "--locomo-category3-use-existing-prompt",
        type=parse_boolean,
        required=True,
        help="Use the existing LoCoMo category-3 answer prompt.",
    )
    parser.add_argument(
        "--prestorage-step1-use-existing-note-prompt",
        type=parse_boolean,
        required=True,
        help="Use the existing pre-storage Step 1 note prompt.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "sglang", "openai", "ollama"],
        help="LLM backend.",
    )
    parser.add_argument(
        "--vllm-host",
        type=str,
        default="http://127.0.0.1",
        help="vLLM host.",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=30000,
        help="vLLM port.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="hpc/cache/memweave_qwen25_3b_locomo_eval",
        help="Cache directory relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hpc/results/memweave_qwen25_3b_locomo_eval.json",
        help="Output json path relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Fraction of dataset to evaluate.",
    )
    parser.add_argument(
        "--temperature-c5",
        type=float,
        default=0.5,
        help="Category 5 temperature used by the upstream repo.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for adversarial-option ordering.",
    )
    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help=(
            "Rebuild every cache used by the selected memory-build mode without "
            "reading artifacts from earlier runs."
        ),
    )
    parser.add_argument(
        "--cost-profiling",
        choices=["on", "off"],
        default="on",
        help="Enable all token, per-stage, per-conversation, and per-QA cost statistics (default: on).",
    )
    add_precompression_args(parser)
    args = parser.parse_args()
    return args


def resolve_path(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def configure_env(repo_root: Path) -> None:
    os.environ.setdefault("HF_HOME", str(repo_root / ".hf_cache"))
    os.environ.setdefault("HF_HUB_CACHE", str(repo_root / ".hf_cache" / "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def patch_sentence_transformer(local_model_dir: Path) -> None:
    import sentence_transformers

    original_cls = sentence_transformers.SentenceTransformer

    def local_sentence_transformer(model_name_or_path, *args, **kwargs):
        requested = str(model_name_or_path)
        if requested == "all-MiniLM-L6-v2":
            model_name_or_path = str(local_model_dir)
        if str(model_name_or_path) == str(local_model_dir):
            kwargs.setdefault("trust_remote_code", True)
            kwargs.setdefault(
                "config_kwargs",
                {"use_memory_efficient_attention": False, "unpad_inputs": False},
            )
        return original_cls(model_name_or_path, *args, **kwargs)

    sentence_transformers.SentenceTransformer = local_sentence_transformer


def import_repo_modules(repo_root: Path, local_model_dir: Path):
    configure_env(repo_root)
    patch_sentence_transformer(local_model_dir)

    sys.path.insert(0, str(repo_root))

    from llm_text_parsers import (
        parse_keywords_response,
        parse_plain_text_answer,
    )
    from load_dataset import load_locomo_dataset
    from memory_layer_robust import RobustAgenticMemorySystem, RobustLLMController

    # Confirm that both HPC jobs resolve the same source files after sys.path patching.
    print(
        "[DEBUG_QWEN_COMPARE] MODULE_PATHS "
        f"driver={Path(__file__).resolve()} "
        f"dataset_loader={Path(sys.modules[load_locomo_dataset.__module__].__file__).resolve()} "
        f"memory_layer={Path(sys.modules[RobustAgenticMemorySystem.__module__].__file__).resolve()}"
    )

    return {
        "parse_keywords_response": parse_keywords_response,
        "parse_plain_text_answer": parse_plain_text_answer,
        "load_locomo_dataset": load_locomo_dataset,
        "RobustAgenticMemorySystem": RobustAgenticMemorySystem,
        "RobustLLMController": RobustLLMController,
    }


def simple_tokenize(text: str) -> list[str]:
    text = str(text)
    return (
        text.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace("!", " ")
        .replace("?", " ")
        .split()
    )


def bleu_tokenize(text: str) -> list[str]:
    return WORD_RE.findall(str(text).lower())


def calc_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens
    if not pred_tokens or not ref_tokens:
        return 0.0
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calc_bleu1(prediction: str, reference: str) -> float:
    pred_tokens = bleu_tokenize(prediction)
    ref_tokens = [bleu_tokenize(reference)]
    if not pred_tokens or not ref_tokens[0]:
        return 0.0
    return sentence_bleu(
        ref_tokens,
        pred_tokens,
        weights=(1.0, 0.0, 0.0, 0.0),
        smoothing_function=SmoothingFunction().method1,
    )


class Table1QwenAgent:
    def __init__(
        self,
        model: str,
        model_type: str,
        backend: str,
        temperature_c5: float,
        host: str,
        port: int,
        seed: int,
        modules: dict,
        cost_profiler: CostProfiler | None = None,
        use_existing_category3_prompt: bool = True,
    ) -> None:
        self._parse_keywords_response = modules["parse_keywords_response"]
        self._parse_plain_text_answer = modules["parse_plain_text_answer"]
        self.cost_profiler = cost_profiler
        self.use_existing_category3_prompt = use_existing_category3_prompt
        # CostProfiler is the sole recorder for the canonical driver.  The
        # controller's legacy standalone tracker remains unset to avoid double
        # accounting.
        self.token_tracker = None
        self.memory_system = modules["RobustAgenticMemorySystem"](
            model_name="all-MiniLM-L6-v2",
            llm_backend=backend,
            llm_model=model,
            llm_model_type=model_type,
            sglang_host=host,
            sglang_port=port,
            usage_tracker=self.token_tracker,
            cost_profiler=cost_profiler,
        )
        self.retriever_llm = modules["RobustLLMController"](
            backend=backend,
            model=model,
            model_type=model_type,
            api_key=None,
            sglang_host=host,
            sglang_port=port,
            usage_tracker=self.token_tracker,
            usage_op_name="retrieval_keyword_query",
            cost_profiler=cost_profiler,
        )
        self.temperature_c5 = temperature_c5
        self.random = random.Random(seed)
        # This records effective controller/model routing without changing controller state.
        print(
            "[DEBUG_QWEN_COMPARE] AGENT_INIT "
            f"model={model} model_type={model_type} backend={backend} host={host} port={port} seed={seed} "
            f"retriever_model=all-MiniLM-L6-v2 category_k={CATEGORY_K} "
            f"memory_build_mode={MEMORY_BUILD_MODE_STEP1_PROMPT_ONLY}"
        )


    def retrieve_memory(self, query: str, category: int) -> tuple[str, int]:
        retrieve_k = CATEGORY_K[int(category)]
        context =find_direct_step1_memories_raw(
                self.memory_system,
                query,
                k=retrieve_k,
            )
        return context, retrieve_k


    def generate_query_llm(self, question: str) -> str:
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""
        self.retriever_llm.usage_op_name = "retrieval_keyword_query"
        response = self.retriever_llm.get_completion(prompt)
        return self._parse_keywords_response(response)

    def answer_question(
        self,
        question: str,
        category: int,
        answer: str | None,
        qa_id: int | str | None = None,
    ) -> tuple[str, str, str, int]:
        question_preview = str(question).replace("\n", " ")[:100]
        print(
            "[DEBUG_QWEN_COMPARE] QA_START "
            f"category={category} question_chars={len(str(question))} "
            f"question_tokens_est={len(simple_tokenize(question))} question_preview={question_preview!r}"
        )
        qa_scope = (
            self.cost_profiler.scope(
                qa_id=qa_id,
                item_type="qa",
                category=category,
            )
            if self.cost_profiler is not None and self.cost_profiler.enabled
            else contextlib.nullcontext()
        )
        with qa_scope:
            query_scope = (
                self.cost_profiler.scope(stage=STAGE_QUERY_REWRITE)
                if self.cost_profiler is not None and self.cost_profiler.enabled
                else contextlib.nullcontext()
            )
            with query_scope:
                # Match SimpleMem's LoCoMo evaluator: retrieval timing starts
                # immediately before the first retrieval operation, excluding
                # the per-QA diagnostics and profiling-scope preparation.
                retrieval_started = time.time()
                keywords = self.generate_query_llm(question)

            retrieval_scope = (
                self.cost_profiler.scope(stage=STAGE_RETRIEVAL_ANSWER)
                if self.cost_profiler is not None and self.cost_profiler.enabled
                else contextlib.nullcontext()
            )
            retrieval_timer = (
                self.cost_profiler.stage_timer(
                    STAGE_RETRIEVAL_ANSWER,
                    "vector_retrieval_and_link_expansion",
                    contributes_to_stage_latency=True,
                    requested_category=category,
                )
                if self.cost_profiler is not None and self.cost_profiler.enabled
                else contextlib.nullcontext()
            )
            with retrieval_scope:
                with retrieval_timer:
                    raw_context, retrieve_k = self.retrieve_memory(keywords, category)
                    retrieval_time = time.time() - retrieval_started
        context = raw_context
        if self.cost_profiler is not None and self.cost_profiler.enabled:
            self.cost_profiler.record_timing(
                stage=STAGE_RETRIEVAL_ANSWER,
                operation="qa_retrieval_total",
                latency_seconds=retrieval_time,
                qa_id=qa_id,
                item_type="qa",
                category=category,
            )
        # As in SimpleMem, answer timing includes answer-prompt construction,
        # generation, and output parsing, but excludes subsequent evaluation.
        answer_started = time.time()

        if category == 5:
            options = []
            if self.random.random() < 0.5:
                options.append("Not mentioned in the conversation")
                options.append(answer)
            else:
                options.append(answer)
                options.append("Not mentioned in the conversation")
            user_prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {options[0]} or {options[1]}  Short answer:"""
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
            temperature = 0.7
        elif category == 3 and not self.use_existing_category3_prompt:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer by combining information from the context with relevant external facts or general knowledge.

Question: {question} Short answer:"""
            temperature = 0.7
        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7

        system_message = self.memory_system.llm_controller.llm.SYSTEM_MESSAGE
        system_tokens_est = len(simple_tokenize(system_message))
        prompt_tokens_est = len(simple_tokenize(user_prompt))
        context_tokens_est = len(simple_tokenize(raw_context))
        # Estimate is intentionally lightweight; server-side exact usage is logged by the controller.
        print(
            "[DEBUG_QWEN_COMPARE] QA_PROMPT "
            f"category={category} retrieve_k={retrieve_k} keywords_chars={len(str(keywords))} "
            f"keywords_tokens_est={len(simple_tokenize(keywords))} context_chars={len(raw_context)} "
            f"context_tokens_est={context_tokens_est} system_chars={len(system_message)} "
            f"system_tokens_est={system_tokens_est} final_prompt_chars={len(user_prompt)} "
            f"final_prompt_tokens_est={prompt_tokens_est} combined_tokens_est={system_tokens_est + prompt_tokens_est} "
            f"max_model_len_expected=8192 over_8192_est={system_tokens_est + prompt_tokens_est > 8192}"
        )

        try:
            self.memory_system.llm_controller.usage_op_name = "qa_answer_generation"
            answer_scope = (
                self.cost_profiler.scope(
                    stage=STAGE_RETRIEVAL_ANSWER,
                    qa_id=qa_id,
                    item_type="qa",
                    category=category,
                )
                if self.cost_profiler is not None and self.cost_profiler.enabled
                else contextlib.nullcontext()
            )
            with answer_scope:
                response = self.memory_system.llm_controller.get_completion(
                    user_prompt,
                    temperature=temperature,
                )
        except Exception as exc:
            # The original behavior remains an empty response; only expose where/why it happened.
            print(
                "[DEBUG_QWEN_COMPARE] QA_EXCEPTION "
                f"category={category} stage=qa_answer_generation exception_type={type(exc).__name__} "
                f"context_tokens_est={context_tokens_est} prompt_tokens_est={prompt_tokens_est} "
                f"message={str(exc)[:100]!r}"
            )
            response = ""
        finally:
            self.memory_system.llm_controller.usage_op_name = "memory_metadata"

        prediction = self._parse_plain_text_answer(response)
        answer_time = time.time() - answer_started
        if self.cost_profiler is not None and self.cost_profiler.enabled:
            self.cost_profiler.record_timing(
                stage=STAGE_RETRIEVAL_ANSWER,
                operation="qa_answer_total",
                latency_seconds=answer_time,
                qa_id=qa_id,
                item_type="qa",
                category=category,
            )
        print(
            "[DEBUG_QWEN_COMPARE] QA_END "
            f"category={category} response_chars={len(str(response))} "
            f"prediction_chars={len(str(prediction))} empty_prediction={not bool(str(prediction).strip())}"
        )
        return prediction, user_prompt, raw_context, retrieve_k

    def get_token_statistics(self) -> dict:
        if self.cost_profiler is None or not self.cost_profiler.enabled:
            return {}
        return self.cost_profiler.token_statistics()


def build_or_load_memories(
    agent: Table1QwenAgent,
    sample,
    cache_dir: Path,
    sample_idx: int,
    turn_compressor,
    prestorage_pipeline: PreStoragePipeline,
    force_rebuild_cache: bool = False,
) -> None:
    prestorage_pipeline.set_usage_tracker(agent.memory_system.usage_tracker)
    populate_agent_memory_from_prestorage(
        agent=agent,
        sample=sample,
        sample_idx=sample_idx,
        cache_dir=cache_dir,
        compressor=turn_compressor,
        pipeline=prestorage_pipeline,
        force_rebuild_cache=force_rebuild_cache,
    )


def summarize(records: list[dict]) -> dict:
    per_category = defaultdict(list)
    for record in records:
        per_category[int(record["category"])].append(record)

    summary = {}
    weighted_f1 = []
    weighted_bleu1 = []
    macro_f1 = []
    macro_bleu1 = []

    for category in sorted(per_category):
        items = per_category[category]
        f1_scores = [item["metrics"]["f1"] for item in items]
        bleu_scores = [item["metrics"]["bleu1"] for item in items]
        token_lengths = [item["prediction_token_length"] for item in items]
        cat_name = CATEGORY_NAMES[category]
        cat_summary = {
            "name": cat_name,
            "count": len(items),
            "retrieve_k": CATEGORY_K[category],
            "f1_mean": statistics.mean(f1_scores),
            "bleu1_mean": statistics.mean(bleu_scores),
            "prediction_token_length_mean": statistics.mean(token_lengths),
            "paper_target": PAPER_TABLE1_TARGET[cat_name],
        }
        summary[f"category_{category}"] = cat_summary
        weighted_f1.extend(f1_scores)
        weighted_bleu1.extend(bleu_scores)
        macro_f1.append(cat_summary["f1_mean"])
        macro_bleu1.append(cat_summary["bleu1_mean"])

    summary["overall_weighted"] = {
        "f1_mean": statistics.mean(weighted_f1) if weighted_f1 else 0.0,
        "bleu1_mean": statistics.mean(weighted_bleu1) if weighted_bleu1 else 0.0,
    }
    summary["overall_macro"] = {
        "f1_mean": statistics.mean(macro_f1) if macro_f1 else 0.0,
        "bleu1_mean": statistics.mean(macro_bleu1) if macro_bleu1 else 0.0,
    }
    return summary


def build_result_payload(
    *,
    args: argparse.Namespace,
    dataset_path: Path,
    turn_compressor,
    results: list[dict],
    total_questions: int,
    is_complete: bool,
) -> dict:
    """Build a readable result payload for both checkpoints and final output."""
    precompression_config = dict(turn_compressor.config_dict())
    # Compression token counters are cost statistics and are written in the
    # unified cost file; retain only non-cost experiment configuration here.
    precompression_config.pop("stats", None)
    return {
        "model": args.model,
        "memory_build_mode": args.memory_build_mode,
        "backend": args.backend,
        "dataset": str(dataset_path),
        "ratio": args.ratio,
        "seed": args.seed,
        "category_k": CATEGORY_K,
        "precompression": precompression_config,
        "ablation": {"components": dict(args.ablation_components)},
        "paper_table1_target": PAPER_TABLE1_TARGET,
        "total_questions": total_questions,
        "summary": summarize(results),
        "individual_results": results,
        "is_complete": is_complete,
    }


def aggregate_qa_cost_batch(qa_costs: list[dict]) -> dict:
    """Aggregate one durable QA batch without changing QA execution behavior."""
    retrieval_llm_fields = (
        "llm_calls",
        "successful_calls",
        "failed_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "llm_latency_seconds",
    )
    retrieval_llm = {
        field: sum(item["retrieval_llm"].get(field, 0) for item in qa_costs)
        for field in retrieval_llm_fields
    }
    answer_llm = {
        field: sum(item["answer_llm"].get(field, 0) for item in qa_costs)
        for field in retrieval_llm_fields
    }
    return {
        "qa_count": len(qa_costs),
        "qa_keys": [
            {
                "conversation_id": item["conversation_id"],
                "qa_id": item["qa_id"],
                "global_qa_index": item["global_qa_index"],
            }
            for item in qa_costs
        ],
        "retrieval_time_seconds": sum(
            float(item["retrieval_time_seconds"]) for item in qa_costs
        ),
        "answer_time_seconds": sum(
            float(item["answer_time_seconds"]) for item in qa_costs
        ),
        "total_time_seconds": sum(
            float(item["total_time_seconds"]) for item in qa_costs
        ),
        "vector_retrieval_time_seconds": sum(
            float(item["vector_retrieval_time_seconds"]) for item in qa_costs
        ),
        "retrieval_llm": retrieval_llm,
        "answer_llm": answer_llm,
    }


def atomic_write_json(output_path: Path, payload: dict) -> None:
    """Atomically replace the result JSON so an older checkpoint stays valid."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, output_path)
        try:
            directory_fd = os.open(output_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some distributed filesystems do not support fsync on directories.
            pass
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("memweave_qwen25_3b_driver")

    repo_root = args.repo_root.resolve()
    dataset_path = resolve_path(repo_root, args.dataset)
    use_existing_category3_prompt = args.locomo_category3_use_existing_prompt
    use_existing_step1_note_prompt = args.prestorage_step1_use_existing_note_prompt
    args.ablation_components = {
        "llmlingua_pre_step1_compression": args.llmlingua_pre_step1_compression,
        "locomo_category3_use_existing_prompt": use_existing_category3_prompt,
        "prestorage_step1_use_existing_note_prompt": use_existing_step1_note_prompt,
    }
    # The explicit HPC ablation switch is the single source of truth for the
    # existing precompression pipeline's enable_precompress argument.
    args.enable_precompress = args.llmlingua_pre_step1_compression
    cache_dir = resolve_path(repo_root, args.cache_dir)
    output_path = resolve_path(repo_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    turn_compressor = build_turn_compressor_from_args(args)
    cost_enabled = args.cost_profiling == "on"
    cost_profiler = CostProfiler(
        enabled=cost_enabled,
        model_name=args.model,
    )
    # The shell places --output under its Job-ID-specific result directory.
    # Deriving the path from that parent keeps exactly one cost file per job.
    cost_output_path = output_path.parent / "cost_statistics.jsonl"
    cost_writer = IncrementalCostWriter(cost_output_path, enabled=cost_enabled)
    cost_writer.start(
        model=args.model,
        backend=args.backend,
        dataset=str(dataset_path),
        result_file=str(output_path),
        cost_file=str(cost_output_path),
    )
    if cost_enabled:
        logger.info("Unified cost statistics enabled: %s", cost_output_path)
    else:
        logger.info("Unified cost statistics disabled")
    for component_name, component_enabled in args.ablation_components.items():
        logger.info("Ablation component: %s=%s", component_name, component_enabled)

    # Emit resolved CLI/environment values so the two shell scripts can be compared directly.
    logger.info(
        "[DEBUG_QWEN_COMPARE] DRIVER_ARGS model=%s model_type=%s backend=%s dataset=%s ratio=%s "
        "memory_build_mode=%s vllm_host=%s vllm_port=%s embedding_model=%s cache_dir=%s output=%s "
        "force_rebuild_cache=%s cuda_visible_devices=%r max_model_len_expected=8192",
        args.model,
        args.model_type,
        args.backend,
        dataset_path,
        args.ratio,
        args.memory_build_mode,
        args.vllm_host,
        args.vllm_port,
        args.embedding_model_dir.resolve(),
        cache_dir,
        output_path,
        args.force_rebuild_cache,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )

    modules = import_repo_modules(repo_root, args.embedding_model_dir.resolve())
    load_locomo_dataset = modules["load_locomo_dataset"]
    cache_tag = turn_compressor.cache_tag()
    if not use_existing_step1_note_prompt:
        cache_tag += "_step1_note_prompt_ablation"
    cache_dir = cache_dir / cache_tag
    prestorage_pipeline = PreStoragePipeline(
        backend=args.backend,
        model=args.model,
        model_type=args.model_type,
        cache_dir=cache_dir,
        sglang_host=args.vllm_host,
        sglang_port=args.vllm_port,
        cost_profiler=cost_profiler,
        use_existing_step1_note_prompt=use_existing_step1_note_prompt,
    )

    logger.info("Loading dataset: %s", dataset_path)
    logger.info("Pre-compression config: %s", turn_compressor.config_dict())
    samples = load_locomo_dataset(dataset_path)
    logger.info(
        "[DEBUG_QWEN_COMPARE] DATASET_LOADED path=%s sample_count_before_ratio=%d",
        dataset_path,
        len(samples),
    )
    if args.ratio < 1.0:
        keep = max(1, int(len(samples) * args.ratio))
        samples = samples[:keep]
        logger.info("Using ratio %.4f -> %d samples", args.ratio, keep)

    results = []
    total_questions = 0
    pending_qa_costs: list[dict] = []
    completed_conversations = 0

    for sample_idx, sample in enumerate(samples):
        cost_profiler.begin_conversation(sample.sample_id)
        logger.info("Preparing sample %d/%d", sample_idx + 1, len(samples))
        session_count = len(sample.conversation.sessions)
        turn_count = sum(len(session.turns) for session in sample.conversation.sessions.values())
        logger.info(
            "[DEBUG_QWEN_COMPARE] SAMPLE_START sample_id=%d ordinal=%d/%d sessions=%d turns=%d qas=%d",
            sample_idx,
            sample_idx + 1,
            len(samples),
            session_count,
            turn_count,
            len(sample.qa),
        )
        agent = Table1QwenAgent(
            model=args.model,
            model_type=args.model_type,
            backend=args.backend,
            temperature_c5=args.temperature_c5,
            host=args.vllm_host,
            port=args.vllm_port,
            seed=args.seed + sample_idx,
            modules=modules,
            cost_profiler=cost_profiler,
            use_existing_category3_prompt=use_existing_category3_prompt,
        )
        logger.info("[DEBUG_QWEN_COMPARE] STAGE_START sample_id=%d stage=memory_build", sample_idx)
        # Match SimpleMem's construction timer: exclude sample/agent setup and
        # start immediately before the complete memory build/load call.
        construction_started = time.time()
        build_or_load_memories(
            agent,
            sample,
            cache_dir,
            sample_idx,
            turn_compressor,
            prestorage_pipeline,
            force_rebuild_cache=args.force_rebuild_cache,
        )
        construction_time = time.time() - construction_started
        logger.info(
            "[DEBUG_QWEN_COMPARE] STAGE_END sample_id=%d stage=memory_build memories=%d retriever_corpus=%d",
            sample_idx,
            len(agent.memory_system.memories),
            len(agent.memory_system.retriever.corpus),
        )
        if cost_enabled:
            memory_build_cost = cost_profiler.conversation_memory_build_summary(
                sample.sample_id,
                construction_time_seconds=construction_time,
            )
            cost_writer.append(
                "conversation_memory_build_completed",
                sample_index=sample_idx,
                conversation_id=str(sample.sample_id),
                memory_count=len(agent.memory_system.memories),
                retriever_corpus_count=len(agent.memory_system.retriever.corpus),
                cost=memory_build_cost,
            )
        completed_conversations += 1
        logger.info("[DEBUG_QWEN_COMPARE] STAGE_START sample_id=%d stage=qa", sample_idx)
        for qa_idx, qa in enumerate(sample.qa):
            if int(qa.category) not in CATEGORY_NAMES:
                continue

            total_questions += 1
            logger.info(
                "[DEBUG_QWEN_COMPARE] QA_ID sample_id=%d qa_index=not_tracked global_qa_index=%d category=%s",
                sample_idx,
                total_questions,
                qa.category,
            )
            prediction, prompt, raw_context, retrieve_k = agent.answer_question(
                qa.question,
                int(qa.category),
                qa.final_answer,
                qa_id=qa_idx,
            )
            if cost_enabled:
                qa_cost = cost_profiler.qa_cost_summary(sample.sample_id, qa_idx)
                qa_cost.update(
                    {
                        "sample_index": sample_idx,
                        "global_qa_index": total_questions,
                        "category": int(qa.category),
                    }
                )
                # Persist immediately after inference returns, before evaluation
                # or result checkpoint work can fail.
                cost_writer.append("qa_completed", cost=qa_cost)
                pending_qa_costs.append(qa_cost)

            reference = qa.final_answer or ""
            metrics = {
                "f1": calc_f1(prediction, reference),
                "bleu1": calc_bleu1(prediction, reference),
            }
            result = {
                "sample_id": sample_idx,
                "question": qa.question,
                "reference": reference,
                "prediction": prediction,
                "category": int(qa.category),
                "category_name": CATEGORY_NAMES[int(qa.category)],
                "retrieve_k": retrieve_k,
                "metrics": metrics,
                "prediction_token_length": len(simple_tokenize(prediction)),
                "prompt": prompt,
                "raw_context": raw_context,
            }
            results.append(result)

            if total_questions % QA_CHECKPOINT_INTERVAL == 0:
                if cost_enabled:
                    cost_writer.append(
                        "qa_batch_checkpoint",
                        checkpoint_reason="interval",
                        global_qa_count=total_questions,
                        batch=aggregate_qa_cost_batch(pending_qa_costs),
                    )
                    pending_qa_costs.clear()
                checkpoint_payload = build_result_payload(
                    args=args,
                    dataset_path=dataset_path,
                    turn_compressor=turn_compressor,
                    results=results,
                    total_questions=total_questions,
                    is_complete=False,
                )
                atomic_write_json(output_path, checkpoint_payload)
                logger.info(
                    "Saved QA checkpoint with %d questions to %s",
                    total_questions,
                    output_path,
                )
                logger.info("Processed %d questions", total_questions)

        cost_profiler.end_conversation(sample.sample_id)
        if cost_enabled:
            logger.info(
                "[DEBUG_QWEN_COMPARE] SAMPLE_END sample_id=%d memories=%d retriever_corpus=%d "
                "processed_global_qas=%d token_stats=%s",
                sample_idx,
                len(agent.memory_system.memories),
                len(agent.memory_system.retriever.corpus),
                total_questions,
                agent.get_token_statistics()["lightmem_style"],
            )
        else:
            logger.info(
                "[DEBUG_QWEN_COMPARE] SAMPLE_END sample_id=%d memories=%d retriever_corpus=%d "
                "processed_global_qas=%d cost_statistics=disabled",
                sample_idx,
                len(agent.memory_system.memories),
                len(agent.memory_system.retriever.corpus),
                total_questions,
            )

    if cost_enabled and pending_qa_costs:
        cost_writer.append(
            "qa_batch_checkpoint",
            checkpoint_reason="final_remainder",
            global_qa_count=total_questions,
            batch=aggregate_qa_cost_batch(pending_qa_costs),
        )
        pending_qa_costs.clear()

    payload = build_result_payload(
        args=args,
        dataset_path=dataset_path,
        turn_compressor=turn_compressor,
        results=results,
        total_questions=total_questions,
        is_complete=True,
    )
    summary = payload["summary"]

    token_statistics = None
    if cost_enabled:
        token_statistics = cost_profiler.token_statistics()
        cost_writer.append(
            "final_summary",
            is_complete=True,
            completed_conversations=completed_conversations,
            completed_qas=total_questions,
            precompression=turn_compressor.config_dict(),
            token_statistics=token_statistics,
            stage_cost_statistics=cost_profiler.snapshot(),
        )

    atomic_write_json(output_path, payload)

    logger.info("Saved results to %s", output_path)
    for key in sorted(k for k in summary if k.startswith("category_")):
        item = summary[key]
        logger.info(
            "%s: F1=%.4f BLEU1=%.4f k=%d",
            item["name"],
            item["f1_mean"],
            item["bleu1_mean"],
            item["retrieve_k"],
        )
    if cost_enabled:
        logger.info(
            "Token stats: summary=%.2fk memory_build=%.2fk qa=%.2fk total_calls=%d",
            token_statistics["lightmem_style"]["summary_tokens_k"],
            token_statistics["lightmem_style"]["memory_build_total_tokens_k"],
            token_statistics["lightmem_style"]["qa_total_tokens_k"],
            token_statistics["lightmem_style"]["calls"],
        )
        logger.info(cost_profiler.terminal_summary())
        logger.info("Saved unified cost statistics to %s", cost_output_path)


if __name__ == "__main__":
    main()
