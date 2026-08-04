# MemWeaveMine

MemWeaveMine is a layered memory construction and evaluation pipeline for long-conversation question answering.

This repository is not a generic agent-memory SDK. It is a research-oriented evaluation workspace built around three combined ideas:

- an A-Mem-style evolving memory layer
- a PREMem-style Step 1 single-message compression stage
- a local PreStorage pipeline that turns turn-level memories into richer retrieval notes and cross-session reasoning memories

The result is a reproducible QA stack that can run against OpenAI-compatible APIs, vLLM, SGLang, or Ollama, with extra HPC entrypoints for Qwen2.5-3B experiments.

## What This Project Does

Given a LoCoMo sample, the pipeline:

1. loads conversation sessions and QA targets from `data/locomo10.json`
2. converts image turns into caption-augmented text
3. optionally compresses each dialogue turn with LLMLingua
4. extracts compact Step 1 metadata for each turn
5. clusters Step 1 memories across sessions and synthesizes Step 2 reasoning memories
6. stores all resulting notes in an evolving long-term memory system
7. retrieves memory snippets to answer LoCoMo questions and scores the predictions

## Repository Workflow

### 1. Dataset loading

- File: [load_dataset.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/load_dataset.py:8)
- Parses LoCoMo QA pairs, sessions, observations, event summaries, and captioned image turns.

### 2. Optional Step 1 pre-compression

- File: [precompression.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/precompression.py:89)
- Wraps LLMLingua as a conservative PREMem-style single-turn compressor.
- Produces `compress_text` and `memory_content` records per turn.

### 3. PreStorage Step 1 note construction

- File: [prestorage_pipeline.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/prestorage_pipeline.py:526)
- Uses an LLM to extract `keywords`, `summary`, and `tags` from each compressed turn.
- Saves note caches and embedding artifacts for later reuse.

### 4. PreStorage Step 2 reasoning memory synthesis

- File: [prestorage_pipeline.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/prestorage_pipeline.py:771)
- Clusters Step 1 embeddings by session, links similar clusters across time, and asks the LLM to produce cross-fragment insights.

### 5. Long-term memory ingestion and evolution

- Files:
  - [memory_layer.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/memory_layer.py:666)
  - [memory_layer_robust.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/memory_layer_robust.py:432)
- Builds searchable memory notes, retrieves semantically related notes, and optionally updates links, tags, and neighbor context.

### 6. QA answering and evaluation

- Files:
  - [test_advanced_robust.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/test_advanced_robust.py:179)
  - [utils.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/utils.py:109)
- Runs LoCoMo QA, generates predictions from retrieved memory context, and computes aggregate metrics.

### 7. Qwen2.5-3B HPC evaluation

- Files:
  - [hpc/memweave_qwen25_3b_locomo_driver.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/hpc/memweave_qwen25_3b_locomo_driver.py:55)
  - [hpc/memweave_qwen25_3b_locomo_eval.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/hpc/memweave_qwen25_3b_locomo_eval.py:1)
  - [hpc/memweave-qwen25-locomo.sh](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/hpc/memweave-qwen25-locomo.sh:1)
- Adds category-specific retrieval `k`, token accounting, cache control, and a Slurm launcher.

## Core Modules

| Module | Responsibility |
|---|---|
| [load_dataset.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/load_dataset.py:8) | LoCoMo parsing and sample construction |
| [precompression.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/precompression.py:20) | LLMLingua-based Step 1 turn compression |
| [prestorage_pipeline.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/prestorage_pipeline.py:403) | Step 1 note building, Step 2 reasoning memory generation, cache orchestration |
| [memory_layer.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/memory_layer.py:261) | Original structured memory notes and evolution logic |
| [memory_layer_robust.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/memory_layer_robust.py:353) | Plain-text robust version without JSON-schema dependency |
| [llm_text_parsers.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/llm_text_parsers.py:128) | Prompt templates and parsers for robust LLM interactions |
| [token_usage.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/token_usage.py:91) | Per-operation token and runtime accounting |
| [test_advanced.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/test_advanced.py:43) | Original evaluation harness |
| [test_advanced_robust.py](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/test_advanced_robust.py:58) | Recommended evaluation harness |

## Recommended Entry Points

### Robust evaluation

```bash
python test_advanced_robust.py \
  --backend vllm \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset data/locomo10.json \
  --output results_robust_qwen25_3b.json
```

### Robust evaluation with Step 1 pre-compression

```bash
python test_advanced_robust.py \
  --backend vllm \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset data/locomo10.json \
  --output results_robust_qwen25_3b_precompress.json \
  --enable-precompress \
  --llmlingua-model-dir /path/to/llmlingua-2-bert-base-multilingual-cased-meetingbank \
  --llmlingua-device cpu \
  --llmlingua-rate 0.90
```

### Slurm / HPC entrypoint

```bash
sbatch hpc/memweave-qwen25-locomo.sh
```

## Naming Notes

- `MemWeaveMine` is the canonical project name in this workspace.
- The name emphasizes the mechanism: weaving turn-level notes, retrieved context, and higher-order reasoning memories into one searchable memory fabric.
- Legacy file paths such as `hpc/locomem-qwen25-locomo.sh`, `hpc/light-amem-qwen25-locomo.sh`, and `hpc/light_amem_qwen25_3b_locomo_eval.py` are kept as compatibility wrappers.
- Internal class names such as `AgenticMemorySystem` and `RobustAgenticMemorySystem` remain unchanged to avoid unnecessary API churn.
- References to A-Mem, PREMem, or LightMem inside the codebase usually describe method lineage, not the canonical project name.

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Core runtime dependencies include `sentence-transformers`, `transformers`, `torch`, `scikit-learn`, `openai`, `ollama`, `litellm`, `llmlingua`, and the metric packages used by `utils.py`.

## Citation and Origin

This workspace descends from the A-Mem codebase and adds local LoCoMo-oriented evaluation and memory-construction changes. If you use the underlying A-Mem method in research, cite the original paper:

```bibtex
@inproceedings{xu2025amem,
  title={A-Mem: Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## License

MIT. See [LICENSE](/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/LICENSE).
