LLMLingua Pre-Compression Integration

This workspace, now named `MemWeave`, keeps the original A-Mem memory layer but
adds a PREMem Step 1 style preconstruction stage before long-term memory notes
are built from LoCoMo dialogue turns.

What changed

- `precompression.py`
  - Wraps `llmlingua.PromptCompressor` in a safe, optional PREMem-style Step 1
    message preconstructor.
  - Replaces PREMem's structured personal-information extraction with a much
    lighter single-message compression pass.
  - The Step 1 output for each message is `compress_text`, and the text stored
    into A-MEM becomes `Speaker <name>says : <compress_text>`.
  - Skips image-caption turns and short turns by default to reduce quality
    loss.
  - Falls back to the original text if LLMLingua is unavailable or compression
    fails.
- `test_advanced.py`
- `test_advanced_robust.py`
- `hpc/memweave_qwen25_3b_locomo_driver.py`
  - Accept the new CLI flags and store PREMem-style Step 1 compressed memories
    in isolated caches.
  - Also save an explicit PREMem-like Step 1 cache file:
    `premem_step1_ours.jsonl`.
- `hpc/memweave-qwen25-locomo.sh`
  - Canonical Slurm launcher for the MemWeave Qwen2.5-3B LoCoMo evaluation run.

New CLI flags

- `--enable-precompress`
- `--llmlingua-model-dir /path/to/llmlingua-2-bert-base-multilingual-cased-meetingbank`
- `--llmlingua-device cpu|cuda`
- `--llmlingua-rate 0.90`
- `--llmlingua-target-token -1`
- `--llmlingua-min-tokens 24`

Recommended first run

```bash
python test_advanced_robust.py \
  --backend vllm \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset data/locomo10.json \
  --enable-precompress \
  --llmlingua-model-dir /home/jwcao/xingzhuo/mimic_dataset_building/models/llmlingua-2-bert-base-multilingual-cased-meetingbank \
  --llmlingua-device cpu \
  --llmlingua-rate 0.90
```

HPC example

```bash
PRECOMPRESS=1 \
LLMLINGUA_MODEL_DIR=/home/jwcao/xingzhuo/mimic_dataset_building/models/llmlingua-2-bert-base-multilingual-cased-meetingbank \
LLMLINGUA_DEVICE=cpu \
sbatch hpc/memweave-qwen25-locomo.sh
```

Notes

- Default behavior is still uncompressed unless you explicitly enable it.
- Compressed runs use a different cache suffix so they do not overwrite the
  original plain caches.
- Conceptually, this project now borrows PREMem Step 1's "history pre-build"
  position in the pipeline, but deletes PREMem's LLM-based three-way structured
  extraction. The only Step 1 artifact here is per-message `compress_text`.
- The Step 1 cache file is stored alongside the memory cache directory, for
  example:
  `cached_memories_robust_<backend>_<model>_<cache_tag>/premem_step1_ours.jsonl`
  or `hpc/cache/memweave_qwen25_3b_locomo_eval/<cache_tag>/premem_step1_ours.jsonl`.
- Each JSONL row contains PREMem-like bookkeeping fields such as `sample_id`,
  `session_id`, `session_date`, `role`, `message_id`, `conversation`, and a
  `response` JSON string; but the response payload contains only
  `compress_text`-style compression results rather than three-way personal-info
  extraction.
- The current defaults are conservative to reduce quality regressions:
  image turns are skipped, short turns are skipped, and `rate=0.90`.
- The HPC script now defaults `LLMLINGUA_MODEL_DIR` to the local MeetingBank
  checkpoint above, so setting that variable is optional unless you want a
  different path.
- The result JSON now includes `token_statistics`, with a `lightmem_style`
  block that mirrors LightMem-style reporting:
  `summary_tokens`, `update_tokens`, `memory_build_total_tokens`,
  `qa_total_tokens`, `total_tokens`, and `calls`.
- In this MemWeave project, `summary_tokens` refers to the offline memory
  construction calls that behave most like LightMem's summary stage:
  metadata extraction + evolution decision + strengthen calls.
- For token evaluation, keep `FORCE_REBUILD_CACHE=1` so cached memories do not
  hide the offline memory-construction token cost.
