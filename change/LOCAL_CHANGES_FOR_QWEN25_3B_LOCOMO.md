Local MemWeave additions for the Qwen2.5-3B LoCoMo evaluation workflow:

1. `hpc/memweave_qwen25_3b_locomo_driver.py`
   - Canonical driver for the Qwen2.5-3B LoCoMo run.
   - Reuses the repo memory code, but applies category-specific `k` values:
     - Multi-hop: 10
     - Temporal: 10
     - Open-domain: 50
     - Single-hop: 10
     - Adversarial: 10
   - Uses the local `all-MiniLM-L6-v2` path by monkeypatching `SentenceTransformer` before importing repo modules.
   - Computes LoCoMo category metrics and token-usage summaries for the current HPC evaluation workflow.

2. `hpc/memweave-qwen25-locomo.sh`
   - Canonical Slurm script for one `a100-pcie-40gb`.
   - Starts a local vLLM server from the local Qwen2.5-3B-Instruct path.
   - Runs the external driver after the server is ready.

Why these files were needed:

- The stock repo entrypoint accepts only one global `--retrieve_k`, but the paper uses category-specific `k` for Qwen2.5-3B.
- The stock evaluation script also computes extra metrics that pull in additional model/runtime dependencies not needed for your Table 1 target.
- The local workspace now uses `MemWeave` as its canonical project name, while legacy `locomem`, `amem`, and `light_amem` module paths remain as compatibility wrappers.
