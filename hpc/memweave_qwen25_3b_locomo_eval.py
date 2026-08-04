#!/usr/bin/env python
"""Canonical MemWeaveMine entrypoint wrapper for the LoCoMo evaluation pipeline."""

import sys
from pathlib import Path

REPO_ROOT_ON_DISK = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_ON_DISK) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_ON_DISK))

from hpc.memweave_qwen25_3b_locomo_driver import main


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # This wrapper is the actual HPC entrypoint; log and preserve any uncaught exit.
        print(
            "[DEBUG_QWEN_COMPARE] PROCESS_EXCEPTION "
            f"entrypoint={Path(__file__).resolve()} exception_type={type(exc).__name__} "
            f"message={str(exc)[:100]!r}"
        )
        raise
