#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODULES = ("numpy", "scipy", "sklearn", "torch", "transformers", "vllm", "agentenv")
DEFAULT_MODELS = {
    "QWEN25_MODEL": "Qwen/Qwen2.5-7B-Instruct",
    "LLAMA32_MODEL": "unsloth/Llama-3.2-3B-Instruct",
    "QWEN3_MODEL": "Qwen/Qwen3-1.7B",
}

def model_ref_ok(value: str) -> bool:
    path = Path(value)
    if path.is_absolute() or value.startswith("."):
        return path.is_dir()
    return "/" in value and not any(char.isspace() for char in value)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-local-models", action="store_true")
    parser.add_argument("--require-alfworld", action="store_true",
                        help="also require the AgentGym ALFWorld environment package")
    args = parser.parse_args()
    failures = []

    for relative in ("scripts/rollout_vllm.py", "src/features.py", "scripts/stacking_cascade.py"):
        path = ROOT / relative
        ok = path.is_file()
        print(f"{'OK' if ok else 'FAIL':<5} source {relative}")
        if not ok:
            failures.append(relative)

    for module in REQUIRED_MODULES:
        ok = importlib.util.find_spec(module) is not None
        print(f"{'OK' if ok else 'FAIL':<5} python module {module}")
        if not ok:
            failures.append(f"module:{module}")

    if args.require_alfworld:
        for module in ("alfworld", "agentenv_alfworld"):
            ok = importlib.util.find_spec(module) is not None
            print(f"{'OK' if ok else 'FAIL':<5} python module {module}")
            if not ok:
                failures.append(f"module:{module}")

    for key, default in DEFAULT_MODELS.items():
        value = os.environ.get(key, default)
        ok = model_ref_ok(value)
        if args.require_local_models:
            ok = Path(value).is_dir()
        print(f"{'OK' if ok else 'FAIL':<5} {key}={value}")
        if not ok:
            failures.append(key)

    try:
        import torch

        has_gpu = bool(torch.cuda.is_available() and torch.cuda.device_count())
        detail = f"cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}"
    except Exception as exc:
        has_gpu = False
        detail = str(exc)
    status = "OK" if has_gpu else ("FAIL" if args.require_gpu else "WARN")
    print(f"{status:<5} GPU {detail}")
    if args.require_gpu and not has_gpu:
        failures.append("gpu")

    if failures:
        print(f"PREFLIGHT FAILED: {len(failures)} required checks")
        return 1
    print("PREFLIGHT PASSED")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
