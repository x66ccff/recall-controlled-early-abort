#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

def run(command: list[str]) -> int:
    print("+ " + " ".join(command), flush=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return subprocess.run(command, cwd=ROOT, env=environment).returncode

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke", help="run the synthetic end-to-end analysis test")
    for name, help_text in (
        ("matrix", "run the TextCraft/WebShop by three-model analysis"),
        ("certify", "certify a frozen cascade on independent outcomes"),
        ("alfworld", "run an ALFWorld workflow stage"),
        ("appendix-i", "render the Appendix-I asynchronous RL-training case-study plot"),
    ):
        sub.add_parser(name, help=help_text, add_help=False)
    args, passthrough = parser.parse_known_args()
    if args.command == "smoke" and passthrough:
        parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
    if args.command == "smoke":
        return run([sys.executable, str(SCRIPTS / "smoke_test.py")])
    if args.command == "matrix":
        return run([sys.executable, str(SCRIPTS / "run_paper_matrix.py"), *passthrough])
    if args.command == "certify":
        return run([sys.executable, str(SCRIPTS / "certify_frozen_cascade.py"), *passthrough])
    if args.command == "appendix-i":
        return run([sys.executable, str(SCRIPTS / "plot_async_vllm_rl_case_study.py"), *passthrough])
    return run(["bash", str(SCRIPTS / "run_alfworld_k4_repro.sh"), *passthrough])

if __name__ == "__main__":
    raise SystemExit(main())
