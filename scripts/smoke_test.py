#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def write_dataset(root: Path) -> None:
    rng = np.random.default_rng(7)
    feature_dir = root / "synthetic" / "features"
    feature_dir.mkdir(parents=True)
    rows = []
    features = []
    n_rounds = 8
    for task in range(80):
        for rollout in range(2):
            success = int((task + 2 * rollout) % 5 != 0)
            signal = 1.0 if success else -1.0
            for round_index in range(n_rounds):
                for anchor, offset in (("pre_gen", 0), ("post_gen", 10)):
                    strength = 0.04 + 0.03 * round_index if anchor == "post_gen" else 0.02
                    vector = rng.normal(0, 1, size=8)
                    vector[:3] += signal * strength
                    features.append(vector[:, None].T)
                    rows.append({
                        "task_idx": task,
                        "rollout_k": rollout,
                        "round": round_index,
                        "anchor": anchor,
                        "token_idx": 20 * round_index + offset,
                        "success": success,
                        "n_rounds": n_rounds,
                        "hit_max_rounds": True,
                        "cur_round_lp": float(-1.2 + 0.02 * signal + rng.normal(0, 0.1)),
                        "prefix_len": 20 * round_index + offset,
                        "n_err": int(not success and round_index > 2),
                    })
    np.savez(
        feature_dir / "feat.shard0.npz",
        X=np.asarray(features, dtype=np.float32),
        layer_ids=np.asarray([2]),
    )
    with (feature_dir / "meta.shard0.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

def main() -> int:
    with tempfile.TemporaryDirectory(prefix="recall-cascade-smoke-") as directory:
        base = Path(directory)
        input_root = base / "input"
        output_root = base / "output"
        write_dataset(input_root)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "batch_cascade_targets.py"),
            "--results-root", str(input_root),
            "--output-results-root", str(output_root),
            "--model", "synthetic",
            "--layer", "2",
            "--scorer", "probe",
            "--targets", "0.90",
            "--seeds", "2",
            "--cal-method", "cp",
            "--margin-mode", "delta",
            "--margin-delta", "0.02",
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning")
        subprocess.run(command, cwd=ROOT, check=True, env=env)
        outputs = list((output_root / "synthetic").glob("cascade_probe_*.json"))
        if len(outputs) != 1:
            raise RuntimeError(f"expected one output JSON, found {len(outputs)}")
        result = json.loads(outputs[0].read_text())
        for strategy in ("searched_best", "best_single_gate", "uniform"):
            if len(result[strategy]["recall"]) != 2 or len(result[strategy]["saved"]) != 2:
                raise RuntimeError(f"invalid output schema for {strategy}")
        if len(result["best_single_gate"]["selected_round"]) != 2:
            raise RuntimeError("best single gate did not record one selected round per seed")
        print("SMOKE TEST PASSED")
        print(f"temporary output validated: {outputs[0].name}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
