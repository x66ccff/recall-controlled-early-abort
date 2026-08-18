#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

def parse_floats(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",") if value.strip()]

def load_sweep(result_dir: Path, scorer: str, strategy: str) -> dict[float, dict[str, object]]:
    sweep = {}
    for path in sorted(result_dir.glob(f"cascade_{scorer}_*.json")):
        data = json.loads(path.read_text())
        config = data.get("_config", {})
        if config.get("margin_mode") != "none" or config.get("scorer") != scorer:
            continue
        target = round(float(config["target_recall"]), 8)
        values = data.get(strategy, {})
        recall = [float(value) for value in values.get("recall", [])]
        saved = [float(value) for value in values.get("saved", [])]
        if recall and len(recall) == len(saved):
            sweep[target] = {"path": path, "recall": recall, "saved": saved}
    if not sweep:
        raise SystemExit(f"no unmargined {scorer!r} cascade results in {result_dir}")
    return sweep

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--scorer", default="probe")
    parser.add_argument("--strategy", default="searched_best")
    parser.add_argument("--targets", default="0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97")
    parser.add_argument("--deltas", default="0,0.01,0.02")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    targets = parse_floats(args.targets)
    deltas = parse_floats(args.deltas)
    sweep = load_sweep(args.result_dir, args.scorer, args.strategy)
    rows = []
    for target in targets:
        for delta in deltas:
            shifted = round(target + delta, 8)
            if shifted not in sweep:
                raise SystemExit(
                    f"missing unmargined target {shifted:.2f}, required for "
                    f"target={target:.2f}, delta={delta:.2f}"
                )
            item = sweep[shifted]
            recalls = item["recall"]
            savings = item["saved"]
            rows.append({
                "target": target,
                "delta": delta,
                "shifted_target": shifted,
                "violations": sum(value < target for value in recalls),
                "seeds": len(recalls),
                "recall_mean": statistics.mean(recalls),
                "saved_pct_mean": 100 * statistics.mean(savings),
                "source": str(item["path"]),
            })

    print("target\t" + "\t".join(f"delta={delta:.2f}" for delta in deltas))
    for target in targets:
        by_delta = {row["delta"]: row for row in rows if row["target"] == target}
        print(
            f"{target:.2f}\t" + "\t".join(
                f"{by_delta[delta]['violations']}/{by_delta[delta]['seeds']}"
                for delta in deltas
            )
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
