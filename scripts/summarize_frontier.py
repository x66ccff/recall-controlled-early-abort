#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

PATTERN = re.compile(
    r"^cascade_(?P<scorer>.+?)_L(?P<layer>\d+)_r(?P<target>[\d.]+)_"
    r"(?P<cal>cp|quantile)(?:_margin-(?P<margin>delta\d*|cp))?"
    r"(?:_delta(?P<delta>[\d.]+))?(?:_s(?P<seeds>\d+))?[.]json$"
)

def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.pstdev(values) if len(values) > 1 else 0.0

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="directory containing cascade JSON files")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if not args.result_dir.is_dir():
        parser.error(f"result directory does not exist: {args.result_dir}")

    rows = []
    for path in sorted(args.result_dir.glob("cascade_*.json")):
        match = PATTERN.match(path.name)
        if not match:
            print(f"skip unrecognized filename: {path.name}")
            continue
        data = json.loads(path.read_text())
        for strategy in ("searched_best", "best_single_gate", "uniform"):
            values = data.get(strategy, {})
            recalls = [float(value) for value in values.get("recall", [])]
            savings = [float(value) for value in values.get("saved", [])]
            if not recalls or len(recalls) != len(savings):
                raise ValueError(f"invalid recall/saved arrays in {path}:{strategy}")
            recall_mean, recall_std = mean_std(recalls)
            saved_mean, saved_std = mean_std(savings)
            rows.append({
                "file": path.name,
                "scorer": match["scorer"],
                "layer": int(match["layer"]),
                "target": float(match["target"]),
                "calibration": match["cal"],
                "delta": float(match["delta"]) if match["delta"] else 0.0,
                "seeds": len(recalls),
                "strategy": strategy,
                "recall_mean": recall_mean,
                "recall_std": recall_std,
                "saved_pct_mean": 100 * saved_mean,
                "saved_pct_std": 100 * saved_std,
            })

    if not rows:
        raise SystemExit("no valid cascade JSON files found")
    print("scorer\ttarget\tstrategy\trecall\tsaved_pct\tseeds")
    for row in rows:
        print(
            f"{row['scorer']}\t{row['target']:.2f}\t{row['strategy']}\t"
            f"{row['recall_mean']:.3f}+/-{row['recall_std']:.3f}\t"
            f"{row['saved_pct_mean']:.1f}+/-{row['saved_pct_std']:.1f}\t{row['seeds']}"
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
