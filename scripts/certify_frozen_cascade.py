#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import beta

def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "kept", "success"}

def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                rows.append(row)
    return rows

def counts_from_rows(rows: list[dict]) -> tuple[int, int]:
    n_success = 0
    n_kept = 0
    for index, row in enumerate(rows, 1):
        if "success" not in row:
            raise ValueError(f"row {index} lacks 'success'")
        success = as_bool(row["success"])
        if not success:
            continue
        if "kept" in row:
            kept = as_bool(row["kept"])
        elif "aborted" in row:
            kept = not as_bool(row["aborted"])
        else:
            raise ValueError(f"row {index} lacks 'kept' or 'aborted'")
        n_success += 1
        n_kept += int(kept)
    return n_success, n_kept

def lower_bound(kept: int, total: int, alpha: float) -> float:
    if total <= 0 or kept <= 0:
        return 0.0
    if kept >= total:
        return alpha ** (1.0 / total)
    return float(beta.ppf(alpha, kept, total - kept + 1))

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="CSV or JSONL independent outcomes")
    source.add_argument("--n-success", type=int, help="number of successful certification episodes")
    parser.add_argument("--n-success-kept", type=int)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-certified", action="store_true")
    args = parser.parse_args()

    if not 0 < args.target <= 1 or not 0 < args.alpha < 1:
        parser.error("target must be in (0,1] and alpha in (0,1)")
    if args.input:
        n_success, n_kept = counts_from_rows(load_rows(args.input))
    else:
        if args.n_success_kept is None:
            parser.error("--n-success-kept is required with --n-success")
        n_success, n_kept = args.n_success, args.n_success_kept
    if n_success < 0 or not 0 <= n_kept <= n_success:
        parser.error("counts must satisfy 0 <= kept <= successes")

    bound = lower_bound(n_kept, n_success, args.alpha)
    result = {
        "n_success": n_success,
        "n_success_kept": n_kept,
        "empirical_recall": n_kept / n_success if n_success else 0.0,
        "alpha": args.alpha,
        "cp_lower_bound": bound,
        "target": args.target,
        "certified": bound >= args.target,
        "maximum_certifiable_target_if_all_kept": (
            args.alpha ** (1.0 / n_success) if n_success else 0.0
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return int(args.require_certified and not result["certified"])

if __name__ == "__main__":
    raise SystemExit(main())
