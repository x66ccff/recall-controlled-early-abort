#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy.stats import binomtest, ttest_rel

def parse_result(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("result must be LABEL=PATH")
    label, raw_path = spec.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("result must be LABEL=PATH")
    return label, Path(raw_path)

def load_result(label: str, path: Path, strategy: str) -> dict[str, object]:
    data = json.loads(path.read_text())
    if strategy not in data:
        raise SystemExit(f"{path} has no strategy {strategy!r}")
    values = data[strategy]
    recall = [float(value) for value in values.get("recall", [])]
    saved = [float(value) for value in values.get("saved", [])]
    if not recall or len(recall) != len(saved):
        raise SystemExit(f"invalid recall/saved arrays in {path}:{strategy}")
    return {"label": label, "path": str(path), "recall": recall, "saved": saved}

def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.pstdev(values) if len(values) > 1 else 0.0

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", type=parse_result, required=True,
                        help="repeat LABEL=PATH for each scorer")
    parser.add_argument("--strategy", default="searched_best")
    parser.add_argument("--target", type=float, default=0.97)
    parser.add_argument("--json", type=Path, help="optional machine-readable report")
    args = parser.parse_args()

    series = [load_result(label, path, args.strategy) for label, path in args.result]
    lengths = {len(item["recall"]) for item in series}
    if len(lengths) != 1:
        raise SystemExit("all result files must contain the same number of paired seeds")

    report: dict[str, object] = {
        "target": args.target,
        "strategy": args.strategy,
        "scorers": {},
        "paired_comparisons": [],
    }
    print("scorer\trecall\tsaved_pct\tmin_recall\tseeds_meeting_target")
    for item in series:
        recall = item["recall"]
        saved = item["saved"]
        recall_mean, recall_std = mean_std(recall)
        saved_mean, saved_std = mean_std(saved)
        meeting = sum(value >= args.target for value in recall)
        row = {
            "path": item["path"],
            "seeds": len(recall),
            "recall_mean": recall_mean,
            "recall_std": recall_std,
            "saved_pct_mean": 100 * saved_mean,
            "saved_pct_std": 100 * saved_std,
            "min_recall": min(recall),
            "seeds_meeting_target": meeting,
            "violations": len(recall) - meeting,
        }
        report["scorers"][item["label"]] = row
        print(
            f"{item['label']}\t{recall_mean:.3f}+/-{recall_std:.3f}\t"
            f"{100 * saved_mean:.1f}+/-{100 * saved_std:.1f}\t"
            f"{min(recall):.3f}\t{meeting}/{len(recall)}"
        )

    print("\npaired savings comparisons (first minus second)")
    for left_index, left in enumerate(series):
        for right in series[left_index + 1:]:
            left_saved = left["saved"]
            right_saved = right["saved"]
            differences = [100 * (a - b) for a, b in zip(left_saved, right_saved)]
            delta_mean = statistics.mean(differences)
            delta_se = (statistics.stdev(differences) / math.sqrt(len(differences))
                        if len(differences) > 1 else 0.0)
            t_result = ttest_rel(left_saved, right_saved)

            left_meets = [value >= args.target for value in left["recall"]]
            right_meets = [value >= args.target for value in right["recall"]]
            wins = sum(a and not b for a, b in zip(left_meets, right_meets))
            losses = sum(b and not a for a, b in zip(left_meets, right_meets))
            discordant = wins + losses
            sign_p = (float(binomtest(wins, discordant, 0.5, alternative="greater").pvalue)
                      if discordant else 1.0)
            comparison = {
                "first": left["label"],
                "second": right["label"],
                "saved_delta_points_mean": delta_mean,
                "saved_delta_points_se": delta_se,
                "paired_t": float(t_result.statistic),
                "paired_t_p_two_sided": float(t_result.pvalue),
                "recall_compliance_wins": wins,
                "recall_compliance_losses": losses,
                "discordant_seeds": discordant,
                "sign_test_p_one_sided": sign_p,
            }
            report["paired_comparisons"].append(comparison)
            print(
                f"{left['label']} - {right['label']}: "
                f"delta={delta_mean:.2f} points, SE={delta_se:.2f}, "
                f"t={t_result.statistic:.2f}, p(two-sided)={t_result.pvalue:.3f}; "
                f"compliance wins/losses={wins}/{losses}, "
                f"sign p(one-sided)={sign_p:.3f}"
            )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
