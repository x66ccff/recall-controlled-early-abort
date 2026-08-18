#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from run_paper_matrix import CELLS

STRATEGIES = (
    ("searched_best", "Cascade"),
    ("best_single_gate", "Best single gate"),
    ("uniform", "Uniform"),
)
MODEL_LABELS = {
    "qwen2.5-7b": "Qwen-2.5-7B",
    "llama3.2-3b": "Llama-3.2-3B",
    "llama3.2-3b-strict": "Llama-3.2-3B",
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen3-1.7b-nothink": "Qwen3-1.7B",
}

def parse_floats(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",") if value.strip()]

def feature_dir(artifacts_root: Path, cell) -> Path:
    root = artifacts_root / cell.artifact_dir / cell.model_key
    for dirname in ("features", "features_v2"):
        candidate = root / dirname
        if list(candidate.glob("meta.shard*.jsonl")):
            return candidate
    raise SystemExit(f"missing feature metadata for {cell.slug}: {root}")

def alive_rows(artifacts_root: Path, cell, max_round: int) -> list[dict[str, object]]:
    episodes = set()
    alive = {round_idx: set() for round_idx in range(1, max_round + 1)}
    directory = feature_dir(artifacts_root, cell)
    for path in sorted(directory.glob("meta.shard*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                meta = json.loads(line)
                episode = (meta["task_idx"], meta["rollout_k"])
                episodes.add(episode)
                round_idx = int(meta["round"]) + 1
                if meta["anchor"] == "post_gen" and round_idx in alive:
                    alive[round_idx].add(episode)
    denominator = len(episodes)
    if not denominator:
        raise SystemExit(f"no episodes found in {directory}")
    return [{
        "environment": cell.environment,
        "cell": cell.slug,
        "model": MODEL_LABELS[cell.model_key],
        "gate_round": round_idx,
        "episodes_alive": len(alive[round_idx]),
        "episodes_total": denominator,
        "fraction_alive": len(alive[round_idx]) / denominator,
    } for round_idx in range(1, max_round + 1)]

def find_result(matrix_root: Path, cell, target: float, scorer: str) -> Path:
    directory = matrix_root / cell.environment / cell.slug / cell.model_key
    candidates = sorted(directory.glob(f"cascade_{scorer}_L{cell.layer}_*.json"))
    for path in candidates:
        data = json.loads(path.read_text())
        config = data.get("_config", {})
        if (config.get("scorer") == scorer
                and config.get("margin_mode") == "delta"
                and abs(float(config.get("margin_delta", -1)) - 0.02) < 1e-12
                and abs(float(config.get("target_recall", -1)) - target) < 1e-12):
            return path
    raise SystemExit(f"missing target={target:.2f} result for {cell.slug} in {directory}")

def frontier_rows(matrix_root: Path, cell, targets: list[float], scorer: str):
    rows = []
    for target in targets:
        path = find_result(matrix_root, cell, target, scorer)
        data = json.loads(path.read_text())
        for key, label in STRATEGIES:
            values = [100 * float(value) for value in data[key]["saved"]]
            rows.append({
                "environment": cell.environment,
                "cell": cell.slug,
                "model": MODEL_LABELS[cell.model_key],
                "target": target,
                "strategy": label,
                "saved_pct_mean": statistics.mean(values),
                "saved_pct_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "seeds": len(values),
                "source": str(path),
            })
    return rows

def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def plot_alive(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), sharey=True)
    for ax, environment in zip(axes, ("textcraft", "webshop")):
        env_rows = [row for row in rows if row["environment"] == environment]
        labels = list(dict.fromkeys(row["model"] for row in env_rows))
        for label in labels:
            points = [row for row in env_rows if row["model"] == label]
            ax.plot([row["gate_round"] for row in points],
                    [row["fraction_alive"] for row in points], marker="o", label=label)
        ax.set_title("TextCraft" if environment == "textcraft" else "WebShop")
        ax.set_xlabel("Gate round")
        ax.set_xticks(range(1, 7))
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Fraction alive")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_frontier(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.0), sharex=True, sharey=True)
    for ax, cell in zip(axes.flat, CELLS):
        cell_rows = [row for row in rows if row["cell"] == cell.slug]
        for _, label in STRATEGIES:
            points = [row for row in cell_rows if row["strategy"] == label]
            ax.errorbar([row["target"] for row in points],
                        [row["saved_pct_mean"] for row in points],
                        yerr=[row["saved_pct_std"] for row in points],
                        marker="o", linewidth=1.5, capsize=2, label=label)
        ax.set_title(f"{cell.environment.title()} {MODEL_LABELS[cell.model_key]}", fontsize=9)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Recall target")
    for ax in axes[:, 0]:
        ax.set_ylabel("Saved (%)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--matrix-root", type=Path,
                        default=root / "outputs" / "paper-matrix")
    parser.add_argument("--artifacts-root", type=Path, default=root / "artifacts")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs" / "paper-diagnostics")
    parser.add_argument("--targets", default="0.90,0.92,0.95,0.97")
    parser.add_argument("--scorer", default="stacking")
    parser.add_argument("--max-round", type=int, default=6)
    parser.add_argument("--csv-only", action="store_true")
    args = parser.parse_args()

    targets = parse_floats(args.targets)
    alive = [row for cell in CELLS for row in alive_rows(args.artifacts_root, cell, args.max_round)]
    frontier = [row for cell in CELLS for row in frontier_rows(
        args.matrix_root, cell, targets, args.scorer)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "alive.csv", alive)
    write_csv(args.output_dir / "frontier.csv", frontier)
    if not args.csv_only:
        plot_alive(args.output_dir / "alive.png", alive)
        plot_frontier(args.output_dir / "frontier.png", frontier)
    print(f"wrote {args.output_dir / 'alive.csv'}")
    print(f"wrote {args.output_dir / 'frontier.csv'}")
    if not args.csv_only:
        print(f"wrote {args.output_dir / 'alive.png'}")
        print(f"wrote {args.output_dir / 'frontier.png'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
