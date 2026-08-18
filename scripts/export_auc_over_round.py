#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

def parse_series(spec: str) -> tuple[str, Path, str, str]:
    parts = spec.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "series must be LABEL:PROBE_JSON:ANCHOR:LAYER")
    label, path, anchor, layer = parts
    return label, Path(path), anchor, str(int(layer))

def load_curve(label: str, path: Path, anchor: str, layer: str) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    try:
        per_round = data["probe"][anchor][layer]["per_round_auc"]
    except KeyError as exc:
        raise SystemExit(
            f"missing expected key: probe/{anchor}/{layer}/per_round_auc in {path}") from exc
    rows = []
    for round_idx, auc in sorted(per_round.items(), key=lambda kv: int(kv[0])):
        rows.append({
            "label": label,
            "probe_json": str(path),
            "anchor": anchor,
            "layer": int(layer),
            "round": int(round_idx) + 1,
            "auc": float(auc),
        })
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", type=parse_series, required=True,
                    help="LABEL:PROBE_JSON:ANCHOR:LAYER")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--png", default="")
    args = ap.parse_args()

    rows = []
    for label, path, anchor, layer in args.series:
        rows.extend(load_curve(label, path, anchor, layer))

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["label", "probe_json", "anchor", "layer", "round", "auc"])
        writer.writeheader()
        writer.writerows(rows)

    if args.png:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        labels = []
        for row in rows:
            if row["label"] not in labels:
                labels.append(row["label"])
        for label in labels:
            pts = [r for r in rows if r["label"] == label]
            ax.plot([r["round"] for r in pts], [r["auc"] for r in pts],
                    marker="o", linewidth=2, label=label)
        ax.axhline(0.5, color="0.5", linewidth=1, linestyle="--")
        ax.set_xlabel("Gate round")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        out_png = Path(args.png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=180)

    print(f"wrote {out_csv}")
    if args.png:
        print(f"wrote {args.png}")

if __name__ == "__main__":
    main()
