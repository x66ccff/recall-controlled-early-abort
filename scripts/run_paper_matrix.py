#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

@dataclass(frozen=True)
class Cell:
    slug: str
    environment: str
    artifact_dir: str
    model_key: str
    layer: int

CELLS = (
    Cell("textcraft-qwen25", "textcraft", "textcraft/qwen2.5-7b", "qwen2.5-7b", 20),
    Cell("textcraft-llama32", "textcraft", "textcraft/llama3.2-3b", "llama3.2-3b", 14),
    Cell("textcraft-qwen3", "textcraft", "textcraft/qwen3-1.7b", "qwen3-1.7b", 28),
    Cell("webshop-qwen25", "webshop", "webshop/qwen2.5-7b", "qwen2.5-7b", 20),
    Cell("webshop-llama32", "webshop", "webshop/llama3.2-3b", "llama3.2-3b-strict", 14),
    Cell("webshop-qwen3", "webshop", "webshop/qwen3-1.7b", "qwen3-1.7b-nothink", 28),
)

def has_features(root: Path, model_key: str) -> bool:
    for dirname in ("features", "features_v2"):
        if list((root / model_key / dirname).glob("feat.shard*.npz")):
            return True
    return False

def command_text(command: list[str]) -> str:
    return " ".join(command)

def run(command: list[str], dry_run: bool) -> None:
    print("+ " + command_text(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)

def batch_command(cell: Cell, input_root: Path, output_root: Path, *, scorer: str,
                  targets: str, seeds: int, margin_mode: str = "delta") -> list[str]:
    return [
        sys.executable,
        str(SCRIPTS / "batch_cascade_targets.py"),
        "--results-root", str(input_root),
        "--output-results-root", str(output_root),
        "--model", cell.model_key,
        "--layer", str(cell.layer),
        "--scorer", scorer,
        "--targets", targets,
        "--seeds", str(seeds),
        "--cal-method", "cp",
        "--margin-mode", margin_mode,
        "--margin-delta", "0.02",
    ]

def stage2_command(cell: Cell, input_root: Path, output_root: Path, cal: str,
                   seeds: int) -> list[str]:
    return [
        sys.executable,
        str(SCRIPTS / "stacking_cascade.py"),
        "--results-root", str(input_root),
        "--output-results-root", str(output_root),
        "--model", cell.model_key,
        "--layer", str(cell.layer),
        "--mode", "stage2",
        "--scorer", "probe",
        "--cal-method", cal,
        "--seeds", str(seeds),
    ]

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=(
        "main", "scorer-ablation", "appendix-b", "appendix-c", "appendix-d", "all"
    ), default="main")
    parser.add_argument("--artifacts-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "paper-matrix")
    parser.add_argument("--cell", action="append", choices=[cell.slug for cell in CELLS])
    parser.add_argument("--targets", default="0.90,0.92,0.95,0.97")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = [cell for cell in CELLS if not args.cell or cell.slug in args.cell]
    required_cells = selected
    if args.scope in {"appendix-b", "appendix-c", "appendix-d"}:
        required_cells = [cell for cell in selected if cell.environment == "textcraft"]
        if args.scope in {"appendix-c", "appendix-d"}:
            required_cells = [cell for cell in required_cells if cell.slug == "textcraft-qwen25"]

    missing = []
    for cell in required_cells:
        cell_input = args.artifacts_root / cell.artifact_dir
        if not has_features(cell_input, cell.model_key):
            missing.append(f"{cell.slug}: {cell_input / cell.model_key / 'features'}")
    if missing and not args.dry_run:
        print("Missing feature artifacts:", file=sys.stderr)
        for item in missing:
            print(f"- {item}", file=sys.stderr)
        print("See the artifact layout in README.md or rerun with --dry-run.", file=sys.stderr)
        return 2

    scopes = {args.scope} if args.scope != "all" else {
        "main", "scorer-ablation", "appendix-b", "appendix-c", "appendix-d"
    }
    for cell in selected:
        cell_input = args.artifacts_root / cell.artifact_dir
        cell_output = args.output_root / cell.environment / cell.slug
        if "main" in scopes:
            run(batch_command(cell, cell_input, cell_output, scorer="stacking",
                              targets=args.targets, seeds=args.seeds), args.dry_run)
        if "scorer-ablation" in scopes:
            for scorer in ("probe", "surface"):
                run(batch_command(cell, cell_input, cell_output, scorer=scorer,
                                  targets="0.95", seeds=args.seeds), args.dry_run)

    textcraft = [cell for cell in selected if cell.environment == "textcraft"]
    if "appendix-b" in scopes:
        for cell in textcraft:
            cell_input = args.artifacts_root / cell.artifact_dir
            cell_output = args.output_root / "appendix-b" / cell.slug
            for cal in ("cp", "quantile"):
                run(stage2_command(cell, cell_input, cell_output, cal, args.seeds), args.dry_run)
    qwen25 = next((cell for cell in textcraft if cell.slug == "textcraft-qwen25"), None)
    if qwen25 and "appendix-c" in scopes:
        cell_input = args.artifacts_root / qwen25.artifact_dir
        cell_output = args.output_root / "appendix-c" / qwen25.slug
        for scorer in ("probe_mlp", "probe", "stacking"):
            run(batch_command(qwen25, cell_input, cell_output, scorer=scorer,
                              targets="0.97", seeds=args.seeds), args.dry_run)
    if qwen25 and "appendix-d" in scopes:
        cell_input = args.artifacts_root / qwen25.artifact_dir
        cell_output = args.output_root / "appendix-d" / qwen25.slug
        fine_grid = ",".join(f"{value / 100:.2f}" for value in range(90, 100))
        run(batch_command(qwen25, cell_input, cell_output, scorer="probe",
                          targets=fine_grid, seeds=args.seeds, margin_mode="none"), args.dry_run)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
