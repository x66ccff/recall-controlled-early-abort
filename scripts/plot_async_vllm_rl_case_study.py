#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

TIDY_COLUMNS = ("wall_time_h", "run", "metric", "value")
PANEL_ORDER = (
    ("validation_success", "Validation success", ""),
    ("sequence_length_m", "Global sequence length", ""),
    ("average_rounds", "Average rounds", ""),
)
METRIC_ALIASES = {
    "success": "validation_success",
    "validation_success": "validation_success",
    "sequence_length": "sequence_length_m",
    "sequence_length_m": "sequence_length_m",
    "average_rounds": "average_rounds",
    "episode_length": "average_rounds",
    "episode_length_mean": "average_rounds",
}
DEFAULT_COLORS = ("#2f78b7", "#8b929a", "#2f8f83", "#a45f6a")


def parse_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def run_id(column: str) -> str:
    parts = column.split("/")
    if len(parts) >= 2:
        return parts[-2]
    return column


def read_tidy_metrics(path: Path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in TIDY_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"missing required columns in {path}: {', '.join(missing)}")
        out: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        for row in reader:
            wall_time = parse_float(row["wall_time_h"])
            value = parse_float(row["value"])
            metric = METRIC_ALIASES.get(row["metric"].strip())
            run = row["run"].strip()
            if wall_time is None or value is None or metric is None or not run:
                continue
            out[metric][run].append((wall_time, value))
    return {metric: {run: sorted(points) for run, points in series.items()}
            for metric, series in out.items()}


def read_wide_export(path: Path, scale: float) -> dict[str, list[tuple[float, float]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "Step" not in fields:
            raise SystemExit(f"missing Step column in {path}")
        out: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in reader:
            step = parse_float(row["Step"])
            if step is None:
                continue
            for column in fields:
                if column == "Step":
                    continue
                value = parse_float(row.get(column, ""))
                if value is not None:
                    out[run_id(column)].append((step, value / scale))
    return {run: sorted(points) for run, points in out.items()}


def shared_initial_success(
    success: dict[str, list[tuple[float, float]]],
) -> float | None:
    values = [
        value
        for points in success.values()
        for step, value in points
        if abs(step) <= 1e-12
    ]
    if not values:
        return None
    first = values[0]
    if any(abs(value - first) > 1e-9 for value in values):
        raise SystemExit(
            "Step-0 validation-success values are not shared; "
            "provide --metrics-csv to specify the intended initial points."
        )
    return first


def align_exports(
    success_csv: Path,
    tokens_csv: Path,
    wall_time_csv: Path,
    rounds_csv: Path | None = None,
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    success = read_wide_export(success_csv, 1.0)
    sequence = read_wide_export(tokens_csv, 1_000_000.0)
    rounds = read_wide_export(rounds_csv, 1.0) if rounds_csv is not None else {}
    wall = read_wide_export(wall_time_csv, 3600.0)
    common_horizon = min(points[-1][1] for points in wall.values() if points)
    initial_success = shared_initial_success(success)

    def align(metric: dict[str, list[tuple[float, float]]]) -> dict[str, list[tuple[float, float]]]:
        out = {}
        for run, points in metric.items():
            wall_lookup = dict(wall.get(run, []))
            aligned = [
                (wall_lookup[step], value)
                for step, value in points
                if step in wall_lookup and wall_lookup[step] <= common_horizon + 1e-9
            ]
            if aligned:
                out[run] = aligned
        return out

    validation_success = align(success)
    if initial_success is not None:
        for run, points in validation_success.items():
            if points and points[0][0] > 1e-9:
                validation_success[run] = [(0.0, initial_success), *points]

    metrics = {
        "validation_success": validation_success,
        "sequence_length_m": align(sequence),
    }
    if rounds:
        metrics["average_rounds"] = align(rounds)
    return metrics


def parse_label(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("labels must have the form RUN=DISPLAY")
    run, label = spec.split("=", 1)
    return run.strip(), label.strip()


def apply_labels(
    metrics: dict[str, dict[str, list[tuple[float, float]]]],
    labels: dict[str, str],
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    if not labels:
        return metrics
    out: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for metric, series in metrics.items():
        out[metric] = {labels.get(run, run): values for run, values in series.items()}
    return out


def render_layout_svg(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    panels = ("Validation success", "Global sequence length", "Average rounds")
    width, height = 680, 230
    panel_w = 188
    panel_h = 112
    gap = 36
    left = 34
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Times,serif;fill:#333} .muted{fill:#777}</style>',
    ]
    for i, title in enumerate(panels):
        x = left + i * (panel_w + gap)
        y = 38
        parts.append(
            f'<text x="{x + panel_w / 2}" y="20" font-size="12" '
            f'text-anchor="middle" font-weight="bold">{title}</text>'
        )
        parts.append(
            f'<rect x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" '
            'fill="none" stroke="#8a8a8a" stroke-width="1"/>'
        )
        parts.append(
            f'<text class="muted" x="{x + panel_w / 2}" y="{y + 50}" '
            'font-size="11" text-anchor="middle">wall-time trace</text>'
        )
    parts.append(
        f'<text x="{width / 2}" y="{height - 28}" font-size="11" '
        'text-anchor="middle">Wall time (h)</text>'
    )
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def render(
    metrics: dict[str, dict[str, list[tuple[float, float]]]] | None,
    output: Path,
) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        if metrics is None and output.suffix.lower() == ".svg":
            render_layout_svg(output)
            return
        raise SystemExit(
            "matplotlib is required for measured trace plotting; "
            "install requirements-analysis.txt or use --layout-only --output out.svg"
        )

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(PANEL_ORDER), figsize=(6.6, 2.15),
                             sharex=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.055, h_pad=0.025, wspace=0.16, hspace=0.02)
    if metrics is None:
        for i, (ax, (_, title, ylabel)) in enumerate(zip(axes, PANEL_ORDER)):
            ax.set_title(title, fontsize=8, fontweight="bold")
            ax.set_xlabel("Wall time (h)" if i == 1 else "", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(color="#e6e8eb", linewidth=0.45)
            ax.text(0.5, 0.5, "wall-time\ntrace", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="#777777")
    else:
        run_order: list[str] = []
        for _, series in metrics.items():
            for run in series:
                if run not in run_order:
                    run_order.append(run)
        colors = {run: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, run in enumerate(run_order)}
        xmax = max(x for _, series in metrics.items() for points in series.values() for x, _ in points)
        for i, (ax, (metric, title, ylabel)) in enumerate(zip(axes, PANEL_ORDER)):
            ax.set_title(title, fontsize=8, fontweight="bold")
            ax.set_xlabel("Wall time (h)" if i == 1 else "", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6.5, length=2.2, colors="#4a4f55")
            ax.grid(axis="y", color="#e6e8eb", linewidth=0.45)
            ax.set_axisbelow(True)
            ax.set_xlim(0, xmax)
            for run in run_order:
                points = sorted(metrics.get(metric, {}).get(run, []))
                if not points:
                    continue
                xs = [x for x, _ in points]
                values = [y for _, y in points]
                color = colors[run]
                ax.plot(xs, values, color=color, linewidth=1.25, label=run)
            if metric == "validation_success":
                ax.set_ylim(0.0, 1.0)
                ax.set_yticks([0.0, 0.5, 1.0])
            elif metric == "sequence_length_m":
                ax.set_yticks([0.2, 0.6, 1.0])
            else:
                ax.margins(y=0.12)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)),
                   bbox_to_anchor=(0.5, -0.10), frameon=False, fontsize=7,
                   handlelength=1.8, columnspacing=1.5)
    suffix = output.suffix.lower()
    if suffix in {".pdf", ".svg"}:
        fig.savefig(output, bbox_inches="tight")
    else:
        fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    default_metrics = root / "artifacts" / "appendix-i" / "async_vllm_rl_metrics.csv"
    parser.add_argument("--metrics-csv", type=Path, default=None,
                        help="tidy CSV with wall_time_h,run,metric,value columns")
    parser.add_argument("--success-csv", type=Path, default=None,
                        help="exported validation-success metric CSV")
    parser.add_argument("--tokens-csv", type=Path, default=None,
                        help="exported sequence-length metric CSV")
    parser.add_argument("--wall-time-csv", type=Path, default=None,
                        help="exported cumulative wall-time metric CSV")
    parser.add_argument("--rounds-csv", type=Path, default=None,
                        help="exported average-rounds metric CSV")
    parser.add_argument("--label", action="append", type=parse_label, default=[],
                        help="rename a run in plots, as RUN=DISPLAY")
    parser.add_argument("--output", type=Path,
                        default=root / "outputs" / "appendix-i" / "async_vllm_rl_case_study.svg")
    parser.add_argument("--bundle", action="store_true",
                        help="write PDF, SVG, and PNG versions using the output stem")
    parser.add_argument("--layout-only", dest="layout_only",
                        action="store_true",
                        help="render the plot panel structure without a trace CSV")
    args = parser.parse_args()

    if args.layout_only:
        metrics = None
    elif args.metrics_csv is not None:
        metrics = read_tidy_metrics(args.metrics_csv)
    elif args.success_csv or args.tokens_csv or args.wall_time_csv or args.rounds_csv:
        if not (args.success_csv and args.tokens_csv and args.wall_time_csv):
            raise SystemExit(
                "--success-csv, --tokens-csv, and --wall-time-csv must be supplied together"
            )
        metrics = align_exports(args.success_csv, args.tokens_csv, args.wall_time_csv, args.rounds_csv)
    elif default_metrics.exists():
        metrics = read_tidy_metrics(default_metrics)
    else:
        raise SystemExit(
            "Provide --metrics-csv, the three exported metric CSVs, "
            "or pass --layout-only."
        )

    metrics = None if metrics is None else apply_labels(metrics, dict(args.label))
    outputs = [args.output]
    if args.bundle:
        outputs = [
            args.output.with_suffix(".pdf"),
            args.output.with_suffix(".svg"),
            args.output.with_suffix(".png"),
        ]
    for output in outputs:
        render(metrics, output)
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
