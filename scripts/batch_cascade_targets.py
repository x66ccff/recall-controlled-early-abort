#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np

import stacking_cascade as sc

def target_tag(target: float) -> str:
    return f"{target:g}"

def output_name(scorer: str, layer: int, target: float, cal_method: str,
                margin_mode: str, margin_delta: float, seeds: int) -> str:
    mtag = ""
    if margin_mode == "delta":
        mtag = f"_delta{margin_delta:g}"
    elif margin_mode == "cp":
        mtag = "_margin-cp"
    return (
        f"cascade_{scorer}_L{layer}_r{target_tag(target)}_"
        f"{cal_method}{mtag}_s{seeds}.json"
    )

def calibrate_threshold(scores_pos, target, cal_method, alpha):
    return sc.calibrate_threshold(scores_pos, target, cal_method, alpha)

def run_batch(data, scorer_fn, seeds, targets, cal_method, alpha, *,
              max_round=6, grid=(1.0, 0.99, 0.98, 0.95, 0.90, 0.85),
              cal_frac=0.4, margin_mode="delta", margin_delta=0.02,
              margin_alpha=0.05):
    success, tokens, task = data["success"], data["tokens"], data["task"]
    n_ep = len(success)
    R = min(max_round, max(data["H"].keys()))
    suffix = np.column_stack([tokens[:, r:].sum(1) for r in range(R)])
    out = {
        t: {
            "searched_best": {"recall": [], "saved": [], "recall_valA": [], "combo": []},
            "best_single_gate": {
                "recall": [], "saved": [], "recall_valA": [], "combo": [],
                "selected_round": [],
            },
            "uniform": {"recall": [], "saved": [], "recall_valA": [], "combo": []},
        }
        for t in targets
    }

    for seed in seeds:
        print(f"[seed {seed}] scoring rounds once")
        score_mat = np.full((n_ep, R), np.nan)
        for r in range(1, R + 1):
            H, S, ep = data["H"][r], data["S"][r], data["ep"][r]
            score_mat[ep, r - 1] = scorer_fn(
                H, S, success[ep].astype(int), task[ep], seed
            )
        alive = ~np.isnan(score_mat)

        rng = np.random.default_rng(seed)
        uniq = np.unique(task)
        rng.shuffle(uniq)
        n_cal = int(cal_frac * len(uniq))
        cal_tasks = uniq[:n_cal]
        cal = np.isin(task, cal_tasks)
        calA = np.isin(task, cal_tasks[: n_cal // 2])
        calB = cal & ~calA

        def thr_at(mask, r, t):
            scores = score_mat[mask & success & alive[:, r], r]
            return calibrate_threshold(scores, t, cal_method, alpha)

        thr_cand = {r: {t: thr_at(calB, r, t) for t in grid} for r in range(R)}

        def make_sim(mask):
            a, sm = alive[mask], score_mat[mask]
            sc_success, sf, tk = success[mask], suffix[mask], tokens[mask].sum()
            idx = np.arange(mask.sum())
            n_pos = int(sc_success.sum())

            def sim(tau):
                hit = a & (sm < tau[None, :])
                any_hit = hit.any(1)
                first = np.argmax(hit, 1)
                lost = int((any_hit & sc_success).sum())
                recall = 1 - lost / max(1, n_pos)
                saved = sf[idx[any_hit], first[any_hit]].sum() / max(1e-9, tk)
                return recall, saved, n_pos - lost

            return sim, n_pos

        sim_A, n_posA = make_sim(calA)
        sim_test, _ = make_sim(~cal)

        for target in targets:
            if margin_mode == "none":
                def feasible(rc, kept):
                    return rc >= target
            elif margin_mode == "delta":
                req = min(target + margin_delta, 1.0)
                def feasible(rc, kept):
                    return rc >= req
            elif margin_mode == "cp":
                kept_min = n_posA + 1
                for kk in range(n_posA, -1, -1):
                    if sc.recall_lower_bound(kk, n_posA, margin_alpha) >= target:
                        kept_min = kk
                    else:
                        break
                def feasible(rc, kept):
                    return kept >= kept_min
            else:
                raise ValueError(margin_mode)

            best = None
            for combo in itertools.product(grid, repeat=R):
                tau = np.array([thr_cand[r][combo[r]] for r in range(R)])
                rc, sv, kept = sim_A(tau)
                if feasible(rc, kept) and (best is None or sv > best[1]):
                    best = (combo, sv, tau, rc)

            best_single = None
            for r in range(R):
                for gate_target in grid:
                    tau = np.full(R, -np.inf)
                    tau[r] = thr_cand[r][gate_target]
                    rc, sv, kept = sim_A(tau)
                    if feasible(rc, kept) and (
                        best_single is None or sv > best_single[1]
                    ):
                        best_single = (r, sv, tau, rc, gate_target)

            strategies = {
                "searched_best": best[2] if best else np.full(R, -np.inf),
                "best_single_gate": (
                    best_single[2] if best_single else np.full(R, -np.inf)
                ),
                "uniform": np.array(
                    [thr_at(cal, r, target ** (1.0 / R)) for r in range(R)]
                ),
            }
            for name, tau in strategies.items():
                rc, sv, _ = sim_test(tau)
                out[target][name]["recall"].append(float(rc))
                out[target][name]["saved"].append(float(sv))
                if name == "searched_best" and best:
                    out[target][name]["combo"].append(list(best[0]))
                    out[target][name]["recall_valA"].append(float(best[3]))
                elif name == "best_single_gate" and best_single:
                    out[target][name]["selected_round"].append(best_single[0] + 1)
                    out[target][name]["combo"].append([best_single[4]])
                    out[target][name]["recall_valA"].append(float(best_single[3]))

    return out

def summarize(target, res):
    parts = []
    for name in ("searched_best", "best_single_gate", "uniform"):
        rec = np.array(res[name]["recall"], dtype=float)
        sav = np.array(res[name]["saved"], dtype=float)
        parts.append(
            f"{name}: recall={rec.mean():.3f}+/-{rec.std():.3f} "
            f"saved={100*sav.mean():.1f}+/-{100*sav.std():.1f}%"
        )
    print(f"[target {target:g}] " + " | ".join(parts))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True,
                    help="artifact root containing features/rollouts")
    ap.add_argument("--output-results-root", default=None,
                    help="output directory for JSON files")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--scorer", choices=sorted(sc.SCORERS), default="stacking")
    ap.add_argument("--targets", default="0.9,0.92,0.95,0.97")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cal-method", choices=["cp", "quantile"], default="cp")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--margin-mode", choices=["none", "delta", "cp"], default="delta")
    ap.add_argument("--margin-delta", type=float, default=0.02)
    ap.add_argument("--margin-alpha", type=float, default=0.05)
    args = ap.parse_args()

    sc.RESULTS_ROOT = args.results_root
    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    data = sc.load_dataset(args.model, args.layer)
    results = run_batch(
        data,
        sc.SCORERS[args.scorer],
        list(range(args.seeds)),
        targets,
        args.cal_method,
        args.alpha,
        margin_mode=args.margin_mode,
        margin_delta=args.margin_delta,
        margin_alpha=args.margin_alpha,
    )

    output_root = args.output_results_root or args.results_root
    out_dir = Path(output_root) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    for target, res in results.items():
        res["_config"] = vars(args) | {"target_recall": target}
        name = output_name(
            args.scorer, args.layer, target, args.cal_method,
            args.margin_mode, args.margin_delta, args.seeds
        )
        path = out_dir / name
        if path.exists():
            backup = path.with_name(path.name + time.strftime(".overwritten_%Y%m%d_%H%M%S"))
            path.rename(backup)
            print(f"[warn] moved existing {path} to {backup}")
        path.write_text(json.dumps(res, indent=2))
        summarize(target, res)
        print(f"[write] {path}")

if __name__ == "__main__":
    main()
