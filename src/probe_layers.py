#!/usr/bin/env python3

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from probe import load_all

def oof_scores(H, y, groups, n_splits, C):

    scores = np.zeros(len(y), dtype=np.float64)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(H, y, groups):
        if len(np.unique(y[tr])) < 2:
            scores[te] = 0.0
            continue
        sc = StandardScaler().fit(H[tr])
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(sc.transform(H[tr]), y[tr])
        scores[te] = clf.decision_function(sc.transform(H[te]))
    return scores

def within_task_auc(scores, y, task):

    num, den = 0.0, 0
    for t in np.unique(task):
        m = task == t
        pos = scores[m][y[m] == 1]
        neg = scores[m][y[m] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        for p in pos:
            num += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        den += len(pos) * len(neg)
    return (num / den if den > 0 else float("nan")), den

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default="artifacts/features")
    ap.add_argument("--out", default="outputs/probe_layers_results.json")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--min_samples", type=int, default=100)
    ap.add_argument("--min_class", type=int, default=15)
    ap.add_argument("--max_round", type=int, default=-1, help="-1=no limit")
    args = ap.parse_args()

    X, metas, layer_ids = load_all(args.feat_dir)
    y = np.array([m["success"] for m in metas], dtype=int)
    task = np.array([m["task_idx"] for m in metas])
    rnd = np.array([m["round"] for m in metas])
    anchor = np.array([m["anchor"] for m in metas])

    max_r = int(rnd.max()) if args.max_round < 0 else args.max_round
    results = {}

    for a in ["pre_gen", "post_gen"]:

        rounds = []
        for t in range(max_r + 1):
            m = (anchor == a) & (rnd == t)
            n = int(m.sum())
            npos = int(y[m].sum())
            if n >= args.min_samples and min(npos, n - npos) >= args.min_class:
                rounds.append(t)

        auc_tab = np.full((len(layer_ids), len(rounds)), np.nan)
        wt_tab = np.full((len(layer_ids), len(rounds)), np.nan)
        pairs_row = []

        for j, t in enumerate(rounds):
            m = (anchor == a) & (rnd == t)
            ym, tm = y[m], task[m]
            npairs = None
            for li in range(len(layer_ids)):
                H = X[m, li, :].astype(np.float32)
                s = oof_scores(H, ym, tm, args.n_splits, args.C)
                auc_tab[li, j] = roc_auc_score(ym, s)
                wt, npairs = within_task_auc(s, ym, tm)
                wt_tab[li, j] = wt
            pairs_row.append(npairs)
            print(f"[{a}] round {t} done (n={int(m.sum())}, pairs={npairs})", flush=True)

        results[a] = {
            "layer_ids": layer_ids,
            "rounds": rounds,
            "n_pairs": pairs_row,
            "auc": auc_tab.tolist(),
            "wt_auc": wt_tab.tolist(),
        }

        for name, tab in [("cross-task AUC", auc_tab), ("within-task AUC", wt_tab)]:
            print(f"\n===== {a} - {name} =====")
            header = "layer |" + "".join(f"  r{t:<4}" for t in rounds)
            print(header)
            print("-" * len(header))
            for li, L in enumerate(layer_ids):
                row = f"{L:>4} |" + "".join(f"  {tab[li, j]:.3f}" for j in range(len(rounds)))

                print(row)
            best = [layer_ids[int(np.nanargmax(tab[:, j]))] for j in range(len(rounds))]
            print("best layer:" + "".join(f"  L{b:<4}" for b in best))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()
