#!/usr/bin/env python

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def load_all(feat_dir):
    Xs, metas, layer_ids = [], [], None
    for f in sorted(glob.glob(os.path.join(feat_dir, "feat.shard*.npz"))):
        d = np.load(f)
        Xs.append(d["X"])
        if layer_ids is None:
            layer_ids = d["layer_ids"].tolist()
        else:
            assert d["layer_ids"].tolist() == layer_ids
        mf = f.replace("feat.shard", "meta.shard").replace(".npz", ".jsonl")
        metas.extend(json.loads(l) for l in open(mf))
    X = np.concatenate(Xs, axis=0)
    assert X.shape[0] == len(metas)
    return X, metas, layer_ids

def oof_scores(feat, y, groups, n_splits=5, C=1.0):

    scores = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(feat, y, groups):
        sc = StandardScaler().fit(feat[tr])
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(sc.transform(feat[tr]), y[tr])
        scores[te] = clf.decision_function(sc.transform(feat[te]))
    assert not np.isnan(scores).any()
    return scores

def oof_scores_mlp(feat, y, groups, n_splits, seed=0):
    s = np.zeros(len(y))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(feat, y, groups):
        sc = StandardScaler().fit(feat[tr])
        clf = MLPClassifier(hidden_layer_sizes=(256,),
                            alpha=1e-3,
                            early_stopping=True,
                            n_iter_no_change=10,
                            max_iter=200,
                            random_state=seed)
        clf.fit(sc.transform(feat[tr]), y[tr])
        s[te] = clf.predict_proba(sc.transform(feat[te]))[:, 1]
    return s

def fit_lr(Xtr, ytr, C):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    pipe.fit(Xtr, ytr)
    return pipe

def fit_mlp(Xtr, ytr, seed=0):

    sc = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3,
                        early_stopping=True, n_iter_no_change=10,
                        max_iter=200, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return lambda Xte: clf.predict_proba(sc.transform(Xte))[:, 1]

def within_task_pair_acc(scores, y, task_ids, rollout_ks):

    by_task = defaultdict(list)
    for s, yy, t in zip(scores, y, task_ids):
        by_task[t].append((s, yy))
    wins, ties, total = 0, 0, 0
    n_mixed = 0
    for t, items in by_task.items():
        pos = [s for s, yy in items if yy == 1]
        neg = [s for s, yy in items if yy == 0]
        if not pos or not neg:
            continue
        n_mixed += 1
        for p in pos:
            for n in neg:
                total += 1
                if p > n:
                    wins += 1
                elif p == n:
                    ties += 1
    acc = (wins + 0.5 * ties) / total if total else float("nan")
    return acc, n_mixed, total

def paired_bootstrap_delta(s_new, s_base, ya, ta, rka, n_boot=1000, seed=0):

    rng = np.random.default_rng(seed)
    uniq = np.unique(ta)
    diffs = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([np.where(ta == t)[0] for t in samp])
        d = (within_task_pair_acc(s_new[sel], ya[sel], ta[sel], rka[sel])[0]
             - within_task_pair_acc(s_base[sel], ya[sel], ta[sel], rka[sel])[0])
        diffs.append(d)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return lo, hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default="artifacts/features")
    ap.add_argument("--out", default="outputs/probe_results.json")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--mlp_sweep", action="store_true",
                    help="run an additional MLP probe on all layers with seed 0")
    args = ap.parse_args()

    X, metas, layer_ids = load_all(args.feat_dir)
    y = np.array([m["success"] for m in metas], dtype=int)
    task = np.array([m["task_idx"] for m in metas])
    rk = np.array([m["rollout_k"] for m in metas])
    rnd = np.array([m["round"] for m in metas])
    anchor = np.array([m["anchor"] for m in metas])
    prefix_len = np.array([m["prefix_len"] for m in metas], dtype=float)
    cur_lp = np.array([m["cur_round_lp"] for m in metas], dtype=float)
    n_err = np.array([m["n_err"] for m in metas], dtype=float)
    pre_token_idx = {
        (m["task_idx"], m["rollout_k"], m["round"]): m["token_idx"]
        for m in metas if m["anchor"] == "pre_gen"
    }
    n_gen = np.array([
        max(0, m["token_idx"] - pre_token_idx[
            (m["task_idx"], m["rollout_k"], m["round"])
        ]) if m["anchor"] == "post_gen" else 0
        for m in metas
    ], dtype=float)

    anchors = [a for a in ["pre_gen", "post_gen", "pre_gen_mean", "post_gen_mean"]
               if (anchor == a).any()]

    def surf_feats(m):

        return np.stack([cur_lp[m], n_gen[m], prefix_len[m], n_err[m]],
                        axis=1).astype(float)

    print(f"[probe] X{X.shape}, positive_rate={y.mean():.3f}, "
          f"n_tasks={len(set(task.tolist()))}, "
          f"max_round={rnd.max()}, anchors={anchors}", flush=True)

    results = {"layer_ids": layer_ids, "pos_rate": float(y.mean()),
               "anchors": anchors}

    results["baseline"] = {}
    baseline_scores = {}
    for a in anchors:
        m = anchor == a
        s = oof_scores(surf_feats(m), y[m], task[m], args.n_splits, C=1.0)
        baseline_scores[a] = s
        auc = roc_auc_score(y[m], s)
        wacc, nmix, npair = within_task_pair_acc(s, y[m], task[m], rk[m])
        results["baseline"][a] = {"auc": float(auc), "within_task_acc": float(wacc)}
        print(f"[baseline {a}] cur_lp+n_gen+prefix_len+n_err: AUC={auc:.3f}, "
              f"within-task={wacc:.3f} ({nmix} mixed-outcome tasks, {npair} pairs)",
              flush=True)

    results["probe"] = {}
    for a in anchors:
        m = anchor == a
        results["probe"][a] = {}
        for li, layer in enumerate(layer_ids):
            feat = X[m, li, :].astype(np.float32)
            s = oof_scores(feat, y[m], task[m], args.n_splits, args.C)
            auc = roc_auc_score(y[m], s)
            wacc, nmix, npair = within_task_pair_acc(s, y[m], task[m], rk[m])

            per_round = {}
            for t in range(int(rnd[m].max()) + 1):
                mm = rnd[m] == t
                if mm.sum() >= 50 and 0 < y[m][mm].mean() < 1:
                    per_round[int(t)] = float(roc_auc_score(y[m][mm], s[mm]))
            results["probe"][a][layer] = {
                "auc": float(auc), "within_task_acc": float(wacc),
                "per_round_auc": per_round}
            pr_str = " ".join(f"r{t}:{v:.2f}" for t, v in
                              sorted(per_round.items())[:6])
            print(f"[probe {a} L{layer:2d}] AUC={auc:.3f} "
                  f"within-task={wacc:.3f} | {pr_str}", flush=True)

    for a in anchors:
        best_l = max(results["probe"][a], key=lambda l: results["probe"][a][l]["auc"])
        results["probe"][a + "_best_layer"] = int(best_l)
        print(f"\n[summary {a}] best layer L{best_l}: "
              f"AUC={results['probe'][a][best_l]['auc']:.3f} "
              f"(baseline {results['baseline'][a]['auc']:.3f}), "
              f"within-task={results['probe'][a][best_l]['within_task_acc']:.3f} "
              f"(baseline {results['baseline'][a]['within_task_acc']:.3f})", flush=True)

    results["concat"] = {}
    for a in anchors:
        best_l = results["probe"][a + "_best_layer"]
        li = layer_ids.index(best_l)
        m = anchor == a
        comb = np.concatenate([X[m, li, :].astype(np.float32), surf_feats(m)], axis=1)
        s = oof_scores(comb, y[m], task[m], args.n_splits, args.C)
        auc = roc_auc_score(y[m], s)
        wacc, _, _ = within_task_pair_acc(s, y[m], task[m], rk[m])
        results["concat"][a] = {"auc": float(auc), "within_task_acc": float(wacc),
                                "layer": int(best_l)}
        print(f"[concat {a} L{best_l}+surf] AUC={auc:.3f} within-task={wacc:.3f} "
              f"(baseline {results['baseline'][a]['auc']:.3f})", flush=True)

    results["stacking_oof"] = {}
    for a in anchors:
        best_l = results["probe"][a + "_best_layer"]
        li = layer_ids.index(best_l)
        m = anchor == a
        probe_oof = oof_scores(X[m, li, :].astype(np.float32), y[m], task[m],
                               args.n_splits, args.C)
        stacked = np.concatenate([probe_oof[:, None], surf_feats(m)], axis=1)
        s = oof_scores(stacked, y[m], task[m], args.n_splits, C=1.0)
        auc = roc_auc_score(y[m], s)
        wacc, _, _ = within_task_pair_acc(s, y[m], task[m], rk[m])
        results["stacking_oof"][a] = {"auc": float(auc),
                                      "within_task_acc": float(wacc),
                                      "layer": int(best_l)}
        print(f"[stacking {a} L{best_l}] AUC={auc:.3f} within-task={wacc:.3f} "
              f"(baseline {results['baseline'][a]['auc']:.3f})", flush=True)

    results["mlp_probe"] = {}
    for a in anchors:
        best_l = results["probe"][a + "_best_layer"]
        li = layer_ids.index(best_l)
        m = anchor == a
        feat = X[m, li, :].astype(np.float32)
        aucs, waccs, seed_scores = [], [], []
        for seed in range(3):
            s = oof_scores_mlp(feat, y[m], task[m], args.n_splits, seed=seed)
            seed_scores.append(s)
            aucs.append(roc_auc_score(y[m], s))
            waccs.append(within_task_pair_acc(s, y[m], task[m], rk[m])[0])
        s_mean = np.mean(seed_scores, axis=0)
        lo, hi = paired_bootstrap_delta(s_mean, baseline_scores[a],
                                        y[m], task[m], rk[m])
        results["mlp_probe"][a] = {
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "within_task_mean": float(np.mean(waccs)),
            "within_task_std": float(np.std(waccs)),
            "delta_vs_baseline_ci95": [float(lo), float(hi)],
            "layer": int(best_l)}
        print(f"[MLP {a} L{best_l}] AUC={np.mean(aucs):.3f}+-{np.std(aucs):.3f} "
              f"within-task={np.mean(waccs):.3f}+-{np.std(waccs):.3f} "
              f"delta  CI=[{lo:.3f},{hi:.3f}] "
              f"(baseline {results['baseline'][a]['auc']:.3f})", flush=True)

    if args.mlp_sweep:
        results["mlp_sweep"] = {}
        for a in anchors:
            m = anchor == a
            results["mlp_sweep"][a] = {}
            for li, layer in enumerate(layer_ids):
                s = oof_scores_mlp(X[m, li, :].astype(np.float32), y[m],
                                   task[m], args.n_splits, seed=0)
                auc = roc_auc_score(y[m], s)
                wacc, _, _ = within_task_pair_acc(s, y[m], task[m], rk[m])
                results["mlp_sweep"][a][layer] = {
                    "auc": float(auc), "within_task_acc": float(wacc)}
                print(f"[MLP-sweep {a} L{layer:2d}] AUC={auc:.3f} "
                      f"within-task={wacc:.3f}", flush=True)

    results["nested_stacking"] = {}
    for a in anchors:
        best_l = results["probe"][a + "_best_layer"]
        li = layer_ids.index(best_l)
        m = anchor == a
        Xh = X[m, li, :].astype(np.float32)
        ya, ta, rka = y[m], task[m], rk[m]
        surf = surf_feats(m)
        s = np.zeros(len(ya))
        outer = GroupKFold(n_splits=args.n_splits)
        for tr, te in outer.split(Xh, ya, groups=ta):

            inner_oof = oof_scores(Xh[tr], ya[tr], ta[tr], args.n_splits, args.C)
            meta = fit_lr(np.concatenate([inner_oof[:, None], surf[tr]], 1), ya[tr], 1.0)

            probe_full = fit_lr(Xh[tr], ya[tr], args.C)
            te_probe = probe_full.decision_function(Xh[te])
            s[te] = meta.predict_proba(
                np.concatenate([te_probe[:, None], surf[te]], 1))[:, 1]
        auc = roc_auc_score(ya, s)
        wacc, _, _ = within_task_pair_acc(s, ya, ta, rka)
        print(f"[nested-stack {a} L{best_l}] AUC={auc:.3f} within-task={wacc:.3f} "
              f"(baseline {results['baseline'][a]['auc']:.3f})", flush=True)

        lo, hi = paired_bootstrap_delta(s, baseline_scores[a], ya, ta, rka)
        print(f"[bootstrap {a}] delta within-task 95% CI = [{lo:.3f}, {hi:.3f}]",
              flush=True)
        results["nested_stacking"][a] = {
            "auc": float(auc), "within_task_acc": float(wacc),
            "layer": int(best_l),
            "delta_within_task": float(wacc - results["baseline"][a]["within_task_acc"]),
            "delta_ci95": [float(lo), float(hi)]}

    results["nested_stacking_mlp"] = {}
    for a in anchors:
        best_l = results["probe"][a + "_best_layer"]
        li = layer_ids.index(best_l)
        m = anchor == a
        Xh = X[m, li, :].astype(np.float32)
        ya, ta, rka = y[m], task[m], rk[m]
        surf = surf_feats(m)
        s = np.zeros(len(ya))
        outer = GroupKFold(n_splits=args.n_splits)
        for tr, te in outer.split(Xh, ya, groups=ta):

            inner_oof = oof_scores_mlp(Xh[tr], ya[tr], ta[tr],
                                       args.n_splits, seed=0)
            meta = fit_lr(np.concatenate([inner_oof[:, None], surf[tr]], 1),
                          ya[tr], 1.0)
            te_probe = fit_mlp(Xh[tr], ya[tr], seed=0)(Xh[te])
            s[te] = meta.predict_proba(
                np.concatenate([te_probe[:, None], surf[te]], 1))[:, 1]
        auc = roc_auc_score(ya, s)
        wacc, _, _ = within_task_pair_acc(s, ya, ta, rka)
        lo, hi = paired_bootstrap_delta(s, baseline_scores[a], ya, ta, rka)
        print(f"[nested-stack-MLP {a} L{best_l}] AUC={auc:.3f} "
              f"within-task={wacc:.3f} delta  CI=[{lo:.3f},{hi:.3f}] "
              f"(baseline {results['baseline'][a]['auc']:.3f})", flush=True)
        results["nested_stacking_mlp"][a] = {
            "auc": float(auc), "within_task_acc": float(wacc),
            "layer": int(best_l),
            "delta_within_task": float(wacc - results["baseline"][a]["within_task_acc"]),
            "delta_ci95": [float(lo), float(hi)]}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[probe] wrote {args.out}", flush=True)

if __name__ == "__main__":
    main()
