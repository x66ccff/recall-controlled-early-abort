#!/usr/bin/env python3

import argparse
import glob
import itertools
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = str(ROOT / "artifacts")
DEFAULT_OUTPUT_ROOT = str(ROOT / "outputs" / "standalone")
RESULTS_ROOT = DEFAULT_RESULTS_ROOT

STAGE2_TARGETS = (0.90, 0.95, 0.98, 0.99)

def load_dataset(model, layer, max_round=6):

    feat_dir = os.path.join(RESULTS_ROOT, model, "features")
    if not os.path.isdir(feat_dir):
        feat_dir = os.path.join(RESULTS_ROOT, model, "features_v2")
    npz_files = sorted(
        glob.glob(os.path.join(feat_dir, "feat.shard*.npz")),
        key=lambda p: int(re.search(r"shard(\d+)", p).group(1)))
    assert npz_files, f"no feat.shard*.npz files found under {feat_dir}"

    metas, Xs = [], []
    for f in npz_files:
        sid = re.search(r"shard(\d+)", f).group(1)
        d = np.load(f)
        lids = d["layer_ids"].tolist()
        assert layer in lids, f"layer={layer} is not available in extracted layers: {lids}"
        Xs.append(d["X"][:, lids.index(layer), :].astype(np.float32))
        rows = [json.loads(l) for l in
                open(os.path.join(feat_dir, f"meta.shard{sid}.jsonl"))]
        assert len(rows) == Xs[-1].shape[0], f"shard{sid} npz/meta row-count mismatch"
        metas.extend(rows)
    Xall = np.concatenate(Xs, 0)

    ep_key = {}
    for m in metas:
        k = (m["task_idx"], m["rollout_k"])
        if k not in ep_key:
            ep_key[k] = len(ep_key)
    n_ep = len(ep_key)
    success = np.zeros(n_ep, bool)
    n_rounds = np.zeros(n_ep, int)
    task = np.zeros(n_ep, int)
    for (ti, rk), e in ep_key.items():
        task[e] = ti
    for m in metas:
        e = ep_key[(m["task_idx"], m["rollout_k"])]
        success[e] = bool(m["success"])
        n_rounds[e] = m["n_rounds"]

    pre_ti, post_ti = {}, {}
    for i, m in enumerate(metas):
        e = ep_key[(m["task_idx"], m["rollout_k"])]
        tgt = pre_ti if m["anchor"] == "pre_gen" else post_ti
        tgt[(e, m["round"])] = (m["token_idx"], i)

    R_full = int(n_rounds.max())
    tokens = np.zeros((n_ep, R_full), dtype=np.float64)
    for e in range(n_ep):
        nr = n_rounds[e]
        for t in range(nr):
            start = pre_ti[(e, t)][0]
            end = (pre_ti[(e, t + 1)][0] if t + 1 < nr
                   else post_ti[(e, nr - 1)][0])
            tokens[e, t] = end - start

    H, S, EP = {}, {}, {}
    R = min(max_round, R_full)
    buckets = {r: [] for r in range(1, R + 1)}
    for i, m in enumerate(metas):
        if m["anchor"] != "post_gen":
            continue
        r = m["round"] + 1
        if r > R:
            continue
        e = ep_key[(m["task_idx"], m["rollout_k"])]
        n_gen = m["token_idx"] - pre_ti[(e, m["round"])][0]
        buckets[r].append(
            (i, e, [m["cur_round_lp"], n_gen, m["prefix_len"], m["n_err"]]))
    for r, items in buckets.items():
        if not items:
            continue
        H[r] = Xall[np.array([i for i, _, _ in items])]
        EP[r] = np.array([e for _, e, _ in items])
        S[r] = np.array([s for _, _, s in items], dtype=np.float32)

    print(f"[load] {model} layer={layer}: {n_ep} eps / {len(np.unique(task))} tasks "
          f"(succ={success.mean():.2%}), R_full={R_full}, "
          f"alive per round: {[len(EP.get(r, [])) for r in range(1, R + 1)]}")
    return dict(success=success, tokens=tokens, task=task, H=H, S=S, ep=EP)

def recall_lower_bound(kept_pos, n_pos, alpha=0.05):

    if n_pos == 0 or kept_pos <= 0:
        return 0.0
    if kept_pos >= n_pos:
        return alpha ** (1.0 / n_pos)
    return float(beta.ppf(alpha, kept_pos, n_pos - kept_pos + 1))

def calibrate_threshold(scores_pos, target, method="cp", alpha=0.05):

    s = np.sort(np.asarray(scores_pos, dtype=float))
    n = len(s)
    if n == 0 or target >= 1.0:
        return -np.inf
    if method == "quantile":
        return float(np.quantile(s, 1 - target))
    thr = -np.inf
    for i in range(n):

        if recall_lower_bound(n - i, n, alpha) >= target:
            thr = s[i]
        else:
            break
    return thr

def _probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))

def _level2():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))

def _constant_score(y_train, n_test, *, positive_score=True):

    if len(y_train) == 0:
        value = 0.0
    else:
        cls = int(np.asarray(y_train).astype(int)[0])
        value = 1.0 if cls == 1 else 0.0
        if not positive_score:
            value = 1.0 if cls == 1 else -1.0
    return np.full(n_test, value, dtype=float)

def _grouped_folds(y, groups, k, seed):

    k = max(2, min(k, int(np.bincount(y.astype(int)).min()),
                   len(np.unique(groups))))
    return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)

def nested_stack_oof(H, S, y, groups, seed, n_outer=5, n_inner=5):

    y = y.astype(int)
    scores = np.full(len(y), np.nan)
    ko = _grouped_folds(y, groups, n_outer, seed)
    for tr, te in ko.split(H, y, groups):
        if len(np.unique(y[tr])) < 2:
            scores[te] = _constant_score(y[tr], len(te), positive_score=True)
            continue
        inner = np.full(len(tr), np.nan)
        ki = _grouped_folds(y[tr], groups[tr], n_inner, seed + 100)
        for itr, ite in ki.split(H[tr], y[tr], groups[tr]):
            if len(np.unique(y[tr][itr])) < 2:
                inner[ite] = _constant_score(y[tr][itr], len(ite), positive_score=False)
            else:
                m = _probe().fit(H[tr][itr], y[tr][itr])
                inner[ite] = m.decision_function(H[tr][ite])
        if len(np.unique(y[tr])) < 2:
            scores[te] = _constant_score(y[tr], len(te), positive_score=True)
            continue
        l2 = _level2().fit(np.column_stack([inner, S[tr]]), y[tr])
        m_full = _probe().fit(H[tr], y[tr])
        Z_te = np.column_stack([m_full.decision_function(H[te]), S[te]])
        scores[te] = l2.predict_proba(Z_te)[:, 1]
    return scores

def probe_only_oof(H, S, y, groups, seed, n_outer=5):

    y = y.astype(int)
    scores = np.full(len(y), np.nan)
    ko = _grouped_folds(y, groups, n_outer, seed)
    for tr, te in ko.split(H, y, groups):
        if len(np.unique(y[tr])) < 2:
            scores[te] = _constant_score(y[tr], len(te), positive_score=False)
        else:
            m = _probe().fit(H[tr], y[tr])
            scores[te] = m.decision_function(H[te])
    return scores

def surface_only_oof(H, S, y, groups, seed, n_outer=5):

    y = y.astype(int)
    scores = np.full(len(y), np.nan)
    ko = _grouped_folds(y, groups, n_outer, seed)
    for tr, te in ko.split(S, y, groups):
        if len(np.unique(y[tr])) < 2:
            scores[te] = _constant_score(y[tr], len(te), positive_score=False)
        else:
            m = _probe().fit(S[tr], y[tr])
            scores[te] = m.decision_function(S[te])
    return scores

def mlp_oof(H, S, y, groups, seed, n_outer=5):

    from sklearn.neural_network import MLPClassifier
    def _mlp():
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(256,), max_iter=500, random_state=seed))
    y = y.astype(int)
    scores = np.full(len(y), np.nan)
    ko = _grouped_folds(y, groups, n_outer, seed)
    for tr, te in ko.split(H, y, groups):
        m = _mlp().fit(H[tr], y[tr])
        scores[te] = m.predict_proba(H[te])[:, 1]
    return scores

SCORERS = {"stacking": nested_stack_oof, "probe": probe_only_oof,
           "surface": surface_only_oof, "probe_mlp": mlp_oof}

def eval_round(scores, y, suffix_tok, total_tok, cal_mask, targets,
               cal_method, alpha):
    out = {}
    cal_succ = scores[cal_mask & (y == 1)]
    te = ~cal_mask
    for t in targets:
        thr = calibrate_threshold(cal_succ, t, cal_method, alpha)
        prune = te & (scores < thr)
        keep_succ = ((y == 1) & te & ~prune).sum()
        recall_alive = keep_succ / max(1, ((y == 1) & te).sum())
        saved = suffix_tok[prune].sum() / max(1e-9, total_tok)
        cut_fail = (prune[(y == 0) & te].sum() / max(1, ((y == 0) & te).sum())
                    if ((y == 0) & te).sum() else 0.0)
        out[t] = dict(recall_alive=recall_alive, saved=saved, cut_fail_rate=cut_fail,
                      n_pruned=int(prune.sum()),
                      pruned_succ=int(((y == 1) & te & prune).sum()))
    return out

def run_stage2(data, scorer, seeds, targets, cal_method, alpha,
               cal_frac=0.4, max_round=6):
    success, tokens, task = data["success"], data["tokens"], data["task"]
    results = {}
    for r in range(1, max_round + 1):
        if r not in data["H"]:
            break
        H, S, ep = data["H"][r], data["S"][r], data["ep"][r]
        y = success[ep].astype(int)
        g = task[ep]
        suffix = tokens[:, r - 1:].sum(1)[ep]
        per_seed = {t: [] for t in targets}
        aucs, recg = [], {t: [] for t in targets}
        n_succ_total = int(success.sum())
        for seed in seeds:
            sc = scorer(H, S, y, g, seed)
            aucs.append(roc_auc_score(y, sc))

            rng = np.random.default_rng(seed)
            uniq = np.unique(g)
            rng.shuffle(uniq)
            cal = np.isin(g, uniq[:int(cal_frac * len(uniq))])
            total_tok = tokens[ep[~cal]].sum()
            m = eval_round(sc, y, suffix, total_tok, cal, targets, cal_method, alpha)
            for t in targets:
                per_seed[t].append(m[t]["saved"])
                te_frac = (~cal).mean()
                lost_global = m[t]["pruned_succ"] / max(1e-9, te_frac)
                recg[t].append(1 - lost_global / n_succ_total)
        results[r] = dict(
            auc=(float(np.mean(aucs)), float(np.std(aucs))),
            table={t: dict(saved_mu=float(np.mean(per_seed[t])),
                           saved_sd=float(np.std(per_seed[t])),
                           recall_global_mu=float(np.mean(recg[t])),
                           recall_global_sd=float(np.std(recg[t])))
                   for t in targets})
        a = results[r]["auc"]
        print(f"\n===== r{r} (AUC={a[0]:.3f}+-{a[1]:.3f}, alive={len(y)}) =====")
        for t in targets:
            v = results[r]["table"][t]
            print(f"  target={t:.2f}  recall_global={v['recall_global_mu']:.3f}"
                  f"+-{v['recall_global_sd']:.3f}"
                  f"  saved={v['saved_mu']*100:.1f}%+-{v['saved_sd']*100:.1f}%")
    return results

def run_cascade(data, scorer, seeds, target_recall, cal_method, alpha,
                max_round=6, grid=(1.0, 0.99, 0.98, 0.95, 0.90, 0.85),
                cal_frac=0.4,
                margin_mode="none", margin_delta=0.02, margin_alpha=0.05):

    success, tokens, task = data["success"], data["tokens"], data["task"]
    n_ep, R = tokens.shape[0], min(max_round, max(data["H"].keys()))
    suffix = np.column_stack([tokens[:, r:].sum(1) for r in range(R)])
    agg = {}
    for seed in seeds:

        score_mat = np.full((n_ep, R), np.nan)
        for r in range(1, R + 1):
            H, S, ep = data["H"][r], data["S"][r], data["ep"][r]
            score_mat[ep, r - 1] = scorer(H, S, success[ep].astype(int),
                                          task[ep], seed)
        alive = ~np.isnan(score_mat)

        rng = np.random.default_rng(seed)
        uniq = np.unique(task)
        rng.shuffle(uniq)
        n_cal = int(cal_frac * len(uniq))
        cal_tasks = uniq[:n_cal]
        cal = np.isin(task, cal_tasks)
        calA = np.isin(task, cal_tasks[:n_cal // 2])
        calB = cal & ~calA

        def thr_at(mask, r, t):
            s = score_mat[mask & success & alive[:, r], r]
            return calibrate_threshold(s, t, cal_method, alpha)

        thr_cand = {r: {t: thr_at(calB, r, t) for t in grid} for r in range(R)}

        def make_sim(mask):
            a, sm = alive[mask], score_mat[mask]
            sc, sf, tk = success[mask], suffix[mask], tokens[mask].sum()
            idx = np.arange(mask.sum())
            n_pos = int(sc.sum())
            def sim(tau):
                hit = a & (sm < tau[None, :])
                any_hit = hit.any(1)
                first = np.argmax(hit, 1)
                lost = int((any_hit & sc).sum())
                recall = 1 - lost / max(1, n_pos)
                saved = sf[idx[any_hit], first[any_hit]].sum() / max(1e-9, tk)
                return recall, saved, n_pos - lost
            return sim, n_pos

        (sim_A, n_posA) = make_sim(calA)
        (sim_test, _) = make_sim(~cal)

        if margin_mode == "none":
            def feasible(rc, kept):
                return rc >= target_recall
        elif margin_mode == "delta":
            req = min(target_recall + margin_delta, 1.0)
            def feasible(rc, kept):
                return rc >= req
        elif margin_mode == "cp":

            kept_min = n_posA + 1
            for kk in range(n_posA, -1, -1):
                if recall_lower_bound(kk, n_posA, margin_alpha) >= target_recall:
                    kept_min = kk
                else:
                    break
            if kept_min > n_posA and seed == seeds[0]:
                print(f"  [warn] CP margin: calA n_pos={n_posA} is insufficient "
                      f"to certify target={target_recall} at alpha={margin_alpha}; "
                      f"searched_best falls back to no pruning (saved=0)")
            def feasible(rc, kept):
                return kept >= kept_min
        else:
            raise ValueError(f"unknown margin_mode: {margin_mode}")

        best = None
        for combo in itertools.product(grid, repeat=R):
            tau = np.array([thr_cand[r][combo[r]] for r in range(R)])
            rc, sv, kept = sim_A(tau)
            if feasible(rc, kept) and (best is None or sv > best[1]):
                best = (combo, sv, tau, rc)
        strategies = {"searched_best": best[2] if best else np.full(R, -np.inf)}

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
        strategies["best_single_gate"] = (
            best_single[2] if best_single else np.full(R, -np.inf)
        )
        per_round_t = target_recall ** (1.0 / R)
        strategies["uniform"] = np.array(
            [thr_at(cal, r, per_round_t) for r in range(R)])

        for name, tau in strategies.items():
            rc, sv, _ = sim_test(tau)
            agg.setdefault(name, {"recall": [], "saved": [],
                                  "combo": [], "recall_valA": []})
            agg[name]["recall"].append(rc)
            agg[name]["saved"].append(sv)
            if name == "searched_best" and best:
                agg[name]["combo"].append(best[0])
                agg[name]["recall_valA"].append(best[3])
            elif name == "best_single_gate" and best_single:
                agg[name].setdefault("selected_round", []).append(best_single[0] + 1)
                agg[name]["combo"].append((best_single[4],))
                agg[name]["recall_valA"].append(best_single[3])

    print(f"\n===== cascade simulation target_global_recall={target_recall} "
          f"({len(seeds)} seeds, test split, cal_method={cal_method}, "
          f"margin={margin_mode}"
          + (f", delta={margin_delta}" if margin_mode == "delta" else "")
          + (f", m_alpha={margin_alpha}" if margin_mode == "cp" else "")
          + ") =====")
    for name, v in agg.items():
        line = (f"  {name:14s} recall={np.mean(v['recall']):.3f}"
                f"+-{np.std(v['recall']):.3f}"
                f"  saved={np.mean(v['saved'])*100:.1f}%"
                f"+-{np.std(v['saved'])*100:.1f}%")
        if v["recall_valA"]:
            line += f"  [valA recall={np.mean(v['recall_valA']):.3f}]"
        print(line)
    if agg.get("searched_best", {}).get("combo"):
        from collections import Counter
        print("  modal searched per-round recall allocation:",
              Counter(agg["searched_best"]["combo"]).most_common(1)[0][0])
    return agg

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin-mode", choices=["none", "delta", "cp"],
                    default="none", help="cascade-search margin")
    ap.add_argument("--margin-delta", type=float, default=0.02)
    ap.add_argument("--margin-alpha", type=float, default=0.05,
                    help="confidence level for the calA recall lower bound when margin-mode=cp")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--mode", choices=["stage2", "cascade"], required=True)
    ap.add_argument("--scorer", choices=["stacking", "probe", "surface", "probe_mlp"],
                    default="stacking")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--cal-method", choices=["cp", "quantile"], default="cp")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="one-sided CP confidence level; pass 0.05/R for a strict per-round union allocation")
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT,
                    help="artifact root containing features/rollouts")
    ap.add_argument("--output-results-root", default=DEFAULT_OUTPUT_ROOT,
                    help="output directory for JSON files")
    args = ap.parse_args()

    RESULTS_ROOT = args.results_root
    output_results_root = args.output_results_root
    data = load_dataset(args.model, args.layer)
    scorer = SCORERS[args.scorer]
    seeds = list(range(args.seeds))

    if args.mode == "stage2":
        res = run_stage2(data, scorer, seeds, list(STAGE2_TARGETS),
                         args.cal_method, args.alpha)
        name = (f"stage2_{args.scorer}_L{args.layer}"
                f"_{args.cal_method}_s{args.seeds}.json")
    else:
        res = run_cascade(data, scorer, seeds, args.target_recall,
                          args.cal_method, args.alpha,
                          margin_mode=args.margin_mode,
                          margin_delta=args.margin_delta,
                          margin_alpha=args.margin_alpha)
        res = {k: {"recall": list(map(float, v["recall"])),
                   "saved": list(map(float, v["saved"])),
                   "recall_valA": list(map(float, v["recall_valA"])),
                   "combo": [list(c) for c in v["combo"]],
                   **({"selected_round": v["selected_round"]}
                      if "selected_round" in v else {})}
               for k, v in res.items()}

        mtag = ""
        if args.margin_mode == "delta":
            dtag = f"{args.margin_delta:g}"
            if not re.fullmatch(r"[\d.]+", dtag):
                raise ValueError(
                    f"margin_delta={args.margin_delta} formats as {dtag}, "
                    f"which is not matched by the summary filename pattern; use a value >=1e-4 or extend the pattern")
            mtag = f"_delta{dtag}"
        elif args.margin_mode == "cp":
            if f"{args.margin_alpha:g}" != "0.05":
                raise ValueError(
                    f"margin_alpha={args.margin_alpha} != 0.05: the current filename schema does not encode alpha, "
                    f"so non-default values would collide with default runs; extend the pattern before sweeping alpha")
            mtag = "_margin-cp"
        name = (f"cascade_{args.scorer}_L{args.layer}"
                f"_r{args.target_recall:g}_{args.cal_method}"
                f"{mtag}_s{args.seeds}.json")

        _pat = re.compile(
            rf"^cascade_(?P<head>.+?)_L{args.layer}_r(?P<target>[\d.]+)_(?P<cal>cp|quantile)"
            rf"(?:_margin-(?P<margin>delta\d*|cp))?"
            rf"(?:_delta(?P<delta>[\d.]+))?"
            rf"(?:_s(?P<seeds>\d+))?\.json$")
        _m = _pat.match(name)
        if not _m or _m["head"] != args.scorer or int(_m["seeds"]) != args.seeds:
            raise RuntimeError(f"filename {name} failed the summary-pattern self-check; refusing to write")

        assert len(res["searched_best"]["recall"]) == args.seeds,\
            (f"searched_best has {len(res['searched_best']['recall'])} seeds "
             f"!= declared {args.seeds}")

    out = os.path.join(output_results_root, args.model, name)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    res["_config"] = vars(args)

    if os.path.exists(out):
        import time
        bak = out + time.strftime(".overwritten_%Y%m%d_%H%M%S")
        os.rename(out, bak)
        print(f"[warn] output exists; moved previous file to {bak}")

    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nwrote {out}")
