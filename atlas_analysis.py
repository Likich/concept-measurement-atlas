"""Measurement-atlas pilot analysis.

For model m, rating i and indicator j:  M[m, i, j] = mean over base pairs of (score_high - score_low), paired within
base_id. Bootstrap resamples base pairs (with replacement) within each indicator.

Confirmatory pilot metrics (pre-specified in the task and the registry):
  own-indicator sensitivity; construct specificity; theory alignment with the frozen expected matrix (Spearman,
  sign agreement, top-rating accuracy, results by expected class, equivalence at +-0.15/0.25/0.35);
  effective dimensionality (raw, row- and column-standardized); null comparisons.
Exploratory: confusion graph, cross-model agreement, Qwen-only scaling, prompt robustness, valence check,
representation analysis.

Usage:  python atlas_analysis.py [--root results_atlas/pilot_v1] [--boot 2000] [--representation]
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import atlas_vignettes as AV

DELTAS = (0.15, 0.25, 0.35)
SIZES = {"Qwen2.5-1.5B-Instruct": 1.5, "Qwen2.5-3B-Instruct": 3.1, "Qwen2.5-7B-Instruct": 7.6,
         "Llama-3.1-8B-Instruct": 8.0, "OLMo-2-1124-13B-Instruct": 13.7, "Qwen2.5-32B-Instruct": 32.5}
FAMILY = {"Qwen2.5-1.5B-Instruct": "Qwen", "Qwen2.5-3B-Instruct": "Qwen", "Qwen2.5-7B-Instruct": "Qwen",
          "Llama-3.1-8B-Instruct": "Llama", "OLMo-2-1124-13B-Instruct": "OLMo", "Qwen2.5-32B-Instruct": "Qwen"}


# ------------------------------------------------------------------ effect matrices

def pair_diffs(scores, stim, cond, ratings, cols, bases=None):
    """D[rating, indicator, base] = score(high) - score(low); NaN where a base pair is not in the condition."""
    s = scores[scores.condition == cond].merge(stim[["id", "base_id", "col", "pole"]], on="id")
    if bases is not None:
        s = s[s.base_id.isin(bases)]
    w = s.pivot_table(index=["rating", "col", "base_id"], columns="pole", values="score")
    d = (w["high"] - w["low"]).rename("d").reset_index()
    d["b"] = d.base_id.str.extract(r"\.b(\d+)$")[0].astype(int)
    D = np.full((len(ratings), len(cols), 6), np.nan)
    ri, ci = {r: k for k, r in enumerate(ratings)}, {c: k for k, c in enumerate(cols)}
    D[d.rating.map(ri).values, d.col.map(ci).values, d.b.values] = d.d.values
    return D


def effect_matrix(D):
    return np.nanmean(D, axis=2)


def boot_indices(D, B, seed=0):
    """Per indicator, resample its available base pairs with replacement -> idx[B, n_indicators, 6] (base indices)."""
    rng = np.random.default_rng(seed)
    avail = [np.where(~np.isnan(D[0, j]))[0] for j in range(D.shape[1])]
    idx = np.full((B, D.shape[1], 6), -1, dtype=int)   # -1 = unused slot (indicators with < 6 pairs in a condition)
    for j, a in enumerate(avail):
        idx[:, j, :len(a)] = rng.choice(a, size=(B, len(a)), replace=True)
    return idx


def boot_matrices(D, idx):
    """[B, ratings, indicators] bootstrap effect matrices from resampled base indices (-1 = unused slot)."""
    B, J, K = idx.shape
    out = np.empty((B, D.shape[0], J))
    for j in range(J):
        ij = idx[:, j]
        valid = ij >= 0
        vals = D[:, j, :][:, np.where(valid, ij, 0)]          # [R, B, K]
        vals = np.where(valid[None], vals, np.nan)
        out[:, :, j] = np.nanmean(vals, axis=2).T
    return out


# ------------------------------------------------------------------ metrics

def own_rows(E):
    """row index of the own construct for each indicator column (the +2 cell)."""
    return np.array([int(np.where(E[:, j] == 2)[0][0]) for j in range(E.shape[1])])


def metrics(M, E):
    """Point metrics for one effect matrix M against the signed expected matrix E (NaN = NA)."""
    R, J = M.shape
    own = own_rows(E)
    eo = M[own, np.arange(J)]
    zero = (E == 0)
    absM = np.abs(M)
    margin = eo - np.array([absM[zero[:, j], j].max() if zero[:, j].any() else 0 for j in range(J)])
    ratio = np.abs(eo) / absM.sum(0)
    valid_theory = (~np.isnan(E)) & (E != 0) & (np.sign(M) == np.sign(np.nan_to_num(E)))
    theory_ratio = (absM * valid_theory).sum(0) / absM.sum(0)
    known = ~np.isnan(E)
    rho = spearmanr(E[known], M[known]).correlation
    nz = known & (E != 0)
    order = np.argsort(-absM, axis=0)
    top1 = np.mean(order[0] == own)
    top3 = np.mean([own[j] in order[:3, j] for j in range(J)])
    out = dict(own_sensitivity=float(eo.mean()), specificity_margin=float(margin.mean()),
               specificity_ratio=float(ratio.mean()), theory_ratio=float(theory_ratio.mean()),
               theory_spearman=float(rho), sign_agreement=float(np.mean(np.sign(M[nz]) == np.sign(E[nz]))),
               top1_accuracy=float(top1), top3_accuracy=float(top3))
    for v, name in [(2, "cells_2"), (1, "cells_1")]:
        m = known & (np.abs(E) == v)
        out[f"{name}_mean_signed"] = float((M[m] * np.sign(E[m])).mean())
    out["cells_0_mean_abs"] = float(absM[zero].mean())
    out["per_indicator"] = dict(own=eo.tolist(), margin=margin.tolist(), ratio=ratio.tolist())
    return out


def normalize(M, kind):
    if kind == "raw":
        return M
    if kind == "row":
        return M / (np.sqrt((M ** 2).mean(1, keepdims=True)) + 1e-12)
    if kind == "col":
        return M / (np.sqrt((M ** 2).mean(0, keepdims=True)) + 1e-12)
    raise ValueError(kind)


def dimensionality(M):
    s = np.linalg.svd(M, compute_uv=False)
    v = s ** 2 / (s ** 2).sum()
    p = s / s.sum()
    p = p[p > 0]
    return dict(singular_values=s.tolist(), var_1=float(v[:1].sum()), var_2=float(v[:2].sum()),
                var_3=float(v[:3].sum()), var_5=float(v[:5].sum()), stable_rank=float((s ** 2).sum() / s[0] ** 2),
                entropy_rank=float(np.exp(-(p * np.log(p)).sum())))


def ci(a):
    a = np.asarray(a, dtype=float)
    return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


def ci_rc(a, point):
    """Percentile interval re-centred on the point estimate. Resampling base pairs adds noise, which attenuates
    correlations and inflates ranks; re-centring removes that bootstrap bias (both kinds of interval are reported)."""
    lo, hi = ci(a)
    shift = point - float(np.nanmedian(a))
    return [lo + shift, hi + shift]


def remove_pc1(M):
    """Post hoc: M minus its best rank-1 approximation."""
    u, sv, vt = np.linalg.svd(M, full_matrices=False)
    return M - sv[0] * np.outer(u[:, 0], vt[0])


def remove_valence(M, val):
    """Post hoc: residual of each rating row after regressing it on [1, registry valence_shift] across indicators."""
    X = np.column_stack([np.ones_like(val, dtype=float), val])
    beta = np.linalg.lstsq(X, M.T, rcond=None)[0]
    return M - (X @ beta).T


# ------------------------------------------------------------------ nulls

def permute_within_family(E, fam, rng):
    """Shuffle indicator-to-construct assignments within family: permute columns of E among same-family indicators."""
    P = E.copy()
    for f in np.unique(fam):
        cols = np.where(fam == f)[0]
        P[:, cols] = E[:, rng.permutation(cols)]
    return P


def permute_all(E, rng):
    return E[:, rng.permutation(E.shape[1])]


def permute_rows(M, rng):
    """Preserve each rating's effect magnitudes, permute their locations across indicators."""
    return np.array([rng.permutation(r) for r in M])


def permute_cols(M, rng):
    return np.array([rng.permutation(c) for c in M.T]).T


def alignment_null(M, E, fam, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    obs = metrics(M, E)
    keys = ["theory_spearman", "sign_agreement", "top1_accuracy", "top3_accuracy", "specificity_margin"]
    out = {}
    for name, fn in [("within_family", lambda: permute_within_family(E, fam, rng)), ("across_all", lambda: permute_all(E, rng))]:
        null = {k: [] for k in keys}
        for _ in range(n):
            m = metrics(M, fn())
            for k in keys:
                null[k].append(m[k])
        out[name] = {k: dict(observed=obs[k], null_mean=float(np.mean(v)), null_95=float(np.percentile(v, 95)),
                             p=float((np.sum(np.array(v) >= obs[k]) + 1) / (n + 1))) for k, v in null.items()}
    return out


def rank_null(M, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    obs = dimensionality(M)
    out = {}
    for name, fn in [("rows_permuted", permute_rows), ("cols_permuted", permute_cols)]:
        dims = [dimensionality(fn(M, rng)) for _ in range(n)]
        er, v1 = [d["entropy_rank"] for d in dims], [d["var_1"] for d in dims]
        out[name] = dict(observed_entropy_rank=obs["entropy_rank"], null_mean=float(np.mean(er)),
                         null_5=float(np.percentile(er, 5)),
                         p_lower=float((np.sum(np.array(er) <= obs["entropy_rank"]) + 1) / (n + 1)),
                         observed_var_1=obs["var_1"], null_var_1_mean=float(np.mean(v1)),
                         p_var1_higher=float((np.sum(np.array(v1) >= obs["var_1"]) + 1) / (n + 1)))
    return out


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    cum = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        cum = min(cum, p[i] * len(p) / rank)
        adj[i] = cum
    return adj


def boot_p(draws):
    """two-sided bootstrap p-value for 'effect differs from 0'."""
    d = np.asarray(draws)
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / (len(d) + 1)))


# ------------------------------------------------------------------ representation

def domain_folds(index):
    """Leave-one-domain-out folds over base pairs: [(held_out_domain, train_base_ids, test_base_ids)]."""
    pairs = index.drop_duplicates("base_id")
    folds = []
    for d in sorted(pairs.domain.unique()):
        folds.append((d, set(pairs[pairs.domain != d].base_id), set(pairs[pairs.domain == d].base_id)))
    return folds


def pair_contrasts(H, index):
    """per base pair: h_high - h_low  -> dict base_id -> [layers, d]"""
    pos = {i: k for k, i in enumerate(index.id)}
    out = {}
    for b, g in index.groupby("base_id"):
        hi = pos[g[g.pole == "high"].id.iloc[0]]
        lo = pos[g[g.pole == "low"].id.iloc[0]]
        out[b] = H[hi] - H[lo]
    return out


def representation_analysis(H, index, cols, rating_M=None):
    """Paired decoder: direction = mean training-pair contrast of an indicator; a held-out pair is decoded correctly
    when its contrast projects positively. Leave-one-domain-out; the layer is chosen by inner leave-one-domain-out
    on the training domains only, then evaluated on the held-out domain (no selection on test folds)."""
    C = pair_contrasts(H, index)
    col_of = dict(zip(index.base_id, index.col))
    folds = domain_folds(index)
    L = H.shape[1]

    def acc_layer(train, test, col):
        tr = [C[b] for b in train if col_of[b] == col]
        te = [C[b] for b in test if col_of[b] == col]
        if not tr or not te:
            return None
        w = np.mean(tr, axis=0)                                  # [L, d]
        return np.mean([(np.einsum("ld,ld->l", c, w) > 0) for c in te], axis=0)  # [L]

    res = {}
    for col in cols:
        outer = []
        for d, train, test in folds:
            # inner leave-one-domain-out on the training domains only, to choose the layer
            inner = [acc_layer(train - test2, test2, col) for d2, _, test2 in folds if d2 != d]
            inner = [x for x in inner if x is not None]
            if not inner:
                continue
            best = int(np.argmax(np.mean(inner, axis=0)))
            o = acc_layer(train, test, col)
            if o is not None:
                n_te = sum(1 for b in test if col_of[b] == col)
                outer.append((o[best], n_te))
        res[col] = float(sum(a * n for a, n in outer) / sum(n for _, n in outer)) if outer else float("nan")
    # dimensionality of indicator contrasts per layer (all pairs; descriptive, no selection)
    con = np.stack([np.mean([C[b] for b in C if col_of[b] == col], axis=0) for col in cols])   # [36, L, d]
    erank = [dimensionality(con[:, l, :])["entropy_rank"] for l in range(1, L)]
    out = dict(decoding_accuracy=res, mean_accuracy=float(np.nanmean(list(res.values()))),
               n_decodable_5of6=int(sum(v >= 5 / 6 - 1e-9 for v in res.values())),
               contrast_entropy_rank_by_layer=erank, contrast_entropy_rank_median=float(np.median(erank)),
               contrast_entropy_rank_max_possible=len(cols))
    if rating_M is not None:
        out["rating_entropy_rank"] = dimensionality(rating_M)["entropy_rank"]
        out["rating_entropy_rank_max_possible"] = min(rating_M.shape)
    return out


# ------------------------------------------------------------------ main

def valence_vector(reg, cols):
    return np.array([reg["constructs"][c.split(".")[0]]["indicators"][c.split(".")[1]]["valence_shift"] for c in cols],
                    dtype=float)


def analyse_model(mdir, stim, reg, E, cols, ratings, fam, B, n_null):
    scores = pd.read_csv(os.path.join(mdir, "scores.csv"))
    res = {"model": os.path.basename(mdir), "conditions": {}}
    mats, draws = {}, {}
    for cond in sorted(scores.condition.unique()):
        D = pair_diffs(scores, stim, cond, ratings, cols)
        M = effect_matrix(D)
        idx = boot_indices(D, B)
        Mb = boot_matrices(D, idx)
        mats[cond], draws[cond] = M, (Mb, idx)
        r = {"point": metrics(M, E)}
        bm = [metrics(m, E) for m in Mb]
        keys = [k for k in r["point"] if k != "per_indicator"]
        r["ci_percentile"] = {k: ci([x[k] for x in bm]) for k in keys}
        r["ci"] = {k: ci_rc([x[k] for x in bm], r["point"][k]) for k in keys}
        r["dimensionality"] = {}
        for kind in ["raw", "row", "col"]:
            pt = dimensionality(normalize(M, kind))
            bd = [dimensionality(normalize(m, kind)) for m in Mb]
            r["dimensionality"][kind] = {"point": {k: v for k, v in pt.items()},
                                         "ci": {k: ci_rc([x[k] for x in bd], pt[k]) for k in pt if k != "singular_values"},
                                         "ci_percentile": {k: ci([x[k] for x in bd]) for k in pt if k != "singular_values"}}
        # equivalence of expected-0 cells and FDR-adjusted leaks
        zero = (E == 0)
        lo, hi = np.percentile(Mb, 2.5, axis=0), np.percentile(Mb, 97.5, axis=0)
        r["equivalence_zero_cells"] = {str(d): float(np.mean((lo[zero] > -d) & (hi[zero] < d))) for d in DELTAS}
        pz = np.array([boot_p(Mb[:, i, j]) for i, j in zip(*np.where(zero))])
        q = bh(pz)
        r["zero_cells_nonzero_fdr05"] = float(np.mean(q < .05))
        r["cells_2_ci_correct"] = float(np.mean([(lo[i, j] > 0) for i, j in zip(*np.where(E == 2))]))
        r["cells_1_ci_correct"] = float(np.mean([((lo[i, j] > 0) if E[i, j] > 0 else (hi[i, j] < 0))
                                                 for i, j in zip(*np.where(np.abs(np.nan_to_num(E)) == 1))]))
        if cond != "reversed":
            r["null_alignment"] = alignment_null(M, E, fam, n=n_null)
            r["null_rank"] = rank_null(M, n=n_null)
        # confusion edges: expected-0 cells with |effect| >= 0.5 and FDR q < .05
        edges = []
        for (i, j), qq in zip(zip(*np.where(zero)), q):
            if abs(M[i, j]) >= .5 and qq < .05:
                edges.append(dict(indicator=cols[j], from_construct=cols[j].split(".")[0], rating=ratings[i],
                                  effect=float(M[i, j]), ci=[float(lo[i, j]), float(hi[i, j])], q=float(qq)))
        r["confusions"] = sorted(edges, key=lambda e: -abs(e["effect"]))
        # valence check: loading of each indicator on the first right singular vector vs registry valence_shift
        u, s, vt = np.linalg.svd(M, full_matrices=False)
        val = valence_vector(reg, cols)
        r["valence_check"] = dict(spearman_pc1_vs_valence_shift=float(abs(spearmanr(vt[0], val).correlation)),
                                  pc1_top_ratings=[ratings[k] for k in np.argsort(-np.abs(u[:, 0]))[:4]])
        # POST HOC (added after inspecting the first two models): is theory alignment left once the dominant
        # axis is removed? Two removals: the first singular component, and a per-rating regression on valence_shift.
        if cond != "reversed":
            keys2 = ["theory_spearman", "sign_agreement", "top1_accuracy", "top3_accuracy", "specificity_margin"]
            post = {}
            for name, fn in [("pc1_removed", remove_pc1), ("valence_regressed", lambda X: remove_valence(X, val))]:
                Rm = fn(M)
                pt = metrics(Rm, E)
                bm2 = [metrics(fn(m), E) for m in Mb]
                post[name] = dict(point={k: pt[k] for k in keys2},
                                  ci={k: ci_rc([x[k] for x in bm2], pt[k]) for k in keys2},
                                  null_within_family=alignment_null(Rm, E, fam, n=n_null)["within_family"],
                                  residual_var_share=float((Rm ** 2).sum() / (M ** 2).sum()))
            r["post_hoc_residual_alignment"] = post
        res["conditions"][cond] = r
        pd.DataFrame(M, index=ratings, columns=cols).to_csv(os.path.join(mdir, f"effects_{cond}.csv"))
        np.savez_compressed(os.path.join(mdir, f"boot_{cond}.npz"), Mb=Mb.astype(np.float32))
    # prompt robustness: definition - simple, paired on identical bootstrap indices
    if "simple" in mats and "definition" in mats:
        Ds = pair_diffs(scores, stim, "simple", ratings, cols)
        Dd = pair_diffs(scores, stim, "definition", ratings, cols)
        idx = boot_indices(Ds, B, seed=1)
        Ms, Md = boot_matrices(Ds, idx), boot_matrices(Dd, idx)
        keys = ["own_sensitivity", "specificity_margin", "specificity_ratio", "theory_spearman", "top1_accuracy"]
        diffs = {k: [] for k in keys + ["entropy_rank_raw"]}
        for a, b in zip(Ms, Md):
            ma, mb = metrics(a, E), metrics(b, E)
            for k in keys:
                diffs[k].append(mb[k] - ma[k])
            diffs["entropy_rank_raw"].append(dimensionality(b)["entropy_rank"] - dimensionality(a)["entropy_rank"])
        pt_s, pt_d = metrics(mats["simple"], E), metrics(mats["definition"], E)
        point = {k: pt_d[k] - pt_s[k] for k in keys}
        point["entropy_rank_raw"] = (dimensionality(mats["definition"])["entropy_rank"]
                                     - dimensionality(mats["simple"])["entropy_rank"])
        rob = {k: dict(diff=float(point[k]), ci=ci(v), p=boot_p(v)) for k, v in diffs.items()}
        # per-construct own sensitivity, FDR across the 12 constructs
        own = own_rows(E)
        per = []
        for ci_, c in enumerate(ratings):
            js = [j for j in range(len(cols)) if own[j] == ci_]
            d = (Md[:, ci_, js].mean(1) - Ms[:, ci_, js].mean(1))
            per.append(dict(construct=c, diff=float(mats["definition"][ci_, js].mean() - mats["simple"][ci_, js].mean()),
                            ci=ci(d), p=boot_p(d)))
        for p_, qq in zip(per, bh([p["p"] for p in per])):
            p_["q"] = float(qq)
        rob["per_construct_own_sensitivity"] = per
        res["definition_minus_simple"] = rob
    if "reversed" in mats and "simple" in mats:
        sub = AV.reversed_subset(stim)
        Ds = pair_diffs(scores, stim, "simple", ratings, cols, bases=sub)
        Ms, Mr = effect_matrix(Ds), mats["reversed"]
        ok = ~np.isnan(Ms) & ~np.isnan(Mr)
        res["reversed_vs_simple"] = dict(cell_pearson=float(np.corrcoef(Ms[ok], Mr[ok])[0, 1]),
                                         mean_abs_diff=float(np.abs(Ms[ok] - Mr[ok]).mean()),
                                         own_sensitivity_simple_subset=metrics(Ms, E)["own_sensitivity"],
                                         own_sensitivity_reversed=metrics(Mr, E)["own_sensitivity"])
    return res, mats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_atlas/pilot_v1")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--null", type=int, default=1000)
    ap.add_argument("--representation", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    reg = AV.load_registry()
    stim = AV.generate(reg)
    Edf = AV.expected_matrix(reg)
    E, cols, ratings = Edf.values, list(Edf.columns), list(Edf.index)
    fam = np.array([reg["constructs"][c.split(".")[0]]["family"] for c in cols])
    out = os.path.join(a.root, "analysis")
    os.makedirs(out, exist_ok=True)
    Edf.to_csv(os.path.join(out, "expected_matrix.csv"))
    all_res, all_mats = {}, {}
    for mdir in sorted(glob.glob(os.path.join(a.root, "*/"))):
        m = os.path.basename(mdir.rstrip("/"))
        if m == "analysis" or not os.path.exists(os.path.join(mdir, "scores.csv")):
            continue
        man = json.load(open(os.path.join(mdir, "manifest.json")))
        assert man["expected_matrix_sha256"] == AV.expected_sha256(reg), f"{m}: expected matrix changed since scoring"
        r, mats = analyse_model(mdir, stim, reg, E, cols, ratings, fam, a.boot, a.null)
        r["quantized"] = man.get("quantized")
        if a.representation and os.path.exists(os.path.join(mdir, "rep_H.npy")):
            H = np.load(os.path.join(mdir, "rep_H.npy"))
            index = pd.read_csv(os.path.join(mdir, "rep_index.csv"))
            r["representation"] = representation_analysis(H, index, cols, mats.get("simple"))
        all_res[m], all_mats[m] = r, mats
        json.dump(r, open(os.path.join(mdir, "analysis.json"), "w"), indent=1)
        print(f"[{time.time() - t0:5.0f}s] analysed {m}: conditions {list(mats)}", flush=True)

    # cross-model similarity of vectorized effect matrices (all cells; expected-0 cells = confusion structure)
    keys = [(m, c) for m in all_mats for c in ("simple", "definition") if c in all_mats[m]]
    zero = (E == 0)
    sim = pd.DataFrame(index=[f"{m}|{c}" for m, c in keys], columns=[f"{m}|{c}" for m, c in keys], dtype=float)
    simz = sim.copy()
    for a_ in keys:
        for b_ in keys:
            x, y = all_mats[a_[0]][a_[1]], all_mats[b_[0]][b_[1]]
            sim.loc[f"{a_[0]}|{a_[1]}", f"{b_[0]}|{b_[1]}"] = np.corrcoef(x.ravel(), y.ravel())[0, 1]
            simz.loc[f"{a_[0]}|{a_[1]}", f"{b_[0]}|{b_[1]}"] = np.corrcoef(x[zero], y[zero])[0, 1]
    sim.to_csv(os.path.join(out, "similarity_all_cells.csv"))
    simz.to_csv(os.path.join(out, "similarity_zero_cells.csv"))
    fams = {}
    simple = [m for m in all_mats if "simple" in all_mats[m]]
    for i, a_ in enumerate(simple):
        for b_ in simple[i + 1:]:
            k = "within_family" if FAMILY.get(a_) == FAMILY.get(b_) else "between_family"
            fams.setdefault(k, []).append(float(simz.loc[f"{a_}|simple", f"{b_}|simple"]))
    # repeated confusions across models and families
    rep = {}
    for m in simple:
        for e in all_res[m]["conditions"]["simple"]["confusions"]:
            key = (e["from_construct"], e["rating"])
            rep.setdefault(key, set()).add(m)
    repeated = sorted([dict(from_construct=k[0], rating=k[1], n_models=len(v), families=sorted({FAMILY.get(x) for x in v}),
                            models=sorted(v)) for k, v in rep.items()], key=lambda d: (-d["n_models"], d["from_construct"]))
    # Qwen-only scaling table
    qwen = sorted([m for m in simple if FAMILY.get(m) == "Qwen"], key=lambda m: SIZES[m])
    scaling = [dict(model=m, size_b=SIZES[m], quantized=all_res[m]["quantized"],
                    **{k: all_res[m]["conditions"]["simple"]["point"][k] for k in
                       ["own_sensitivity", "specificity_margin", "theory_spearman", "top1_accuracy"]},
                    entropy_rank_raw=all_res[m]["conditions"]["simple"]["dimensionality"]["raw"]["point"]["entropy_rank"])
               for m in qwen]
    glob_res = dict(models=list(all_res), within_vs_between_family_zero_cell_similarity={k: dict(mean=float(np.mean(v)), n=len(v))
                                                                                        for k, v in fams.items()},
                    repeated_confusions=repeated, qwen_scaling=scaling,
                    # POST HOC reference: the theory matrix itself (NA -> 0) is not low-rank by construction
                    expected_dimensionality={k: v for k, v in dimensionality(np.nan_to_num(E)).items()},
                    expected_pc1_vs_valence_spearman=float(abs(spearmanr(
                        np.linalg.svd(np.nan_to_num(E))[2][0], valence_vector(reg, cols)).correlation)),
                    registry_sha256=AV.file_sha256(), expected_matrix_sha256=AV.expected_sha256(reg),
                    stimuli_sha256=AV.stimuli_sha256(stim), n_boot=a.boot, n_null=a.null)
    json.dump(glob_res, open(os.path.join(out, "global.json"), "w"), indent=1)
    json.dump(all_res, open(os.path.join(out, "all_models.json"), "w"), indent=1)
    rows = []
    for m, r in all_res.items():
        for c, rc in r["conditions"].items():
            rows.append(dict(model=m, condition=c, quantized=r["quantized"],
                             **{k: v for k, v in rc["point"].items() if k != "per_indicator"},
                             **{f"{k}_ci": v for k, v in rc["ci"].items()},
                             entropy_rank_raw=rc["dimensionality"]["raw"]["point"]["entropy_rank"],
                             entropy_rank_row=rc["dimensionality"]["row"]["point"]["entropy_rank"],
                             entropy_rank_col=rc["dimensionality"]["col"]["point"]["entropy_rank"],
                             var_3_raw=rc["dimensionality"]["raw"]["point"]["var_3"],
                             stable_rank_raw=rc["dimensionality"]["raw"]["point"]["stable_rank"]))
    pd.DataFrame(rows).to_csv(os.path.join(out, "model_comparison.csv"), index=False)
    print(f"analysis done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
