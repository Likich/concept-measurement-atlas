"""Pre-specified analysis for atlas_v2 behavioral scores.

No model is loaded here. The module validates score files, aggregates condition
D without hiding item scores, estimates the 12 x 36 effect matrices, preserves
strong/weak/null/unspecified registry classes, and implements paired family
bootstraps for the A--D comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import atlas_phase1 as P
import atlas_vignettes as AV

BOOT_SEED = 20261103
N_BOOT = 2000
EQUIV = 0.25


def validate_scores(scores: pd.DataFrame, stimuli: pd.DataFrame, parent: dict) -> None:
    required = {"model", "id", "condition", "rating", "score", "abstain", "mass"}
    missing = required - set(scores)
    if missing:
        raise ValueError(f"missing score columns: {sorted(missing)}")
    if not set(scores.id).issubset(set(stimuli.id)):
        raise ValueError("score file contains unknown stimulus ids")
    if not set(scores.condition).issubset(P.CONDITIONS):
        raise ValueError("unknown condition")
    if not set(scores.rating).issubset(parent["rating_order"]):
        raise ValueError("unknown rating")
    d = scores[scores.condition == "D_indicators"]
    if len(d):
        if "indicator" not in d or d.indicator.isna().any():
            raise ValueError("condition D requires the rated indicator column")
        valid = {(c, i) for c in parent["rating_order"] for i in parent["constructs"][c]["indicators"]}
        if not set(zip(d.rating, d.indicator)).issubset(valid):
            raise ValueError("condition D has an indicator outside its rating's operationalization")
        if "response_digit" not in d:
            raise ValueError("condition D requires response_digit so abstentions are not silently imputed")
    keys = ["model", "id", "condition", "rating"] + (["indicator"] if "indicator" in scores else [])
    if scores.duplicated(keys).any():
        raise ValueError("duplicate scores")


def aggregate_d(scores: pd.DataFrame, parent: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return construct composites and unmodified item rows for condition D.

    Primary composite: the equal mean of all three item scores.  No item is
    dropped based on its held-out response, so modal abstention cannot change
    the effective weights.  `score_available` is a diagnostic mean over items
    whose modal response is 1--5. Coverage, abstention, and every item score
    remain visible. Published weights could replace the equal vector only
    through a frozen amendment made before scoring.
    """
    items = scores[scores.condition == "D_indicators"].copy()
    if items.empty:
        return items, items
    items["available"] = items.response_digit.astype(int).ne(0)
    items["available_score"] = items.score.where(items.available)
    keys = ["model", "id", "rating"]
    comp = items.groupby(keys, sort=False).agg(
        score=("score", "mean"), score_available=("available_score", "mean"),
        coverage=("available", "sum"),
        mean_abstain=("abstain", "mean"), mean_mass=("mass", "mean"),
    ).reset_index()
    comp["condition"] = "D_indicators"
    comp["all_missing"] = comp.coverage.eq(0)
    return comp, items


def construct_scores(scores: pd.DataFrame, parent: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    non_d = scores[scores.condition != "D_indicators"].copy()
    d, items = aggregate_d(scores, parent)
    keep = ["model", "id", "condition", "rating", "score"]
    return pd.concat([non_d[keep], d[keep]], ignore_index=True), items


def pair_effects(scores: pd.DataFrame, stimuli: pd.DataFrame) -> pd.DataFrame:
    x = scores.merge(stimuli[["id", "base_id", "col", "construct", "indicator", "family", "domain", "pole",
                              "validation_status"]],
                     on="id", validate="many_to_one")
    wide = x.pivot(index=["model", "condition", "rating", "base_id", "col", "construct", "indicator",
                          "family", "domain", "validation_status"], columns="pole", values="score").reset_index()
    wide["complete_pair"] = wide[["high", "low"]].notna().all(axis=1)
    wide["effect"] = wide.high - wide.low
    return wide


def registry_cells(parent: dict) -> pd.DataFrame:
    E = AV.expected_matrix(parent)
    rows = []
    for rating in E.index:
        for col in E.columns:
            value = E.loc[rating, col]
            rows.append({"rating": rating, "col": col, "expected": value,
                         "class": "unspecified" if np.isnan(value) else
                                  ("null" if value == 0 else ("strong" if abs(value) == 2 else "weak")),
                         "expected_sign": np.nan if np.isnan(value) or value == 0 else np.sign(value)})
    return pd.DataFrame(rows)


def _equal_construct_mean(df: pd.DataFrame, value: str) -> float:
    return float(df.groupby("rating")[value].mean().mean())


def _cells(effects: pd.DataFrame, parent: dict) -> pd.DataFrame:
    """Cell = model x condition x rating x indicator column; effects averaged over scenario families."""
    cells = registry_cells(parent)
    g = effects[effects.complete_pair].groupby(["model", "condition", "rating", "col", "construct"])
    x = g.effect.agg(mean="mean", sd="std", n="count").reset_index().merge(cells, on=["rating", "col"],
                                                                          validate="many_to_one")
    x["abs_mean"] = x["mean"].abs()
    # noise-corrected |effect|: sqrt(max(0, m^2 - var/n)); |mean| alone is biased upward by noise, and differently
    # for conditions with different noise (e.g. the three-item composite of D)
    x["abs_corrected"] = np.sqrt(np.clip(x["mean"] ** 2 - x["sd"].fillna(0) ** 2 / x["n"], 0, None))
    x["signed"] = x["mean"] * x.expected_sign
    x["own"] = x.rating.eq(x.construct)
    return x


def metrics(effects: pd.DataFrame, parent: dict) -> pd.DataFrame:
    """Construct-level metrics per model x condition (pre-run correction C3: every unjustified-sensitivity quantity
    uses |cell-mean effect| over families, not the mean of per-family |effect|)."""
    x = _cells(effects, parent)
    out = []
    for (model, condition), panel in x.groupby(["model", "condition"], sort=False):
        for rating, g in panel.groupby("rating", sort=False):
            own = g[g.own].set_index("col")["mean"]
            margin = []
            for col, own_effect in own.items():
                nulls = panel[(panel.col == col) & (panel["class"] == "null")]
                if len(nulls):
                    margin.append(own_effect - nulls.abs_mean.max())
            null = g[g["class"] == "null"]
            out.append({
                "model": model, "condition": condition, "construct": rating,
                "own_convergence": own.mean(), "own_min_indicator": own.min(), "own_max_indicator": own.max(),
                "strong_sensitivity": g[g["class"] == "strong"].signed.mean(),
                "weak_sensitivity": g[g["class"] == "weak"].signed.mean() if (g["class"] == "weak").any() else np.nan,
                "null_abs_sensitivity": null.abs_mean.mean(),
                "null_abs_corrected": null.abs_corrected.mean(),
                "unspecified_abs_descriptive": g[g["class"] == "unspecified"].abs_mean.mean()
                if (g["class"] == "unspecified").any() else np.nan,
                "discrimination_margin": np.mean(margin) if margin else np.nan,
                "discrimination_share": np.mean(np.asarray(margin) > 0) if margin else np.nan,
                "family_sd_own_mean": g[g.own]["sd"].mean(),
            })
    return pd.DataFrame(out)


def global_primary(construct_metrics: pd.DataFrame) -> pd.DataFrame:
    return construct_metrics.groupby(["model", "condition"], sort=False).agg(
        R_own=("own_convergence", "mean"), R_strong=("strong_sensitivity", "mean"),
        U=("null_abs_corrected", "mean"), U_raw=("null_abs_sensitivity", "mean"),
        margin=("discrimination_margin", "mean")).reset_index()


SUCCESS = dict(retention=0.90, loss=0.20, loss_strict=0.15, u_reduction=0.10)


def _family_array(effects: pd.DataFrame, parent: dict, model: str, conditions: list[str]):
    """E[cond, rating, col, family] for one model; families indexed within each indicator column."""
    ratings = parent["rating_order"]
    e = effects[(effects.model == model) & effects.condition.isin(conditions)]
    present = set(e.col)  # validated-only analyses drop the four pending indicators entirely
    cols = [f"{c}.{i}" for c in ratings for i in parent["constructs"][c]["indicators"] if f"{c}.{i}" in present]
    fam = {b: k for col, g in e.groupby("col") for k, b in enumerate(sorted(g.base_id.unique()))}
    nf = max(fam.values()) + 1
    E = np.full((len(conditions), len(ratings), len(cols), nf), np.nan)
    ci, ri, ji = ({v: k for k, v in enumerate(seq)} for seq in (conditions, ratings, cols))
    for r in e.itertuples():
        E[ci[r.condition], ri[r.rating], ji[r.col], fam[r.base_id]] = r.effect
    assert np.isfinite(E).all(), "incomplete pairs: every family needs both poles in every condition and rating"
    return E, ratings, cols


def _stats(M, cells, ratings, cols, Mabs=None):
    """M[..., cond, rating, col] cell means -> equal-construct R_own, R_strong and U (from Mabs if given)."""
    own = np.array([[c.startswith(r + ".") for c in cols] for r in ratings])
    cls = cells.set_index(["rating", "col"])
    E = np.array([[cls.loc[(r, c), "expected"] for c in cols] for r in ratings])
    strong, null = np.abs(E) == 2, E == 0
    sign = np.nan_to_num(np.sign(E))
    def cmean(V, mask):  # equal-construct mean: within-construct mean over masked cells, then mean over ratings
        w = mask.astype(float)
        return ((V * w).sum(-1) / np.maximum(w.sum(-1), 1)).mean(-1)
    return cmean(M, own), cmean(M * sign, strong), cmean(np.abs(M) if Mabs is None else Mabs, null)


def paired_bootstrap(effects: pd.DataFrame, parent: dict, comparison: str, n_boot: int = N_BOOT,
                     seed: int = BOOT_SEED, families: str = "all") -> pd.DataFrame:
    """Family-level bootstrap (pre-run correction C3). Within each indicator column, scenario families are resampled
    with replacement; each draw is shared by all 12 ratings and both conditions (repeated measures)."""
    if comparison == "A_simple":
        raise ValueError("comparison must be B, C, or D")
    cells = registry_cells(parent)
    rows = []
    for model in effects.model.unique():
        e = effects
        if families == "validated_only":
            e = effects[effects.validation_status != "pending_human_check"]
        E, ratings, cols = _family_array(e, parent, model, ["A_simple", comparison])
        keep = [k for k, c in enumerate(cols) if np.isfinite(E[0, 0, k]).all()]
        rng = np.random.default_rng(seed)
        nf = E.shape[-1]
        idx = np.concatenate([np.arange(nf)[None], rng.integers(0, nf, (n_boot, nf))])  # draw 0 = observed
        Eb = np.stack([E[..., d] for d in idx])  # [B+1, 2, R, J, F]: the same families for every rating and arm
        Mb = Eb.mean(-1)
        # primary U: noise-corrected |cell mean| (|mean| of 6 families is biased by ~sd/sqrt(6), and the bias differs
        # between conditions with different noise); the plain |mean| is the sensitivity analysis U_raw
        Mc = np.sqrt(np.clip(Mb ** 2 - Eb.var(-1, ddof=1) / nf, 0, None))
        own, strong, null = _stats(Mb, cells, ratings, cols, Mc)
        null_raw = _stats(Mb, cells, ratings, cols)[2]
        RA, Rh, UA, Uh, SA, Sh = own[:, 0], own[:, 1], null[:, 0], null[:, 1], strong[:, 0], strong[:, 1]
        ret = Rh / np.where(RA > 0, RA, np.nan)
        stat = dict(R_A=RA, R_h=Rh, retention=ret, absolute_loss=RA - Rh, U_A=UA, U_h=Uh, U_reduction=UA - Uh,
                    U_relative_reduction=(UA - Uh) / np.where(UA > 0, UA, np.nan), Rstrong_A=SA, Rstrong_h=Sh,
                    selectivity=(UA - Uh) / np.where(UA > 0, UA, np.nan) - (1 - ret),
                    U_raw_A=null_raw[:, 0], U_raw_h=null_raw[:, 1],
                    U_raw_relative_reduction=(null_raw[:, 0] - null_raw[:, 1]) / np.where(null_raw[:, 0] > 0, null_raw[:, 0], np.nan))
        row = {"model": model, "comparison": comparison, "families": families, "n_boot": n_boot}
        for name, v in stat.items():
            lo, hi = (5, 95) if name in ("retention", "absolute_loss") else (2.5, 97.5)
            row[name] = float(v[0])
            row[f"{name}_lo"], row[f"{name}_hi"] = (float(np.nanpercentile(v[1:], q)) for q in (lo, hi))
        row["p_U_reduction"] = float((np.sum(stat["U_reduction"][1:] <= 0) + 1) / (n_boot + 1))
        # point-estimate criteria (descriptive) and interval criteria (statistical), reported separately
        row["point_retention_090"] = row["retention"] >= SUCCESS["retention"]
        row["point_loss_020"] = row["absolute_loss"] <= SUCCESS["loss"]
        row["point_loss_015"] = row["absolute_loss"] <= SUCCESS["loss_strict"]
        row["point_U_reduction_10pct"] = row["U_relative_reduction"] >= SUCCESS["u_reduction"]
        row["point_success_020"] = bool(row["point_retention_090"] and row["point_loss_020"] and row["point_U_reduction_10pct"])
        row["point_success_015"] = bool(row["point_retention_090"] and row["point_loss_015"] and row["point_U_reduction_10pct"])
        row["ci_retention_090"] = row["retention_lo"] >= SUCCESS["retention"]
        row["ci_loss_020"] = row["absolute_loss_hi"] <= SUCCESS["loss"]
        row["ci_loss_015"] = row["absolute_loss_hi"] <= SUCCESS["loss_strict"]
        row["ci_U_reduction_gt0"] = row["U_reduction_lo"] > 0
        row["ci_U_reduction_10pct"] = row["U_relative_reduction_lo"] >= SUCCESS["u_reduction"]
        row["ci_success_020"] = bool(row["ci_retention_090"] and row["ci_loss_020"] and row["ci_U_reduction_10pct"])
        row["ci_success_015"] = bool(row["ci_retention_090"] and row["ci_loss_015"] and row["ci_U_reduction_10pct"])
        # a uniform shrinkage of all effects is not a correction: relative null reduction must exceed relative loss
        row["uniform_shrinkage"] = bool(row["U_relative_reduction"] > 0 and row["retention"] < 1
                                        and row["selectivity_lo"] <= 0)
        rows.append(row)
    return pd.DataFrame(rows)


def holm(df: pd.DataFrame) -> pd.DataFrame:
    """Holm adjustment of p_U_reduction over the three planned comparisons within model and family set."""
    df = df.copy()
    df["p_U_reduction_holm"] = np.nan
    for _, g in df.groupby(["model", "families"]):
        order = g.p_U_reduction.sort_values().index
        adj = np.maximum.accumulate([min(1.0, (len(order) - k) * df.loc[i, "p_U_reduction"]) for k, i in enumerate(order)])
        df.loc[order, "p_U_reduction_holm"] = adj
    return df


def analyse(scores_path: Path, out: Path) -> None:
    parent, _, specs = P.load()
    stimuli = P.generate(parent, specs)
    scores = pd.read_csv(scores_path)
    validate_scores(scores, stimuli, parent)
    constructs, items = construct_scores(scores, parent)
    effects = pair_effects(constructs, stimuli)
    out.mkdir(parents=True, exist_ok=True)
    effects.to_csv(out / "family_effects.csv", index=False)
    _cells(effects, parent).to_csv(out / "cell_effects.csv", index=False)
    for fam in ("all", "validated_only"):
        e = effects if fam == "all" else effects[effects.validation_status != "pending_human_check"]
        cm = metrics(e, parent)
        cm.to_csv(out / f"construct_metrics_{fam}.csv", index=False)
        global_primary(cm).to_csv(out / f"global_primary_{fam}.csv", index=False)
    items.to_csv(out / "condition_D_item_scores.csv", index=False)
    comps = [paired_bootstrap(effects, parent, h, families=f) for h in ("B_inclusion", "C_inclusion_exclusions",
                                                                         "D_indicators") for f in ("all", "validated_only")
             if h in set(effects.condition)]
    if comps:
        holm(pd.concat(comps, ignore_index=True)).to_csv(out / "comparisons.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--out", default="reports/atlas_phase1/analysis")
    a = ap.parse_args()
    analyse(Path(a.scores), Path(a.out))


if __name__ == "__main__":
    main()
