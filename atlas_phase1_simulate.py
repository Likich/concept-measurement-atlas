"""Model-free simulation of alternative non-inferiority margins.

The six stored simple-condition atlas matrices supply the observed effect-size
and family-variation range. Synthetic correction methods alter registered
strong and null cell means; no language model is loaded or called.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import atlas_analysis as AA
import atlas_vignettes as AV

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "results_atlas" / "pilot_v1"
OUT = ROOT / "reports" / "atlas_phase1" / "simulation"
MODELS = [
    "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct", "OLMo-2-1124-13B-Instruct", "Qwen2.5-32B-Instruct",
]
SCENARIOS = {
    "no_change": (1.00, 1.00),
    "selective_strong": (0.95, 0.50),
    "selective_boundary": (0.90, 0.70),
    "uniform_20pct_suppression": (0.80, 0.80),
    "leakage_reduced_but_relevance_harmed": (0.70, 0.45),
}
RELATIVE_MARGINS = (0.80, 0.85, 0.90, 0.95)
ABSOLUTE_MARGINS = (0.10, 0.15, 0.20, 0.25)


def data():
    reg = AV.load_registry()
    stim = AV.generate(reg)
    exp = AV.expected_matrix(reg)
    E, ratings, cols = exp.values, list(exp.index), list(exp.columns)
    panels = {}
    for model in MODELS:
        scores = pd.read_csv(PILOT / model / "scores.csv")
        D = AA.pair_diffs(scores, stim, "simple", ratings, cols)
        M = np.nanmean(D, axis=2)
        n = np.sum(~np.isnan(D), axis=2)
        se = np.nanstd(D, axis=2, ddof=1) / np.sqrt(n)
        # Avoid zero Monte Carlo variance for cells whose six pilot pairs happen
        # to be identical; floor is conservative relative to the 1--5 scale.
        se = np.maximum(np.nan_to_num(se, nan=.20), .08)
        panels[model] = (M, se)
    return reg, E, ratings, cols, panels


def summaries(X, E, ratings):
    """Return equal-cell construct vectors R and U from a 12 x 36 matrix."""
    R, U = [], []
    for i, _ in enumerate(ratings):
        strong = np.abs(E[i]) == 2
        null = E[i] == 0
        R.append(np.mean(X[i, strong] * np.sign(E[i, strong])))
        U.append(np.mean(np.abs(X[i, null])))
    return np.asarray(R), np.asarray(U)


def intervals(A, H, se, E, ratings, n_boot, rng):
    """Parametric approximation to the registered paired-family bootstrap.

    Cell-level uncertainty comes from the six observed matched families. The
    same 60% noise component enters both prompt arms to reflect paired texts.
    Aggregation gives equal weight to constructs before intervals are formed.
    """
    shared = rng.normal(size=(n_boot,) + A.shape)
    ab = A + se * (np.sqrt(.60) * shared + np.sqrt(.40) * rng.normal(size=shared.shape))
    hb = H + se * (np.sqrt(.60) * shared + np.sqrt(.40) * rng.normal(size=shared.shape))
    ra_parts, rh_parts, ua_parts, uh_parts = [], [], [], []
    for i, _ in enumerate(ratings):
        strong = np.abs(E[i]) == 2
        null = E[i] == 0
        ra_parts.append(np.mean(ab[:, i, strong] * np.sign(E[i, strong]), axis=1))
        rh_parts.append(np.mean(hb[:, i, strong] * np.sign(E[i, strong]), axis=1))
        ua_parts.append(np.mean(np.abs(ab[:, i, null]), axis=1))
        uh_parts.append(np.mean(np.abs(hb[:, i, null]), axis=1))
    ra = np.stack(ra_parts).mean(0)
    rh = np.stack(rh_parts).mean(0)
    ua = np.stack(ua_parts).mean(0)
    uh = np.stack(uh_parts).mean(0)
    retention = rh / ra
    loss = ra - rh
    reduction = ua - uh
    Ra, Ua = summaries(A, E, ratings)
    Rh, Uh = summaries(H, E, ratings)
    return {
        "retention_lo90": np.percentile(retention, 5),
        "retention_hi90": np.percentile(retention, 95),
        "loss_lo90": np.percentile(loss, 5),
        "loss_hi90": np.percentile(loss, 95),
        "reduction_lo95": np.percentile(reduction, 2.5),
        "reduction_hi95": np.percentile(reduction, 97.5),
        "R_A": Ra.mean(), "R_h": Rh.mean(), "U_A": Ua.mean(), "U_h": Uh.mean(),
    }


def simulate(n_rep=400, n_boot=400, seed=20261103):
    reg, E, ratings, cols, panels = data()
    rng = np.random.default_rng(seed)
    strong = np.abs(E) == 2
    null = E == 0
    records = []
    for model, (M, se) in panels.items():
        for scenario, (r_mult, u_mult) in SCENARIOS.items():
            target = M.copy()
            target[strong] *= r_mult
            target[null] *= u_mult
            for rep in range(n_rep):
                shared = rng.normal(size=M.shape)
                ea = se * (np.sqrt(.60) * shared + np.sqrt(.40) * rng.normal(size=M.shape))
                eh = se * (np.sqrt(.60) * shared + np.sqrt(.40) * rng.normal(size=M.shape))
                A = M + ea
                H = target + eh
                Ra, Ua = summaries(A, E, ratings)
                Rh, Uh = summaries(H, E, ratings)
                z = intervals(A, H, se, E, ratings, n_boot, rng)
                z.update(model=model, scenario=scenario, rep=rep,
                         true_relevant_multiplier=r_mult, true_null_multiplier=u_mult)
                records.append(z)
    draws = pd.DataFrame(records)
    rows = []
    for rel in RELATIVE_MARGINS:
        for absolute in ABSOLUTE_MARGINS:
            q = draws.copy()
            q["passes_U"] = q.reduction_lo95 > 0
            q["passes_relative"] = q.retention_lo90 >= rel
            q["passes_absolute"] = q.loss_hi90 <= absolute
            q["passes_all"] = q.passes_U & q.passes_relative & q.passes_absolute
            for (model, scenario), g in q.groupby(["model", "scenario"], sort=False):
                rows.append({
                    "model": model, "scenario": scenario,
                    "relative_margin": rel, "absolute_margin": absolute,
                    "pass_rate": g.passes_all.mean(), "U_gate_rate": g.passes_U.mean(),
                    "relative_gate_rate": g.passes_relative.mean(),
                    "absolute_gate_rate": g.passes_absolute.mean(), "n_rep": len(g),
                })
    grid = pd.DataFrame(rows)
    aggregate = grid.groupby(["scenario", "relative_margin", "absolute_margin"], sort=False).agg(
        pass_rate_mean=("pass_rate", "mean"), pass_rate_min_model=("pass_rate", "min"),
        pass_rate_max_model=("pass_rate", "max"), U_gate_mean=("U_gate_rate", "mean"),
        relative_gate_mean=("relative_gate_rate", "mean"), absolute_gate_mean=("absolute_gate_rate", "mean"),
    ).reset_index()
    return draws, grid, aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=400)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    draws, grid, aggregate = simulate(a.rep, a.boot)
    draws.to_csv(out / "simulation_draws.csv", index=False)
    grid.to_csv(out / "margin_grid_by_model.csv", index=False)
    aggregate.to_csv(out / "margin_grid_aggregate.csv", index=False)
    primary = aggregate[(aggregate.relative_margin == .90) & (aggregate.absolute_margin == .15)]
    primary.to_csv(out / "provisional_090_015.csv", index=False)
    print(primary[["scenario", "pass_rate_mean", "pass_rate_min_model", "pass_rate_max_model"]].to_string(index=False))


if __name__ == "__main__":
    main()
