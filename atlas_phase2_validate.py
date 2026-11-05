"""Validate the frozen 24-text atlas_v2 technical pilot after GPU execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import atlas_phase1 as P
import atlas_phase1_analysis as A


def expected_prompt_hash(parent: dict, measurement: dict, full: pd.DataFrame) -> str:
    import hashlib
    selected = {f"{c}.{i}" for c, i in measurement["causal_screening"]["indicators"].items()}
    stimuli = full[full.col.isin(selected) & full.base_id.str.endswith("h0")].copy()
    jobs = []
    for condition in P.CONDITIONS:
        if condition == "D_indicators":
            jobs.extend((condition, rating, indicator) for rating in parent["rating_order"]
                        for indicator in parent["constructs"][rating]["indicators"])
        else:
            jobs.extend((condition, rating, None) for rating in parent["rating_order"])
    lines = []
    for condition, rating, indicator in jobs:
        for row in stimuli.itertuples():
            prompt = P.build_prompt(parent, measurement, rating, row.text, condition, indicator)
            lines.append("\t".join((row.id, condition, rating, indicator or "", prompt.replace("\n", "\\n"))))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def validate(scores_path: Path, manifest_path: Path) -> tuple[dict, pd.DataFrame]:
    parent, measurement, specs = P.load()
    full = P.generate(parent, specs)
    selected = {f"{c}.{i}" for c, i in measurement["causal_screening"]["indicators"].items()}
    stim = full[full.col.isin(selected) & full.base_id.str.endswith("h0")].copy()
    scores = pd.read_csv(scores_path)
    manifest = json.loads(manifest_path.read_text())
    A.validate_scores(scores, stim, parent)
    expected = {
        "A_simple": 24 * 12, "B_inclusion": 24 * 12,
        "C_inclusion_exclusions": 24 * 12, "D_indicators": 24 * 36,
    }
    observed = scores.groupby("condition").size().to_dict()
    construct_scores, items = A.construct_scores(scores, parent)
    effects = A.pair_effects(construct_scores, stim)
    own = effects[(effects.rating == effects.construct)]
    selected_own = own[own.col.isin(selected)].copy()
    expected_hash = expected_prompt_hash(parent, measurement, full)
    direction_by_condition = selected_own.groupby("condition").effect.agg(
        n="size", positive_share=lambda x: float((x > 0).mean()), mean="mean"
    ).reset_index().to_dict("records")
    parsing_valid = bool(
        np.isfinite(scores.score).all()
        and scores.response_digit.between(0, 5).all()
        and scores.mass.between(0, 1 + 1e-5).all()      # float32 softmax sums can exceed 1 by ~2e-7
        and scores.abstain.between(0, 1 + 1e-5).all()
    )
    report = {
        "model": manifest.get("model"), "gpu": manifest.get("gpu_name"),
        "visible_gpu_count": manifest.get("visible_gpu_count"),
        "hardware_is_a6000": any(g in str(manifest.get("gpu_name", "")).upper() for g in ("H100", "A6000")),  # name kept for compatibility: approved hardware
        "stimuli_sha256_matches": manifest.get("stimuli_sha256") == P.manifests(parent, measurement, full)["stimuli_sha256"],
        "expected_prompt_sha256": expected_hash,
        "prompt_sha256_matches": manifest.get("prompt_sha256") == expected_hash,
        "expected_rows": expected, "observed_rows": observed,
        "complete": observed == expected and len(scores) == 1728,
        "duplicates": int(scores.duplicated(["model", "id", "condition", "rating", "indicator"]).sum()),
        "all_four_conditions_present": set(observed) == set(P.CONDITIONS),
        "rating_extraction_valid": parsing_valid,
        "mass_median": float(scores.mass.median()),
        "mass_p05": float(scores.mass.quantile(.05)),
        "share_mass_ge_050": float((scores.mass >= .50).mean()),
        "mean_abstain": float(scores.abstain.mean()),
        "share_modal_zero": float((scores.response_digit == 0).mean()),
        "condition_D_mean_coverage": float(items.groupby(["id", "rating"]).available.sum().mean()),
        "condition_D_all_missing_share": float(items.groupby(["id", "rating"]).available.sum().eq(0).mean()),
        "selected_own_positive_share": float((selected_own.effect > 0).mean()),
        "selected_own_effect_mean": float(selected_own.effect.mean()),
        "expected_response_direction_by_condition": direction_by_condition,
        "technical_pass": False,
        "note": "Expected-response direction is diagnostic only. Technical success is not evidence of construct validity and cannot be used to tune prompts or stimuli.",
    }
    report["technical_pass"] = bool(
        report["complete"] and report["duplicates"] == 0 and report["stimuli_sha256_matches"]
        and report["prompt_sha256_matches"] and report["visible_gpu_count"] == 1
        and report["hardware_is_a6000"] and report["all_four_conditions_present"]
        and report["rating_extraction_valid"] and report["share_mass_ge_050"] >= .95
    )
    return report, selected_own[["model", "condition", "construct", "indicator", "domain", "effect", "complete_pair"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("manifest")
    ap.add_argument("--out", default="reports/atlas_phase2")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report, directions = validate(Path(a.scores), Path(a.manifest))
    (out / "pilot_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    directions.to_csv(out / "pilot_selected_own_effects.csv", index=False)
    print(json.dumps(report, indent=2))
    if not report["technical_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
