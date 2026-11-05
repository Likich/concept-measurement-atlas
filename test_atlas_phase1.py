"""Model-free tests for atlas_v2 Phase 1. Run with: python test_atlas_phase1.py"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import atlas_phase1 as P
import atlas_phase1_analysis as A
import atlas_vignettes as AV
import atlas_phase2_validate as V2

PARENT, MEASUREMENT, SPECS = P.load()
STIM = P.generate(PARENT, SPECS)


def test_structure_and_independence():
    issues, lex, checks = P.audit(PARENT, MEASUREMENT, SPECS, STIM)
    assert all(c["pass"] for c in checks), [c for c in checks if not c["pass"]]
    assert issues.empty
    assert len(STIM) == 432 and STIM.base_id.nunique() == 216 and STIM.col.nunique() == 36
    assert STIM.groupby("col").base_id.nunique().eq(6).all()


def test_condition_c_reconstructs_existing_definition():
    for c in PARENT["rating_order"]:
        m = MEASUREMENT["constructs"][c]
        assert m["c_matches_existing_definition"] is True
        combined = " ".join((m["inclusion"].strip(), m["exclusions"].strip()))
        assert combined == PARENT["constructs"][c]["definition"].strip(), (c, combined)


def test_prompts_are_distinct_and_d_is_item_level():
    text = STIM.iloc[0].text
    a = P.build_prompt(PARENT, MEASUREMENT, "trust", text, "A_simple")
    b = P.build_prompt(PARENT, MEASUREMENT, "trust", text, "B_inclusion")
    c = P.build_prompt(PARENT, MEASUREMENT, "trust", text, "C_inclusion_exclusions")
    d = P.build_prompt(PARENT, MEASUREMENT, "trust", text, "D_indicators", "ability")
    assert "TRUST =" not in a
    assert MEASUREMENT["constructs"]["trust"]["inclusion"] in b
    assert MEASUREMENT["constructs"]["trust"]["exclusions"] not in b
    assert MEASUREMENT["constructs"]["trust"]["exclusions"] in c
    assert "How competent" in d and "complete trust" not in d
    assert len({a, b, c, d}) == 4


def test_prompt_budget():
    n = len(STIM)
    assert n * 12 * 3 == 15552
    assert n * 36 == 15552
    assert n * 12 * 3 + n * 36 == 31104


def test_registry_classes_preserved():
    cells = A.registry_cells(PARENT)
    assert cells["class"].value_counts().to_dict() == {
        "null": 346, "strong": 37, "weak": 26, "unspecified": 23
    }


def test_d_aggregation_and_item_visibility():
    rows = []
    item_scores = [2.0, 5.0, 4.0]
    for k, indicator in enumerate(PARENT["constructs"]["trust"]["indicators"]):
        rows.append({"model": "m", "id": STIM.iloc[0].id, "condition": "D_indicators", "rating": "trust",
                     "indicator": indicator, "score": item_scores[k], "abstain": .1, "mass": .9,
                     "response_digit": 0 if k == 1 else k + 2})
    scores = pd.DataFrame(rows)
    comp, items = A.aggregate_d(scores, PARENT)
    assert len(items) == 3 and comp.iloc[0].coverage == 2
    assert np.isclose(comp.iloc[0].score, (2 + 5 + 4) / 3)
    assert np.isclose(comp.iloc[0].score_available, (2 + 4) / 2)
    assert not comp.iloc[0].all_missing


def planted_scores():
    E = AV.expected_matrix(PARENT)
    rows = []
    for s in STIM.itertuples():
        pole = .5 if s.pole == "high" else -.5
        for rating in PARENT["rating_order"]:
            e = E.loc[rating, s.col]
            e = 0 if np.isnan(e) else e
            rows.append({"model": "planted", "id": s.id, "condition": "A_simple", "rating": rating,
                         "score": 3 + pole * e, "abstain": 0., "mass": 1.})
    return pd.DataFrame(rows)


def test_planted_analysis_recovery():
    scores = planted_scores()
    A.validate_scores(scores, STIM, PARENT)
    effects = A.pair_effects(scores, STIM)
    cm = A.metrics(effects, PARENT)
    assert np.allclose(cm.own_convergence, 2)
    assert np.allclose(cm.null_abs_sensitivity, 0)
    assert (cm.strong_sensitivity > 0).all()
    assert (cm.discrimination_margin > 0).all()


def test_paired_bootstrap_retains_null_cells():
    a = planted_scores()
    b = a.copy()
    b["condition"] = "B_inclusion"
    scores = pd.concat([a, b], ignore_index=True)
    effects = A.pair_effects(scores, STIM)
    out = A.paired_bootstrap(effects, PARENT, "B_inclusion", n_boot=100, seed=4)
    assert len(out) == 1
    assert np.isclose(out.iloc[0].U_A, 0) and np.isclose(out.iloc[0].U_h, 0)
    assert np.isclose(out.iloc[0].retention, 1)


def test_technical_pilot_end_to_end_contract():
    selected = {f"{c}.{i}" for c, i in MEASUREMENT["causal_screening"]["indicators"].items()}
    pilot = STIM[STIM.col.isin(selected) & STIM.base_id.str.endswith("h0")]
    rows = []
    for condition in P.CONDITIONS:
        for rating in PARENT["rating_order"]:
            indicators = list(PARENT["constructs"][rating]["indicators"]) if condition == "D_indicators" else [None]
            for indicator in indicators:
                for s in pilot.itertuples():
                    own = rating == s.construct
                    targeted_item = condition != "D_indicators" or indicator == s.indicator
                    score = 3.0 + ((1.0 if s.pole == "high" else -1.0) if own and targeted_item else 0.0)
                    rows.append({
                        "model": "Qwen/Qwen2.5-7B-Instruct", "id": s.id, "condition": condition,
                        "rating": rating, "indicator": indicator, "score": score,
                        "abstain": 0.01, "mass": 0.99, "response_digit": int(round(score)),
                    })
    scores = pd.DataFrame(rows)
    manifest = P.manifests(PARENT, MEASUREMENT, STIM)
    manifest.update({
        "model": "Qwen/Qwen2.5-7B-Instruct", "gpu_name": "NVIDIA RTX A6000",
        "visible_gpu_count": 1, "prompt_sha256": V2.expected_prompt_hash(PARENT, MEASUREMENT, STIM),
    })
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        scores.to_csv(td / "scores.csv", index=False)
        (td / "manifest.json").write_text(json.dumps(manifest))
        report, directions = V2.validate(td / "scores.csv", td / "manifest.json")
    assert report["technical_pass"]
    assert report["complete"] and report["all_four_conditions_present"]
    assert report["rating_extraction_valid"] and report["prompt_sha256_matches"]
    assert len(report["expected_response_direction_by_condition"]) == 4
    assert directions.condition.nunique() == 4


def test_a_and_c_are_the_original_atlas_prompts():
    import atlas as OLD
    text = STIM.text.iloc[0]
    for c in PARENT["rating_order"]:
        assert P.build_prompt(PARENT, MEASUREMENT, c, text, "A_simple") == OLD.build_prompt(PARENT, c, text, "simple")
        assert P.build_prompt(PARENT, MEASUREMENT, c, text, "C_inclusion_exclusions") == OLD.build_prompt(PARENT, c, text, "definition")
        for i in PARENT["constructs"][c]["indicators"]:
            d = P.build_prompt(PARENT, MEASUREMENT, c, text, "D_indicators", i)
            assert "(low)" not in d and "(high)" not in d and "described the same way" not in d


def _synthetic_effects(rng, sd_family, cond, shared=True):
    rows = []
    for col in sorted(STIM.col.unique()):
        construct = col.split(".")[0]
        for b in sorted(STIM[STIM.col == col].base_id.unique()):
            u = rng.normal()
            for rating in PARENT["rating_order"]:
                own = 1.0 if rating == construct else 0.0
                noise = u if shared else rng.normal()
                rows.append(dict(model="m", condition=cond, rating=rating, base_id=b, col=col, construct=construct,
                                 effect=own + sd_family * noise, complete_pair=True, validation_status="x"))
    return pd.DataFrame(rows)


def test_family_bootstrap_keeps_dependence_across_ratings():
    # a family's effect is shared by all 12 ratings: row-level resampling would shrink the interval ~sqrt(12)-fold
    rng = np.random.default_rng(0)
    e = pd.concat([_synthetic_effects(rng, 0.5, c) for c in ("A_simple", "B_inclusion")])
    out = A.paired_bootstrap(e, PARENT, "B_inclusion", n_boot=400, seed=1).iloc[0]
    per_indicator = e[(e.condition == "A_simple") & (e.rating == e.construct)].groupby("col").effect
    se = np.sqrt((per_indicator.var() / 6).mean() / 36)  # equal-construct mean over 36 independent indicators
    width = (out.R_A_hi - out.R_A_lo) if "R_A_hi" in out else None
    assert width is not None and width > 2 * 1.645 * se * 0.6, (width, se)


def test_noise_difference_does_not_create_null_reduction():
    # true null effects are 0 in both arms; arm h is three times less noisy (like a 3-item composite)
    rng = np.random.default_rng(2)
    a = _synthetic_effects(rng, 0.6, "A_simple", shared=False)
    d = _synthetic_effects(rng, 0.2, "D_indicators", shared=False)
    out = A.paired_bootstrap(pd.concat([a, d]), PARENT, "D_indicators", n_boot=300, seed=3).iloc[0]
    assert out.U_raw_relative_reduction > 0.4  # the plain |mean| version is fooled
    assert out.U_A < 0.12 and out.U_h < 0.12   # the corrected version stays near the true value 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok ", test.__name__)
    print(f"all {len(tests)} Phase-1 tests passed")
