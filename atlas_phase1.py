"""Phase-1 utilities for the confirmatory 12-construct atlas.

This module is deliberately model-free. It loads the immutable pilot expected-
effect registry, the separate atlas_v2 measurement registry, and 216 newly
authored matched vignette families. It generates the 432 texts, constructs the
four measurement prompts, audits the material, and writes frozen manifests.

Usage:
    python atlas_phase1.py export --out reports/atlas_phase1
    python atlas_phase1.py check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import atlas_vignettes as pilot

ROOT = Path(__file__).resolve().parent
V2 = ROOT / "atlas" / "atlas_v2"
PARENT = ROOT / "atlas" / "pilot_registry.yaml"
MEASUREMENT = V2 / "measurement_registry.yaml"
STIM_FILES = [
    V2 / "stimuli_affect.yaml",
    V2 / "stimuli_social.yaml",
    V2 / "stimuli_evaluation.yaml",
    V2 / "stimuli_agency.yaml",
]
DOMAINS = ("work", "personal", "public", "consumer")
CONDITIONS = ("A_simple", "B_inclusion", "C_inclusion_exclusions", "D_indicators")
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_frame(df: pd.DataFrame, cols: list[str]) -> str:
    lines = df.sort_values("id")[cols].astype(str).agg("\t".join, axis=1)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def load() -> tuple[dict, dict, dict]:
    parent = yaml.safe_load(PARENT.read_text())
    measurement = yaml.safe_load(MEASUREMENT.read_text())
    stimuli: dict = {}
    for path in STIM_FILES:
        part = yaml.safe_load(path.read_text())
        overlap = set(stimuli) & set(part)
        if overlap:
            raise ValueError(f"duplicate constructs in stimulus files: {sorted(overlap)}")
        stimuli.update(part)
    return parent, measurement, stimuli


def generate(parent: dict | None = None, specs: dict | None = None) -> pd.DataFrame:
    parent, _, specs = load() if parent is None or specs is None else (parent, None, specs)
    rows = []
    for c in parent["rating_order"]:
        for i in parent["constructs"][c]["indicators"]:
            for b, item in enumerate(specs[c][i]):
                base_id = f"{c}.{i}.h{b}"
                for pole in ("high", "low"):
                    evidence = item[pole].strip()
                    rows.append({
                        "id": f"{base_id}.{pole}", "base_id": base_id,
                        "construct": c, "indicator": i, "col": f"{c}.{i}",
                        "family": parent["constructs"][c]["family"],
                        "domain": item["domain"], "pole": pole,
                        "context": item["context"].strip(), "evidence": evidence,
                        "closing": item["closing"].strip(),
                        "text": " ".join((item["context"].strip(), evidence, item["closing"].strip())),
                        "material": "new_confirmatory_2026",
                    })
    df = pd.DataFrame(rows)
    pending = set(yaml.safe_load(MEASUREMENT.read_text()).get("validation_pending", []))
    # C5: the 24 families awaiting a human manipulation check stay in every run and are labelled, never dropped
    df["validation_status"] = np.where(df.col.isin(pending), "pending_human_check", "text_audit_only")
    return df


def indicator_anchors(parent: dict, construct: str, indicator: str) -> tuple[str, str]:
    desc = parent["constructs"][construct]["indicators"][indicator]["description"]
    # Descriptions are registered as "high ... vs low ..."; preserve the full
    # wording when an automatic split would be ambiguous.
    bits = re.split(r"\s+vs\s+", desc, maxsplit=1, flags=re.I)
    if len(bits) == 2:
        hi, lo = bits[0].strip(" ."), bits[1].strip(" .")
    else:
        hi, lo = f"strong evidence for {indicator}", f"strong evidence against {indicator}"
    return lo, hi


def scale(low: str, high: str) -> str:
    # Pre-run correction C1 (Claude, 2026-11-04): the original atlas wording (atlas.build_prompt), so that
    # A is the historical `simple` prompt and C the historical `definition` prompt word for word.
    return f"0 (not enough evidence in the text), 1 ({low}), 2 to 4 (increasing levels) to 5 ({high})"


def build_prompt(parent: dict, measurement: dict, rating: str, text: str,
                 condition: str, indicator: str | None = None) -> str:
    c = parent["constructs"][rating]
    if condition == "D_indicators":
        if indicator is None:
            raise ValueError("condition D requires an indicator")
        q = measurement["constructs"][rating]["indicator_questions"][indicator]
        # Pre-run correction C2: explicit registered anchors (the automatic split of the registry descriptions
        # carried "(low)"/"(high)" tags and an author's design note into the prompt).
        a = measurement["constructs"][rating]["indicator_anchors"][indicator]
        lo, hi = a["low"], a["high"]
        body = f"{q} Answer on a scale from {scale(lo, hi)}. Reply with a single digit."
        return f"Read the text below.\n\nText: {text}\n\n{body}"
    lo, hi = c["anchors"]["low"], c["anchors"]["high"]
    body = f"{c['question']} Answer on a scale from {scale(lo, hi)}. Reply with a single digit."
    if condition == "A_simple":
        return f"Read the text below.\n\nText: {text}\n\n{body}"
    m = measurement["constructs"][rating]
    definition = m["inclusion"]
    if condition == "C_inclusion_exclusions":
        definition += " " + m["exclusions"]
    elif condition != "B_inclusion":
        raise ValueError(condition)
    return f"You are coding texts for a social-science study.\n{definition}\n\nText: {text}\n\n{body}"


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def _ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    toks = _tokens(text)
    return {tuple(toks[k:k+n]) for k in range(max(0, len(toks) - n + 1))}


def audit(parent: dict, measurement: dict, specs: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    checks: list[dict] = []
    add = lambda name, ok, detail: checks.append({"check": name, "pass": bool(ok), "detail": detail})
    expected_cols = [f"{c}.{i}" for c in parent["rating_order"] for i in parent["constructs"][c]["indicators"]]
    add("432 texts", len(df) == 432, len(df))
    add("216 matched families", df.base_id.nunique() == 216, df.base_id.nunique())
    add("36 indicators", sorted(df.col.unique()) == sorted(expected_cols), df.col.nunique())
    g = df.groupby("base_id")
    add("paired poles", g.pole.apply(lambda x: sorted(x) == ["high", "low"]).all(), "high+low per family")
    add("shared context", g.context.nunique().eq(1).all(), "context invariant within pair")
    add("shared closing", g.closing.nunique().eq(1).all(), "closing invariant within pair")
    add("unique text", df.text.nunique() == len(df), df.text.nunique())
    add("six families per indicator", df.groupby("col").base_id.nunique().eq(6).all(),
        df.groupby("col").base_id.nunique().value_counts().to_dict())
    dc = df.drop_duplicates("base_id").groupby(["col", "domain"]).size().unstack(fill_value=0)
    add("domain coverage", set(dc.columns) == set(DOMAINS) and dc.min().min() >= 1 and dc.max().max() <= 2,
        f"range={dc.min().min()}..{dc.max().max()}")
    forbidden = []
    for r in df.itertuples():
        hits = [p for p in pilot.FORBIDDEN if re.search(p, r.text, re.I)]
        if hits:
            forbidden.append({"id": r.id, "issue": "construct_word_leakage", "detail": " | ".join(hits)})
    add("no construct-name leakage", not forbidden, f"{len(forbidden)} texts flagged")

    old = pilot.generate(parent)
    old_texts, old_evidence = set(old.text), set(old.evidence)
    exact = df[df.text.isin(old_texts) | df.evidence.isin(old_evidence)]
    add("no exact pilot reuse", exact.empty, f"{len(exact)} exact text/evidence matches")
    old_ng = set().union(*(_ngrams(x) for x in old.text))
    overlap_rows = []
    for r in df.itertuples():
        ng = _ngrams(r.text)
        shared = len(ng & old_ng)
        if shared:
            overlap_rows.append({"id": r.id, "issue": "pilot_8gram_overlap", "detail": str(shared)})
    add("no pilot 8-gram overlap", not overlap_rows, f"{len(overlap_rows)} texts flagged")

    # Pole-exclusive repeated tokens are review flags, not automatic failures:
    # some are the semantic content of the manipulation.
    lex_rows = []
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "that", "it", "was",
            "were", "is", "are", "my", "i", "he", "she", "they", "his", "her", "their", "this", "as", "at"}
    for col, gg in df.groupby("col", sort=False):
        counts = {}
        for pole in ("high", "low"):
            counts[pole] = Counter(t for s in gg.loc[gg.pole == pole, "evidence"] for t in set(_tokens(s)) if t not in stop)
        for pole, other in (("high", "low"), ("low", "high")):
            for token, n in counts[pole].items():
                if n >= 2 and counts[other][token] == 0:
                    lex_rows.append({"col": col, "pole": pole, "token": token, "families": n,
                                     "rate": n / 6, "review": "semantic cue or lexical shortcut"})

    issues = forbidden + overlap_rows
    issue_df = pd.DataFrame(issues, columns=["id", "issue", "detail"])
    lex_df = pd.DataFrame(lex_rows, columns=["col", "pole", "token", "families", "rate", "review"])
    return issue_df, lex_df, checks


def semantic_audit_template(df: pd.DataFrame) -> pd.DataFrame:
    base = df.sort_values(["construct", "indicator", "base_id", "pole"]).pivot(
        index=["base_id", "construct", "indicator", "family", "domain", "context", "closing"],
        columns="pole", values="evidence").reset_index()
    base.columns.name = None
    base["author_direction_check"] = "pass"
    base["author_single_manipulation_check"] = "pass"
    base["independent_semantic_review"] = "pending"
    base["review_target_present"] = ""
    base["review_direction_correct"] = ""
    base["review_only_target_changes"] = ""
    base["review_unintended_indicators"] = ""
    base["review_naturalness_1_5"] = ""
    base["review_notes"] = ""
    # Known design-sensitive strata are explicitly marked rather than hidden.
    mask = base.indicator.eq("uncertainty")
    base.loc[mask, "author_single_manipulation_check"] = "flag: constant probability/severity wording is artificial"
    mask = base.indicator.eq("exceeded_expectations")
    base.loc[mask, "author_single_manipulation_check"] = "flag: outcome/expectation separation requires independent review"
    return base


def manifests(parent: dict, measurement: dict, df: pd.DataFrame) -> dict:
    return {
        "version": measurement["version"], "status": measurement["status"],
        "parent_registry_sha256": sha256_file(PARENT),
        "expected_matrix_sha256": pilot.expected_sha256(parent),
        "measurement_registry_sha256": sha256_file(MEASUREMENT),
        "stimulus_source_sha256": {p.name: sha256_file(p) for p in STIM_FILES},
        "stimuli_sha256": sha256_frame(df, ["id", "text"]),
        "n_texts": len(df), "n_families": df.base_id.nunique(),
        "n_indicators": df.col.nunique(), "conditions": list(CONDITIONS),
        "causal_screening": measurement["causal_screening"],
    }


def export(out: Path) -> None:
    parent, measurement, specs = load()
    assert sha256_file(PARENT) == measurement["parent_registry_sha256"]
    assert pilot.expected_sha256(parent) == measurement["expected_matrix_sha256"]
    df = generate(parent, specs)
    issues, lex, checks = audit(parent, measurement, specs, df)
    failures = [x for x in checks if not x["pass"]]
    if failures:
        raise AssertionError(failures)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit").mkdir(exist_ok=True)
    df.to_csv(out / "heldout_stimuli.csv", index=False)
    semantic_audit_template(df).to_csv(out / "audit" / "semantic_review.csv", index=False)
    issues.to_csv(out / "audit" / "automated_issues.csv", index=False)
    lex.to_csv(out / "audit" / "pole_lexical_review.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "audit" / "structural_checks.csv", index=False)
    (out / "frozen_manifest.json").write_text(json.dumps(manifests(parent, measurement, df), indent=2) + "\n")
    print(f"exported {len(df)} texts / {df.base_id.nunique()} families; {len(lex)} lexical review flags")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["check", "export"])
    ap.add_argument("--out", default="reports/atlas_phase1")
    a = ap.parse_args()
    parent, measurement, specs = load()
    df = generate(parent, specs)
    _, _, checks = audit(parent, measurement, specs, df)
    failures = [x for x in checks if not x["pass"]]
    if failures:
        raise SystemExit(json.dumps(failures, indent=2))
    if a.action == "export":
        export(ROOT / a.out)
    else:
        print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
