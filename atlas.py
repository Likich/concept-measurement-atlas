"""Measurement-atlas pilot: rate every stimulus text for all 12 constructs (atlas/pilot_registry.yaml).

Rating format: 0 = not enough evidence in the text; 1 = very low or clearly absent; 2-4 increasing; 5 = very high.
Scores are expected values over 1-5 from the next-token digit probabilities (as in validity.py); the probability of
answering 0 (abstention) and the total probability mass on digits are saved separately.

Conditions
  simple      construct question + labelled scale
  definition  adds the concise operational definition from the frozen registry
  reversed    simple prompt with reversed anchors (1 = very high ... 5 = very low) on a balanced 25% subset of base
              pairs; recoded as 6 - score before comparison (raw score kept)

Resumable: each (condition, rating) is saved atomically to <out>/chunks/ and skipped if present. CUDA OOM halves the
batch size. --representation additionally records the final-input-token residual stream for the simple prompt,
averaged over the 12 rating prompts per text (rep_H.npy, [432, layers+1, d]).

Usage:  python atlas.py --model Qwen/Qwen2.5-7B-Instruct [--conditions simple definition reversed] [--load_4bit]
        python atlas.py --model Qwen/Qwen2.5-7B-Instruct --stimset valence --conditions simple   (valence control)
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

import atlas_vignettes as AV

CONDITIONS = ["simple", "definition", "reversed"]


def build_prompt(reg, rating, text, cond):
    c = reg["constructs"][rating]
    lo, hi = c["anchors"]["low"], c["anchors"]["high"]
    if cond == "reversed":
        scale = f"0 (not enough evidence in the text), 1 ({hi}), 2 to 4 (decreasing levels) to 5 ({lo})"
    else:
        scale = f"0 (not enough evidence in the text), 1 ({lo}), 2 to 4 (increasing levels) to 5 ({hi})"
    tail = f"{c['question']} Answer on a scale from {scale}. Reply with a single digit."
    if cond == "definition":
        return f"You are coding texts for a social-science study.\n{c['definition']}\n\nText: {text}\n\n{tail}"
    return f"Read the text below.\n\nText: {text}\n\n{tail}"


def recode_reversed(score):
    """Map a reversed-anchor expected score (1 = very high) back to the standard orientation (5 = very high)."""
    return 6.0 - np.asarray(score, dtype=float)


def texts_for(df, cond):
    return df[df.base_id.isin(AV.reversed_subset(df))] if cond == "reversed" else df


def forward_safe(V, model, tok, prompts, ids, bs, hidden=False, log=print):
    import torch
    while True:
        try:
            return V.forward(model, tok, prompts, ids, hidden=hidden, bs=bs) + (bs,)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            log(f"CUDA OOM: retrying with batch size {bs}")


def write_atomic_csv(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run(a):
    import torch

    import validity as V

    reg = AV.load_registry()
    if a.stimset == "valence":  # valence control: 1,296 texts x (12 ratings + valence), simple prompt only
        import atlas_valence as AVal
        assert a.conditions == ["simple"] and not a.representation, "valence stimset: simple prompt only"
        reg, ratings = AVal.registry_with_valence(reg)
        stim = AVal.generate(reg)
        assert all(ok for _, ok, _ in AVal.checks(stim)), "stimulus checks failed: run python atlas_valence.py --check"
    else:
        ratings = reg["rating_order"]
        stim = AV.generate(reg)
        assert all(ok for _, ok, _ in AV.checks(stim, reg)), "stimulus checks failed: run python atlas_vignettes.py --check"
    out = os.path.join(a.out, a.model.split("/")[-1])
    os.makedirs(os.path.join(out, "chunks"), exist_ok=True)
    status = lambda s: open(os.path.join(out, "STATUS"), "w").write(f"{time.strftime('%H:%M:%S')} {s}\n")
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    t0 = time.time()
    tok, model = V.load(a.model, a.dtype, a.load_8bit, a.load_4bit)
    ids = V.digit_ids(tok)
    bs = a.bs
    n_prompts = 0
    for cond in a.conditions:
        sub = texts_for(stim, cond)
        for rating in ratings:
            path = os.path.join(out, "chunks", f"{cond}__{rating}.csv")
            if os.path.exists(path):
                continue
            status(f"scoring {cond} {rating}")
            prompts = V.chat(tok, [build_prompt(reg, rating, t, cond) for t in sub.text])
            P, _, bs = forward_safe(V, model, tok, prompts, ids, bs, log=log)
            s, abst, mass = V.scores(P)
            chunk = pd.DataFrame(dict(id=sub.id.values, condition=cond, rating=rating, raw_score=s,
                                      score=recode_reversed(s) if cond == "reversed" else s, abstain=abst, mass=mass))
            write_atomic_csv(chunk, path)
            n_prompts += len(prompts)
            log(f"{cond:10s} {rating:16s} n={len(prompts)} mass {mass.mean():.3f} abstain {abst.mean():.3f} bs {bs}")
        open(os.path.join(out, f"DONE_{cond}"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))

    if a.representation and not os.path.exists(os.path.join(out, "rep_H.npy")):
        status("representation: simple prompts with hidden states")
        acc = None
        for rating in reg["rating_order"]:
            prompts = V.chat(tok, [build_prompt(reg, rating, t, "simple") for t in stim.text])
            _, H, bs = forward_safe(V, model, tok, prompts, ids, bs, hidden=True, log=log)
            acc = H if acc is None else acc + H
            log(f"representation {rating} done")
        np.save(os.path.join(out, "rep_H.npy.tmp.npy"), (acc / len(reg["rating_order"])).astype(np.float32))
        os.replace(os.path.join(out, "rep_H.npy.tmp.npy"), os.path.join(out, "rep_H.npy"))
        stim[["id", "base_id", "col", "pole", "domain"]].to_csv(os.path.join(out, "rep_index.csv"), index=False)
        open(os.path.join(out, "DONE_representation"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))

    # merge, summarise, manifest
    chunks = [pd.read_csv(os.path.join(out, "chunks", f)) for f in sorted(os.listdir(os.path.join(out, "chunks")))
              if f.endswith(".csv")]
    scores = pd.concat(chunks, ignore_index=True)
    write_atomic_csv(scores, os.path.join(out, "scores.csv"))
    summ = scores.groupby(["condition", "rating"]).agg(n=("score", "size"), mean_score=("score", "mean"),
                                                       mean_abstain=("abstain", "mean"), mean_mass=("mass", "mean"))
    summary = {"model": a.model, "conditions_done": sorted(scores.condition.unique()),
               "by_condition_rating": {f"{c}|{r}": v for (c, r), v in summ.round(4).to_dict("index").items()},
               "min_digit_mass": float(scores["mass"].min())}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    prev = json.load(open(os.path.join(out, "manifest.json"))) if os.path.exists(os.path.join(out, "manifest.json")) else {}
    manifest = dict(prev, model=a.model, dtype=a.dtype, quantized="4-bit NF4" if a.load_4bit else ("8-bit" if a.load_8bit else None),
                    registry_sha256=AV.file_sha256(), expected_matrix_sha256=AV.expected_sha256(reg),
                    stimuli_sha256=AV.stimuli_sha256(stim), registry_version=reg["version"], stimset=a.stimset,
                    **({"valence_spec_sha256": AVal.spec_sha256()} if a.stimset == "valence" else {}),
                    conditions=sorted(scores.condition.unique()), n_scores=int(len(scores)),
                    prompts_this_session=n_prompts, final_batch_size=bs,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    torch=torch.__version__, python=platform.python_version(),
                    seconds_this_session=round(time.time() - t0, 1),
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    if all(os.path.exists(os.path.join(out, f"DONE_{c}")) for c in a.conditions):
        open(os.path.join(out, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    status("done")
    log(f"finished {a.model}: {len(scores)} scores in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--load_8bit", action="store_true")
    ap.add_argument("--representation", action="store_true")
    ap.add_argument("--stimset", default="pilot", choices=["pilot", "valence"])
    ap.add_argument("--out", default=None, help="default results_atlas/pilot_v1 or results_atlas/valence_v1")
    a = ap.parse_args()
    a.out = a.out or ("results_atlas/valence_v1" if a.stimset == "valence" else "results_atlas/pilot_v1")
    run(a)


if __name__ == "__main__":
    sys.exit(main())
