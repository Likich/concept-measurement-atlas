"""Frozen single-GPU runner for the atlas_v2 A--D behavioral comparison.

This file is prepared in Phase 1 but must not be executed until the Phase-2/3
approval gates. It never uses more than one visible GPU.

Example (after approval):
  CUDA_VISIBLE_DEVICES=0 python atlas_phase1_run.py --model Qwen/Qwen2.5-7B-Instruct --pilot
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

import atlas_phase1 as P

# Pinned model revisions (pre-run correction C4): the run fails if the loaded revision differs.
REVISIONS = {
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
}
APPROVED_GPUS = ("H100", "A6000")  # one device only; the H100 was approved by the user on 2026-11-04


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run(a) -> None:
    # Must happen before torch/transformers import. An explicit CLI choice is
    # converted into one visible device; multi-GPU and device_map="auto" are not used.
    parent, measurement, specs = P.load()
    stimuli = P.generate(parent, specs)
    if a.pilot:
        # Frozen technical pilot: first family of the selected indicator from
        # each construct, both poles (24 texts). It checks execution, formatting,
        # and direction only; it is not used for prompt tuning or paper estimates.
        selected = {f"{c}.{i}" for c, i in measurement["causal_screening"]["indicators"].items()}
        stimuli = stimuli[stimuli.col.isin(selected) & stimuli.base_id.str.endswith("h0")].copy()
    ratings = parent["rating_order"]
    jobs = []
    for condition in a.conditions:
        if condition == "D_indicators":
            jobs.extend((condition, rating, indicator) for rating in ratings
                        for indicator in parent["constructs"][rating]["indicators"])
        else:
            jobs.extend((condition, rating, None) for rating in ratings)
    manifest = P.manifests(parent, measurement, P.generate(parent, specs))
    import hashlib
    raw_prompt_lines = []
    for condition, rating, indicator in jobs:
        for row in stimuli.itertuples():
            prompt = P.build_prompt(parent, measurement, rating, row.text, condition, indicator)
            raw_prompt_lines.append("\t".join((row.id, condition, rating, indicator or "", prompt.replace("\n", "\\n"))))
    prompt_hash = hashlib.sha256("\n".join(raw_prompt_lines).encode()).hexdigest()
    if a.dry_run:
        result = dict(manifest, dry_run=True, model=a.model, pilot=bool(a.pilot),
                      maximum_visible_gpus=1, run_n_texts=len(stimuli),
                      run_n_families=stimuli.base_id.nunique(), run_n_indicators=stimuli.col.nunique(),
                      n_jobs=len(jobs),
                      n_prompts=len(raw_prompt_lines), prompt_sha256=prompt_hash)
        target = Path(a.dry_run_manifest)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    vis = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if len(vis) != 1:  # the launcher (run_atlas_v2.sh) picks one idle GPU; the runner never chooses one itself
        raise SystemExit("set CUDA_VISIBLE_DEVICES to exactly one idle GPU (use run_atlas_v2.sh)")
    import torch
    import transformers
    import validity as V
    import atlas as old_runner
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("atlas_v2 requires exactly one visible CUDA device")
    gpu_name = torch.cuda.get_device_name(0)
    if not any(g in gpu_name.upper() for g in APPROVED_GPUS):
        raise RuntimeError(f"GPU {gpu_name} is not an approved device {APPROVED_GPUS}")
    revision = a.revision or REVISIONS.get(a.model)
    if revision is None:
        raise SystemExit(f"no pinned revision for {a.model}; pass --revision")
    out = Path(a.out) / a.model.split("/")[-1]
    chunks = out / "chunks"; chunks.mkdir(parents=True, exist_ok=True)
    # validity.load with an explicit revision (same tokenizer and model settings)
    tok = AutoTokenizer.from_pretrained(a.model, revision=revision, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=revision, dtype=getattr(torch, a.dtype)).to("cuda:0").eval()
    if getattr(model.config, "_commit_hash", revision) != revision:
        raise RuntimeError(f"loaded revision {model.config._commit_hash} != pinned {revision}")
    ids = V.digit_ids(tok)
    bs, n_prompts, started = a.bs, 0, time.time()
    for condition, rating, indicator in jobs:
        suffix = f"__{indicator}" if indicator else ""
        path = chunks / f"{condition}__{rating}{suffix}.csv"
        if path.exists():
            continue
        prompts = [P.build_prompt(parent, measurement, rating, text, condition, indicator)
                   for text in stimuli.text]
        prompts = V.chat(tok, prompts)
        probs, _, bs = old_runner.forward_safe(V, model, tok, prompts, ids, bs, log=print)
        score, abstain, mass = V.scores(probs)
        response_digit = probs.argmax(1)
        row = pd.DataFrame({
            "model": a.model, "id": stimuli.id.values, "condition": condition,
            "rating": rating, "indicator": indicator, "score": score,
            "abstain": abstain, "mass": mass, "response_digit": response_digit,
        })
        atomic_csv(row, path)
        n_prompts += len(prompts)
        print(f"{condition} {rating} {indicator or '-'} n={len(row)} mass={mass.mean():.3f} bs={bs}", flush=True)

    frames = [pd.read_csv(path) for path in sorted(chunks.glob("*.csv"))]
    scores = pd.concat(frames, ignore_index=True)
    atomic_csv(scores, out / "scores.csv")
    run_manifest = dict(
        manifest, model=a.model, model_revision=getattr(model.config, "_commit_hash", None),
        dtype=a.dtype, quantization="4-bit NF4" if a.load_4bit else ("8-bit" if a.load_8bit else None),
        gpu_index_requested=a.gpu, visible_gpu_count=torch.cuda.device_count(),
        gpu_name=torch.cuda.get_device_name(0), pilot=bool(a.pilot), conditions=a.conditions,
        maximum_visible_gpus=1, run_n_texts=len(stimuli),
        run_n_families=stimuli.base_id.nunique(), run_n_indicators=stimuli.col.nunique(),
        prompt_sha256=prompt_hash,
        prompts_this_session=n_prompts, n_scores=len(scores), final_batch_size=bs,
        seconds_this_session=round(time.time() - started, 2), python=platform.python_version(),
        torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_uuid_and_driver=os.popen("nvidia-smi --query-gpu=uuid,driver_version --format=csv,noheader -i 0").read().strip(),
        pinned_revision=revision, decoding="single forward pass, next-token digit probabilities (no sampling)",
        git=json.loads(Path(".git_commit.json").read_text()) if Path(".git_commit.json").exists() else None,
        finished=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    (out / "manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", nargs="+", choices=P.CONDITIONS, default=list(P.CONDITIONS))
    ap.add_argument("--gpu", type=int, default=0, help="ignored: the launcher sets CUDA_VISIBLE_DEVICES to one idle GPU")
    ap.add_argument("--revision", help="model revision (default: the pinned one in REVISIONS)")
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--load_8bit", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build and hash prompts without importing torch")
    ap.add_argument("--dry-run-manifest", default="reports/atlas_phase2/dry_run_manifest.json")
    ap.add_argument("--out", default="results_atlas/atlas_v2")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
