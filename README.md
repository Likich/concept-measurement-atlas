# Concept Measurement Atlas

**Do LLM ratings of abstract concepts measure what they claim to measure?**

When a language model rates a text for "loneliness" or "trust", the score looks like a measurement. It may really be a
proxy, such as how positive or negative the text sounds. This project tests that with methods from psychological
measurement: each concept is broken into theory-derived indicators, those indicators are manipulated one at a time,
and the model's ratings are checked against expectations fixed in advance.

*Research in progress. A paper is in preparation, so this repository shares the design and code, not the results.*

## Design

- **12 constructs:** anger, sadness, fear, loneliness, social isolation, belonging, satisfaction, trust, fairness,
  blame, intentionality, political stance.
- **36 indicators**, three per construct, each tied to published theory (sources are in the registry).
- **432 controlled texts:** 36 indicators × 6 text families × 2 poles (indicator high or low).
- **4 prompting conditions:** the construct's name only, an inclusion definition, inclusions plus exclusions, and
  indicator-by-indicator rating.
- **A frozen expectation matrix:** which rating should move for each indicator, marked strong, weak, null or
  unspecified. It is fixed before any model runs.

## Methods

- Family-level paired bootstrap, shared across ratings and conditions, so comparisons respect how the texts were built
- Noise-corrected estimates of unjustified sensitivity, with Holm correction for multiple tests
- Pre-specified correction criteria for deciding whether a prompt really improves measurement
- Expected-score readout from the model's digit probabilities, including a "not enough evidence" option
- Reproducibility checks: pinned model revisions, hashed stimuli, run manifests

## Repository

| Path | Contents |
|---|---|
| `atlas/pilot_registry.yaml` | Constructs, indicators, theory sources, expectations |
| `atlas/atlas_v2/` | Measurement registry and stimuli |
| `atlas_phase1.py` | Stimulus validation, prompt construction, experiment design |
| `atlas_phase1_run.py` | Model scoring on one GPU |
| `atlas_phase1_analysis.py` | Effect estimates, bootstrap, multiple-testing correction |
| `atlas_phase1_simulate.py` | Simulation of correction margins (needs stored pilot results, not included here) |
| `atlas_phase2_validate.py` | Output validation |
| `atlas.py`, `atlas_vignettes.py`, `atlas_analysis.py` | Pilot-stage modules these scripts build on |
| `test_atlas_phase1.py` | Tests |

## Run the tests

No GPU or model download is needed.

```bash
pip install numpy pandas scipy pyyaml pytest
python -m pytest test_atlas_phase1.py
```

## Author

Angelina Parfenova, Lucerne University of Applied Sciences and Arts (HSLU)

All rights reserved until the paper is published.
