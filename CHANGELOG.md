# Changelog

## 1.2.0 — 2026-10-26

- Add Cubert lentil calibration, FIT preprocessing, fixed-recipe training, locked-model full-frame evaluation and count summarisation with configurable paths.
- Add saved-logit spatial analysis and synthetic CPU contract tests.
- Preserve the existing pork models and six-model timing implementation.
- Document input assets and validation scope. Experiment results, checkpoint selection and software licence are unchanged.

## 1.1.0 — 2026-10-24

- Document the six-model timing as two matched-environment three-model execution blocks.
- Add a configurable 128-file-to-CPU-mask timing entry point, with all first-use, warmup and formal records retained.
- Add a default schedule-only mode, input/checkpoint identity checks and synthetic protocol tests.
- Clarify data/asset interfaces and the scope of released source code.
- Preserve the nine model definitions, labels and training/evaluation implementation from v1.0.0.

The measurement operations were checked against the archived timing scripts. Release validation uses lightweight synthetic CPU checks; no training, model forward or new hardware timing was performed for this update.

## 1.0.0 — 2026-10-23

Initial source-code release with model definitions, labels, training/evaluation entry points and the original three-model timing interface.
