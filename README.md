# Compact hyperspectral segmentation of foreign materials on pork

Source code accompanying *Hyperspectral localisation of foreign materials on pork surfaces using a compact residual segmentation model*.

The workflow uses all 184 measured bands. A FIT-derived standardiser and frozen PCA16 projection feed a 40-channel residual network on a 20 × 16 patch grid. The comparison includes nine model configurations and tissue-specific error metrics for meat and fat.

## Installation and lightweight checks

Use Python 3.11 or later with an official platform-appropriate PyTorch 2.7.0 build:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/workflow.py --help
python src/file_timing.py --config configs/file_timing.example.json
```

The last command prints the timing schedule. Actual processing requires the explicit `--execute` flag and separately supplied inputs. Tests use synthetic CPU data to check labels, metrics, model construction, parameter counts, sampling, file paths and timing schedules. Lentil tests also run small synthetic CPU forwards without updating weights. The portable training and timing commands have not been replayed end to end as a release test.

## Models and entry points

| Family ID | Configuration |
|---|---|
| `B0` | PCA16 compact U-Net |
| `B1` | PCA16 residual, 40 channels, block dilation 1/1/2/2 |
| `T` | Adapted raw184 pixel Transformer |
| `RAW184_MLP` | Standardised raw184 pointwise MLP |
| `PCA16_DS_RESIDUAL` | Depthwise-separable residual |
| `RAW184_UNET` | Standardised raw184 compact U-Net |
| `B1_ALL_D1` | Residual with all block dilations set to one |
| `B1_NO_SKIP` | Residual without identity additions |
| `PCA16_T_SAME_WIDTH` | PCA16 Transformer with the same attention width as `T` |

- `src/models.py`: model definitions, frozen preprocessing, training loss and source-level pixel metrics.
- `src/labels.py`: class-ID/RGB decoding. IDs 0–9 are foreground, IDs 10–12 are meat/fat/conveyor-belt background, and unknown labels remain ignored. ID0 is foreground.
- `src/workflow.py`: single-run training, fixed-member evaluation and the preceding three-model timing entry point, retained for compatibility.
- `src/file_timing.py`: the two-triplet file-to-mask timing protocol corresponding to the manuscript's six-model comparison.
- `configs/file_timing.example.json`: editable path/hash example, without experiment members or fitted values.

Version 1.2.0 preserves the pork model, label, training and evaluation implementations from v1.0.0 and the six-model timing from v1.1.0. See [CHANGELOG.md](CHANGELOG.md) for the publication update.

## Lentil replication and saved-logit spatial analysis

Version 1.2.0 adds lentil readback/calibration, FIT-only standardisation/PCA, fixed eight-run training, full-frame window stitching, held-out-date evaluation and saved-count summarisation. See [the lentil reproduction guide](docs/LENTIL_REPRODUCTION.md) and `configs/lentil.example.json`. The default plan command does not read raw containers.

`src/analyse_saved_spatial.py` reproduces patch-local foreground/background transitions and meat/fat interface strata from externally supplied saved class IDs and logits. Configure all four models and four seeds with `configs/spatial_analysis.example.json`, then run `python src/analyse_saved_spatial.py --config /path/to/spatial.json --out /path/to/analysis`. It checks identities and original source counts before writing results and does not load a network.

## Public data and experiment inputs

The public pork dataset, version 1.0.0, is available from [Zenodo](https://doi.org/10.5281/zenodo.17242553). Cite Ghimpeteanu and colleagues' [originating study](https://doi.org/10.3390/s25227015) and follow the dataset's CC BY 4.0 terms. Provide the extracted dataset under a local data root; members in manifests are relative paths such as those under `training_dataset_20x16/`.

The training/evaluation commands expect a separately supplied asset directory:

```text
assets-root/
  artifacts/preprocessing.npz
  artifacts/projection_biases.json       # optional archived FP32 centring biases
  artifacts/plans/schedule_<seed>.npy
  artifacts/plans/flips_<seed>.npy
  manifests/fit_dev.csv
  manifests/comparative.csv
  manifests/timing128.csv
  manifests/label_mapping.json
```

`preprocessing.npz` supplies frozen FIT statistics: `mean`, `scale` and `pca_mean` vectors of length 184, a 16 × 184 `components` matrix and `explained_variance`. The optional bias JSON maps `family_seed` to a 16-element archived projection bias. Preprocessing is loaded, not fitted, by these entry points.

Manifests use `parent`, `image_member`, `label_member` and the relevant `role`. The label mapping supplies `original_id_to_rgb`, `derived_foreground` and `derived_background`; use the provider's actual mapping. Sampling and flip arrays have shape 6000 × 64. Checkpoint files contain a `model` state dictionary.

The research partitions comprise 7168 FIT patches, 1024 DEV patches and 2048 comparative patches. Parent source denotes the original-image filename prefix; it is not a verified batch or session ID. Overlapping patches can repeat source pixels. The comparison sources had already informed later model hypotheses; four seeds describe training randomness on those fixed members.

This release provides source code and interface examples. Exact recorded-run reproduction additionally requires the selected member lists, sampling plans, frozen preprocessing and checkpoints, which are not distributed in this repository. Supporting numerical tables are part of the manuscript's supplementary files.

## Training and evaluation

```sh
python src/workflow.py train --data-root /path/to/data --assets-root /path/to/assets --output /path/to/new-run --family B1 --seed 42 --learning-rate 0.001 --execute
python src/workflow.py evaluate --data-root /path/to/data --assets-root /path/to/assets --output /path/to/new-evaluation --family B1 --seed 42 --checkpoint /path/to/dice_best.pt --execute
```

Use a new output directory outside the source checkout and compatible CUDA/BF16 support. A training run uses 6000 updates, effective batch 64, FP32 preprocessing/loss and BF16 network execution. DEV is evaluated every 200 updates, selecting source-macro Dice at logit zero with earlier exact ties retained. The finite learning-rate comparison uses DEV seeds 42/43, followed by seeds 44/45 at the selected rate; the two structural controls inherit the residual recipe. Selection and model locking precede comparative evaluation.

## Six-model file-to-mask timing

The manuscript's current timing comprises two separate three-model execution blocks:

- **A:** `PCA16_T_SAME_WIDTH`, `PCA16_DS_RESIDUAL`, `B1_ALL_D1`.
- **B:** `B0`, `B1`, `T`.

Both use seed42 Dice-best checkpoints, the same 128 FIT inputs, batches 1/8/64, and three cyclic model orders. Each formal pass is preceded by one complete-input warmup; each model's first condition use is also recorded. A block therefore records 27 formal, 27 warmup and nine first-use passes. The two blocks are not a six-model fully interleaved or paired-repeat experiment.

Copy the example JSON to a working directory, set real paths and SHA256 hashes, and select a block. For block B, replace the checkpoint keys with `B0`, `B1` and `T`. Relative paths resolve against the configuration file's directory. Inspect the schedule before execution:

```sh
python src/file_timing.py --config /path/to/block_a.json
python src/file_timing.py --config /path/to/block_a.json --output /path/to/new-timing-a --execute
python src/file_timing.py --config /path/to/block_b.json --output /path/to/new-timing-b --execute
```

The measured interval includes NPY reading, FP32 CHW conversion, host-to-device transfer, frozen preprocessing, BF16 prediction, logit-zero masks, CPU return and concatenation. Models are already loaded and files are warm. Acquisition, model loading, mask writes and downstream handling are outside the interval. CUDA-event sums are recorded separately from complete-file wall time. A 128-file total divided by 128 is not a measured single-patch response latency.

The recorded environment was Windows 11, Ryzen 7 9700X, RTX 5070 Ti, Python 3.13.5, NumPy 2.1.3 and PyTorch 2.7.0+cu128, with four intra-op threads, TF32 disabled, cuDNN benchmarking disabled and default legal SDPA policy. Enabled SDPA flags identify policy, not a profiled selected kernel. Environment and input/checkpoint identities are written for each execution; compare these receipts before combining separately run blocks. Other machines or backends may produce different timings.

Outputs include every first-use/warmup/formal pass, three-pass condition medians and ranges, allocator peaks and completion status. The runner stops between conditions after a 600-second budget and retains partial records on failure. The archived manuscript measurements used identical `host`, `file_run` and `measure` operations; this release adds configurable paths, validation and receipts around those operations.

The older `workflow.py time` entry remains a separate original-three-model device/file protocol. Its earlier environment and measurements should not be pooled with the current six-model file-workflow results.

## Attribution and software licence

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for architectural attribution and dependency licences. Study-authored source remains copyright-retained without an additional software reuse licence. Dataset and dependency terms apply separately.

Use the versioned Git tag and `CITATION.cff` when citing the code. A scholarly archive DOI has not been assigned.
