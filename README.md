# Compact hyperspectral segmentation of foreign materials on pork

**Code-only release** accompanying *Hyperspectral localisation of foreign materials on pork surfaces using a compact residual segmentation model*.

The workflow uses all 184 measured hyperspectral bands. A FIT-derived standardiser and frozen PCA16 projection feed a compact 40-channel residual network that preserves the 20×16 patch grid. PCA16 is a mathematical projection, not a 16-band camera specification.

## Code

- `src/models.py`: compact PCA16 U-Net (`B0`), compact residual (`B1`), adapted raw184 pixel Transformer (`T`), raw184 pointwise MLP, PCA16 depthwise-separable residual, raw184 U-Net, all-dilation-one/no-identity residual controls and same-width PCA16 Transformer. It also contains the per-patch class-balanced BCE plus masked Dice loss and source-level pixel metrics.
- `src/labels.py`: exact class-ID/RGB decoding. IDs 0–9 are foreign-material foreground; MEAT/FAT/CB (10–12) are background. Unknown IDs/colours remain ignored; ID0 is foreground.
- `src/workflow.py`: portable single-run training, fixed-member evaluation and original-three-model timing entry points. Inputs are supplied explicitly by the user; importing the module starts no job.
- `tests/test_code.py`: synthetic CPU tests without model forwards or optimiser updates.

This repository contains **no raw data, complete masks, trained/initial checkpoints, numerical result tables, member lists, fitted preprocessing, sampling plans, unpublished manuscript or private research records**. Those remain local or with the original data providers. Public source access does not make the complete experimental asset bundle available.

## Dependencies and checks

Use Python 3.10 or later. Install an official, platform-appropriate PyTorch 2.7.0 build, then:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/workflow.py --help
```

The dependency pins describe the packaging environment; no clean-environment installation is claimed. Current checks cover synthetic label/ignore behaviour, hand-counted pixel metrics and loss, nine model definitions/parameter counts, and path safety. They do not train, run a model forward, read original data or validate throughput. Local archived-initial-state comparisons were also performed before publication; the compared tensors and records are not distributed.

## Data and external assets

The public pork dataset, version 1.0.0, is distributed separately at [Zenodo](https://doi.org/10.5281/zenodo.17242553), with attribution to Ghimpeteanu and colleagues; the associated paper is [doi:10.3390/s25227015](https://doi.org/10.3390/s25227015). Obtain and use data according to the provider's CC BY 4.0 terms. This repository does not download data.

The CLI requires `--data-root` containing `training_dataset_20x16/` and a separate `--assets-root` prepared by the user:

```text
assets-root/
  artifacts/preprocessing.npz
  artifacts/projection_biases.json       # optional historical centring biases
  artifacts/plans/schedule_<seed>.npy
  artifacts/plans/flips_<seed>.npy
  manifests/fit_dev.csv
  manifests/comparative.csv
  manifests/timing128.csv
  manifests/label_mapping.json
```

`preprocessing.npz` contains FIT-derived `mean`, `scale` and `pca_mean` vectors of length 184, a 16×184 `components` matrix, and the fitted explained variance. `load_preprocessing(assets_root)` loads these frozen statistics; optional historical FP32 centring biases avoid recalculating a near-zero matrix product with different BLAS rounding. Nothing is refitted by this code.

Manifests use `parent`, `image_member`, `label_member`, and, for fitting/development/timing, `role`. File members are relative to the data root. Labels may be ID arrays or exact published RGB colours. The JSON mapping uses `original_id_to_rgb`, `derived_foreground` and `derived_background`, as consumed by `labels.py`; users must obtain the real colour mapping from the source annotations, not guess it from appearance. The original study used 7,168 FIT, 1,024 DEV, 2,048 already-observed comparison patches and 128 timing inputs. Sampling and flip arrays have shape 6000×64. These descriptions define interfaces; exact source members and fitted arrays are intentionally not published in this code-only release.

## Execution examples

The following commands are for future execution with separately supplied assets and compatible CUDA/BF16 support:

```sh
python src/workflow.py train --data-root /path/to/data --assets-root /path/to/assets --output /path/to/new-run --family B1 --seed 42 --learning-rate 0.001 --execute
python src/workflow.py evaluate --data-root /path/to/data --assets-root /path/to/assets --output /path/to/new-evaluation --family B1 --seed 42 --checkpoint /path/to/dice_best.pt --execute
python src/workflow.py time --data-root /path/to/data --assets-root /path/to/assets --output /path/to/new-timing --checkpoint-dir /path/to/three-checkpoints --execute
```

Output directories must be new and outside the source directory. The timing command expects `B0_42.pt`, `B1_42.pt` and `T_42.pt`. Checkpoints and assets are not publicly downloadable from this repository. The reorganised complete train/evaluate/timing workflow has not been replayed end to end.

Training runs one recipe, with 6,000 updates, effective batch 64, FP32 preprocessing/loss and BF16 network execution. It evaluates DEV every 200 updates and retains the earlier exact tie when selecting the largest DEV source-macro Dice at logit zero. The paper's finite rate selection used DEV seeds 42/43, then repeated the chosen recipe at seeds 44/45; the two structural controls used the residual model's fixed recipe. This CLI does not launch a new search automatically. Freeze models and selection rules before evaluating comparison members.

Parent source means a filename prefix, not a verified meat batch, physical object or acquisition session. Overlapping patches can repeat source pixels. Training seeds are not independent acquisitions, and previously observed comparison sources are not a new confirmation set.

## Timing and applicability

Analytical MACs, device execution and the warm 128-file NPY-to-CPU-mask interval have distinct scopes. The historical matched timing compares the original PCA16 U-Net, residual and raw184 adapted Transformer only; the added configurations were not re-timed. The recorded Transformer measurements used a mathematical attention fallback; default legal SDPA backend selection is retained in this portable code. Runtime on another backend or machine is not an equivalent remeasurement of the paper's record. Acquisition and physical removal are outside the file interval.

The manuscript retains material omissions, meat/fat false positives, and a separate oat replication with lower Dice/slower full-file processing relative to U-Net. This executable package concerns pork; neither oat assets nor unpublished numerical tables are distributed here.

## Attribution and software licence

See `THIRD_PARTY_NOTICES.md` for source and architectural attribution. Dependencies are installed separately and retain their own licences. The authors have not assigned an additional software reuse licence; copyright is retained. No blanket MIT or Apache licence is implied for study-authored code or source datasets.

Version **1.0.0** is identified by its Git tag/commit and `CITATION.cff`. No article DOI or scholarly archive DOI is invented. Source-code access and complete experimental reproducibility are separate scopes.
