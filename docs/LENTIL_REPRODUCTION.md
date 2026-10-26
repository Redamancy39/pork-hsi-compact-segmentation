# Cross-day lentil replication

The lentil workflow uses dataset-specific fitting and training with the published Cubert data. It does not apply pork-trained weights to a new sensor.

## Inputs and environment

Obtain the source containers and COCO metadata from [Cubert's public dataset](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils), revision `ce3a213e96f83b0b2499514c09a661e52332110e`. Retain the provider's `data/day*/...` relative paths below both the container root and metadata root. Containers use `.cu3s`; the corresponding annotations use `.json`. Install the provider's Cuvis SDK and its Python binding separately. The recorded implementation used binding 3.6.1.0, host processing with four threads, and session white/dark references in reflectance mode. Native reflectance values are converted to FP32 without clipping to [0,1].

Use `lentil_replication/research_members.csv` from Online Resource 2 as the membership input. Its 660 rows specify 340 FIT frames, 60 DEV frames and 260 held-out-date frames; four lighting frames share a scene group. Research-role aliases in the runner preserve the executed internal role names. The exported manuscript manifest records the superseded proposal status separately from the completed execution status.

Install `requirements.txt`, an official PyTorch 2.7.0 CUDA build, and `requirements-cuvis.txt`. The executed environment was Windows 11, Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.0+cu128, RTX 5070 Ti (16 GB). CUDA/BF16 is required by the execution path. Set `cuvis_bin` if the native SDK libraries are not discoverable. No SDK binaries are bundled.

Copy `configs/lentil.example.json` outside this checkout and set its four input/work paths. The work directory must be new or belong to the same locked inputs. Plan inspection is the default:

```sh
python src/lentil_workflow.py prepare --config /path/to/lentil.json
```

Each execution stage is a separate process; this also bounds the Cuvis native runtime lifetime:

```sh
python src/lentil_workflow.py prepare --config /path/to/lentil.json --execute
python src/lentil_workflow.py train --config /path/to/lentil.json --execute
python src/lentil_workflow.py evaluate --config /path/to/lentil.json --execute
python src/lentil_workflow.py summarise --config /path/to/lentil.json --execute
```

Preparation reads FIT/DEV only, generates deterministic FIT crop/spectrum coordinates, fits population standardisation and a non-whitened PCA16 transform on 348,160 FIT spectra, and stores projected caches. A complete flushed FIT cache can be checked and reused with `prepare --recover-fit-cache --execute`; this option never changes the frozen PCA. Keep adequate disk space for raw containers, caches and saved logits: logits alone occupy about 9 GB for the eight-model held-out evaluation.

## Fixed training and evaluation

Four model families × seeds 42/43 give eight runs: B1, PCA16 U-Net, PCA16 Transformer and All-d1. Every run has 6000 updates, batch 64 and the same 8192 FIT crop coordinates with deterministic sampling/flips. B1/All-d1 use Adam at 0.001, U-Net Adam at 0.0003, and the Transformer AdamW at 0.001 with its recorded warmup/polynomial schedule. No learning-rate search is performed on lentils. Optimisation, initialisation, loss and metric bodies are carried over from the completed research run.

DEV selection evaluates all 60 frames every 1000 updates at logit zero. Frame Dice is averaged within scene groups and then across foreground-containing groups; exact ties retain the earlier checkpoint. Empty-annotation groups do not contribute a Dice of one. The evaluation command requires all eight checkpoint identities and preprocessing to be locked before reading held-out containers. It retains every model's scores.

Full-frame prediction unfolds 20 × 16 windows at stride 10 × 8, averages overlapping logits on the 1000 × 1080 grid, then applies logit ≥ 0. Lentil ID0 is background; IDs 1–7 are foreground. This differs from pork ID0, which is foreground. All full-frame background denominators count each pixel once. Group macros, class recalls and positive/zero-region background counts are saved separately.

The runner retains the recorded two-hour preparation and five-hour training/evaluation limits. Slower hardware can stop before completion; progress and completed outputs remain in the external work directory. Saved timing fields measure cached PCA features through CPU logits, with source reading and PCA shared across models. They are not a raw-container-to-mask benchmark.

## Release checks and scope

Synthetic CPU checks cover polygon rasterisation, PCA dimensions, labels, group separation, path boundaries, overlap reconstruction, model shapes and default dry runs. Locally, all eight deterministic initial states matched archived tensors exactly, and the portable summariser reproduced all eight saved run summaries. The numerical bodies of 20 model/metric/preparation/training functions or classes were checked against the executed source. The portable package has not undergone a new complete SDK/GPU experiment replay.

The public source includes pork and lentil execution paths and saved-logit spatial analysis. The oat replication remains documented by its methods, results and figures in Online Resources 1–2; its historical canonical-cube workspace and fitted assets are not packaged as a standalone workflow here. The repository does not claim complete reproduction of every historical experiment from its source archive alone.
