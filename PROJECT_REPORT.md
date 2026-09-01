# AI Image Colorizer — Project Report

**Generated:** 2026-09-01
**Last updated:** 2026-09-01 (refreshed after adding evaluation, benchmarking, REST API, experiment tracking, and Docker)
**Repository:** ImageColoriser (git, branch `main`, 1 commit: `21b5682 Initial commit`)

## 1. Summary

AI Image Colorizer adds color to grayscale photographs using a pretrained deep
convolutional model (Zhang, Isola & Efros, *Colorful Image Colorization*, ECCV
2016), run through OpenCV's DNN module on a Caffe checkpoint. It ships three
interfaces — a Tkinter desktop app, a Streamlit web app, and a FastAPI REST
service — built on a single shared core library (`colorizer/`), plus a
PSNR/SSIM(/LPIPS) evaluation pipeline, CPU/GPU latency benchmarking,
SQLite-backed run + experiment history, a 54-test unittest suite, CI, and
optional Docker support.

## 2. Demo Walkthrough (Step-by-Step)

A script for showing this project to someone else, end to end. Run everything
from the repository root. Commands are PowerShell (this project's primary
shell); a Bash equivalent is noted where it differs.

### 2.0 Prerequisites (one-time)

- Python 3.11+ and the `Model/` folder populated with `colorization_deploy_v2.prototxt`,
  `colorization_release_v2.caffemodel`, and `pts_in_hull.npy` (already present in this repo).
- Dependencies installed into the project's virtual environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2.1 Prove it works: run the test suite

Fast, no model weights or GPU required — good opener to show the project is tested.

```powershell
python -m unittest discover -s tests
```

Expect: `Ran 54 tests in ~2s — OK`.

### 2.2 Desktop app (Tkinter)

```powershell
python main.py
```

What to show: the model loads in the background (status label, controls
disabled until ready) → **Open Image** → **Colorize** → side-by-side
original/colorized preview → **Save Result**. Then **Batch Colorize Folder**
against `images\batch\` to show folder-level processing with a progress
status and a summary dialog (succeeded/failed counts).

### 2.3 Web app (Streamlit)

```powershell
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Walk through the three tabs:

1. **Single image studio** — upload an image (e.g. from `images\batch\`),
   click **Colorize image**, show the result and the **Download PNG**
   button, then drag the before/after comparison slider.
2. **Batch workspace** — upload several images at once, click **Colorize
   all**, show the progress bar, the **Download ZIP** button, the expandable
   before/after previews, and the processing-time summary (mean/median/P95,
   fastest/slowest file).
3. **Evaluation** — this is the ML-evaluation half of the demo (see §2.5 to
   seed a dataset first if you haven't). Pick a dataset and model(s), click
   **Run evaluation**, and show: the PSNR/SSIM/LPIPS/latency results table,
   the per-image metric distribution charts, the original/grayscale/
   prediction/difference example previews, and the experiment history table
   at the bottom (this is the SQLite-backed record of every past run).

### 2.4 REST API (FastAPI)

```powershell
uvicorn api.main:app --reload
```

Open `http://localhost:8000/docs` — FastAPI's interactive Swagger UI (opening
the bare `http://localhost:8000/` also redirects here, so there's no bare
404 to explain mid-demo). This is the easiest way to demo the API live:
expand `/colorize`, click **Try it out**, upload a file, and execute it
directly in the browser. No curl needed.

To demo from the command line instead (Windows: use `curl.exe` explicitly, not
the `curl` alias for `Invoke-WebRequest`):

```powershell
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/models
curl.exe -X POST http://localhost:8000/colorize -F "file=@images/batch/lion.jpg" -F "model=vibrant" -F "saturation=1.2" -o colorized.png
curl.exe http://localhost:8000/history
curl.exe http://localhost:8000/experiments
```

Talking points: `/colorize` returns `X-Model` and `X-Processing-Time`
response headers; invalid/oversized/unsupported uploads return `400`/`413`;
an unknown or unavailable model returns `404`; every check reuses
`colorizer/validation.py`, the same validation the Streamlit UI uses — no
duplicated logic between the two front ends.

### 2.5 Evaluation CLI

The dataset directory isn't committed to the repo (by design — see
`evaluation/README.md`), so seed one from the sample images already in the
repo for a quick demo:

```powershell
New-Item -ItemType Directory -Force evaluation\datasets\demo | Out-Null
Copy-Item images\batch\lion.jpg, images\batch\valley.jpg evaluation\datasets\demo\
```

Then run:

```powershell
python scripts/evaluate.py --dataset evaluation/datasets/demo --model vibrant
```

What to show: the console summary (PSNR/SSIM mean+median, LPIPS reported as
"unavailable" since it's an optional dependency, latency mean/median/P95),
the generated `evaluation/results/vibrant.csv` and `.json`, and that the run
now also shows up in the Streamlit Evaluation tab's experiment history (and
via `curl.exe http://localhost:8000/experiments`) — one SQLite table backing
all three surfaces.

### 2.6 Benchmark CLI

```powershell
python scripts/benchmark.py --model vibrant --images evaluation/datasets/demo --iterations 3
```

What to show: **model load time** reported separately from **inference
latency**; mean/median/P95 latency; images/sec throughput; and the `Device:`
line (CPU here, since no CUDA-capable OpenCV build is in use — the code path
exists but was never something we had to install or configure).

### 2.7 Docker (optional — skip if Docker isn't installed)

```powershell
docker compose up
```

Then open `http://localhost:8501` (same Streamlit app, now containerized).
For the API too: `docker compose --profile api up api` → `http://localhost:8000/docs`.
Talking point: Docker is entirely optional — everything above already worked
without it.

### 2.8 Feature checklist (what you just demoed)

- CNN colorization via OpenCV DNN, LAB color space, two selectable models
- Desktop (Tkinter) and web (Streamlit) UIs sharing one core library
- Single-image and batch processing, with ZIP export and a before/after slider
- Quantitative evaluation (PSNR/SSIM, optional LPIPS) against a real dataset
- CPU/GPU latency + throughput benchmarking with proper warm-up
- SQLite-backed run history *and* experiment tracking, visible from the UI, the API, and the CLI
- A REST API with interactive docs, input validation, and proper HTTP error codes
- 54 passing tests, CI on every push, optional Docker packaging

## 3. Architecture

```text
Image -> Normalize -> LAB -> Extract L channel -> CNN -> Predict AB channels
      -> Merge with original L -> Convert LAB to BGR -> Output image

Evaluation: color image -> grayscale -> colorize -> compare vs. original
                                                       (PSNR / SSIM / LPIPS)
```

```text
ImageColoriser/
├── colorizer/                Shared core library (importable package)
│   ├── __init__.py           Public API surface
│   ├── config.py             Paths & tunables, override-able via env vars
│   ├── models.py             Model registry, loading, GPU detection, caching
│   ├── pipeline.py           colorize_image(): the LAB colorization pipeline
│   ├── validation.py         Shared validation for untrusted image uploads
│   ├── evaluation.py         PSNR/SSIM/LPIPS evaluation against a dataset
│   ├── benchmarking.py       Latency/throughput benchmarking
│   ├── history.py            SQLite-backed run + experiment history
│   └── logging_setup.py      Idempotent rotating file + console logging
├── api/                      FastAPI REST service (main.py, schemas.py)
├── evaluation/                Evaluation workspace (datasets/results/plots/examples; nothing but docs committed)
├── main.py                    Tkinter desktop GUI
├── streamlit_app.py           Streamlit web UI (single, batch, evaluation tabs)
├── scripts/                   download_models.py, evaluate.py, benchmark.py CLIs
├── tests/                     54-test unittest suite
├── Dockerfile, docker-compose.yml, .dockerignore   Optional containerization
├── .github/workflows/tests.yml  CI: syntax check, import check, tests, Docker build check
├── Model/                     Pretrained weights (prototxt, caffemodel, cluster points)
├── history/, logs/, images/   Runtime output directories
└── requirements.txt
```

Both UIs and the API import from `colorizer` rather than duplicating logic —
this replaced an earlier single-file `colorizer_core.py` (now removed from
the working tree in favor of the `colorizer/` package).

## 4. Core library (`colorizer/`)

- **`models.py`** — Declares a `MODEL_REGISTRY` of two `ModelSpec`s: `vibrant`
  (required, class-rebalanced, more saturated) and `natural` (optional, muted,
  fetched separately). `available_models()` filters to specs whose weight file
  actually exists on disk, so a UI never offers a broken option. `load_model()`
  is `lru_cache`d per model id, wires the cluster-center layers Zhang et al.'s
  model needs, and transparently switches to CUDA when OpenCV was built with
  CUDA support and a GPU is present. `device_name()` reports `"CUDA"`/`"CPU"`
  for the evaluation, benchmarking, and API/health surfaces.
- **`pipeline.py`** — `colorize_image()` validates the input is a 3-channel BGR
  image, converts to LAB, resizes the L channel to 224×224 for the network,
  runs inference, resizes the predicted AB channels back to the original
  resolution, optionally scales them for a `saturation` control, and remerges
  with the original-resolution L channel before converting back to BGR. Model
  inference is serialized per-model with a lock, since `cv2.dnn.Net` isn't
  documented as thread-safe and Streamlit can invoke it concurrently across
  sessions sharing the cached net.
- **`validation.py`** — Single shared validation path (extension/size/decode/
  pixel-count checks) for untrusted uploads, used by both the Streamlit UI and
  the REST API so the checks are never duplicated or allowed to drift apart.
- **`evaluation.py`** — Grayscales a ground-truth color image, colorizes it,
  and scores the prediction with PSNR/SSIM (via `scikit-image`) and optional
  LPIPS (only if the `lpips`/`torch` packages happen to be installed — neither
  is a required dependency). `evaluate_dataset()` discovers images
  recursively, aggregates mean/median metrics and mean/median/P95 latency,
  and writes CSV + JSON.
- **`benchmarking.py`** — Times model load separately from inference,
  optionally warms up the model with one pass, then measures
  mean/median/P95 latency and images/sec throughput over a fixed image set —
  no background threads, no continuous polling, runs only when invoked.
- **`history.py`** — SQLite-backed record of past runs (filename, model,
  saturation, dimensions, elapsed time, output PNG path) *and* experiments
  (dataset evaluations/benchmarks: PSNR/SSIM/LPIPS, latency percentiles,
  device, git commit). Opens a short-lived connection per call rather than
  caching one at module scope, because Streamlit reruns scripts on a fresh
  thread per interaction and sqlite3 connections are thread-affine. The
  `experiments` table is added via `CREATE TABLE IF NOT EXISTS`, so an older
  database that only has `runs` keeps working with no migration step.
- **`config.py`** — Central paths and limits (`MODEL_DIR`, `MAX_IMAGE_PIXELS`,
  `MAX_UPLOAD_BYTES`, history/log/evaluation locations), each override-able
  via `IMAGECOLORIZER_*` environment variables.
- **`logging_setup.py`** — Idempotent setup of a rotating file handler
  (1 MB × 3 backups) plus console output; safe to call repeatedly across
  Streamlit reruns without duplicating handlers.

## 5. Desktop app (`main.py`)

Tkinter GUI with Open/Colorize/Save and a batch-folder mode.

- Loads the model on a background thread at startup so the UI isn't blocked;
  controls stay disabled with a "Loading model..." status until it's ready,
  and a friendly error dialog appears if the model files are missing.
- Single-image workflow: open → preview → colorize → save as PNG/JPEG.
- Batch workflow runs in its own background thread, writes to a
  `colorized_output` subfolder, tracks per-file success/failure, and reports a
  summary dialog with counts and any failed filenames.
- Rejects images over `MAX_IMAGE_PIXELS` before processing.

## 6. Web app (`streamlit_app.py`)

Streamlit UI with custom CSS theming and three workspaces.

- **Model & saturation controls** — model selector populated from
  `available_models()`, saturation slider (0.0–2.0) passed through to
  `colorize_image`.
- **Single image studio** — upload, validate, preview, colorize, download PNG,
  and a custom before/after comparison slider implemented as an inline HTML
  component (drag-to-reveal, rendered in its own sandboxed iframe since
  `components.html` can't see the page's outer stylesheet).
- **Batch workspace** — multi-file upload, per-file progress bar and status
  text, ZIP export of all successful results, per-file failure list, expandable
  before/after previews, and a processing summary (count, average/total time,
  fastest/slowest file).
- **Evaluation** — pick a local dataset and one or more models, run PSNR/SSIM
  (+LPIPS if available) evaluation on demand, view a results table, per-image
  metric distributions, original/grayscale/prediction/difference previews for
  a few sample images, and the experiment history table. Nothing here runs
  automatically — only on button click.
- Validation layer (shared with the API via `colorizer/validation.py`) rejects
  empty uploads, unsupported extensions, undecodable images, and oversized
  images, with per-file error messages rather than aborting the whole batch.

## 7. REST API (`api/`)

Thin FastAPI layer over `colorizer` — no separate validation or model-loading
logic.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, available model ids, device (CPU/CUDA) |
| GET | `/models` | Full model registry with per-model availability |
| POST | `/colorize` | Upload (`file`, `model`, `saturation`) → PNG, with `X-Model`/`X-Processing-Time` headers |
| GET | `/history` | Recent colorization runs (never exposes server filesystem paths) |
| GET | `/experiments` | Recent evaluation/benchmark experiments |

Status codes: `400` invalid image/request, `413` upload too large, `404`
unknown/unavailable model, `500` unexpected failure (full exception logged
server-side, generic message returned to the client — no stack traces leak
out).

## 8. Tests & CI

54 unit tests across six files, all passing as of this report:

| File | Focus |
|---|---|
| `tests/test_pipeline.py` | Input validation, output shape/dtype, model-id pass-through, saturation — model inference mocked out |
| `tests/test_models.py` | `_require_file`, unknown model id rejection, `available_models()` disk-existence filtering |
| `tests/test_history.py` | Runs + experiments: record/list/get/clear round-trips, ordering, pruning, old-DB backward compatibility |
| `tests/test_evaluation.py` | PSNR/SSIM correctness, dataset discovery, per-image + aggregate evaluation, CSV/JSON output — model inference mocked |
| `tests/test_benchmarking.py` | Percentile math, model-load-time timing, warm-up behavior, latency aggregation — model inference mocked |
| `tests/test_api.py` | `/health`, `/models`, `/colorize` validation (bad extension/oversized/undecodable/unknown model) and success path, `/history`, `/experiments` — via FastAPI's `TestClient` |

```
python -m unittest discover -s tests
Ran 54 tests in ~2s — OK
```

`.github/workflows/tests.yml` runs, on every push/PR to `main`: a syntax
check (`py_compile`), import checks (`colorizer`, `api.main`, both CLIs'
`--help`), the full test suite, and a Docker build check — all without a
GPU, the internet, or the 123MB model weights.

## 9. Dependencies

```
opencv-python<5   # pinned below 5.x: the Caffe importer (readNetFromCaffe) this project relies on requires it
numpy
Pillow
streamlit
scikit-image      # PSNR/SSIM
fastapi
uvicorn
python-multipart  # required by FastAPI for file uploads
httpx             # test-only, for FastAPI's TestClient
```

LPIPS (`lpips` + `torch`) is intentionally **not** in `requirements.txt` — it's
an optional perceptual metric; evaluation works fully without it.

## 10. Model assets (`Model/`)

| File | Size | Role |
|---|---|---|
| `colorization_deploy_v2.prototxt` | 12 KB | Network architecture definition |
| `colorization_release_v2.caffemodel` | 123 MB | Pretrained weights (`vibrant`, required) |
| `pts_in_hull.npy` | 8 KB | 313 quantized ab cluster centers used to seed the output layer |

The optional `natural` (non-rebalanced) weights are not present locally; they
can be fetched with `python scripts/download_models.py natural`. Note: the
`vibrant` weights above are already committed in this repo's git history
(`git ls-files Model/` confirms it) — worth knowing if repo size ever becomes
a concern, since that's 123MB sitting in every clone.

## 11. Current working-tree state (uncommitted, relative to the single `21b5682` commit)

- **Deleted:** `colorizer_core.py` — superseded by the `colorizer/` package.
- **Modified:** `main.py`, `streamlit_app.py`, `README.md`, `.gitignore`.
- **New, untracked:** `colorizer/`, `api/`, `evaluation/`, `tests/`, `scripts/`,
  `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`,
  `.github/workflows/tests.yml`.

In short: the repo's single commit predates a substantial buildout — the
original one-file core became the `colorizer` package plus a FastAPI service,
an evaluation/benchmarking pipeline, and experiment tracking; both UIs were
rebuilt or extended on top of it; and a 54-test suite, CI, and optional Docker
packaging were added. None of this is committed yet.

## 12. Notable design decisions worth knowing

- **Shared core, three front ends** — Tkinter, Streamlit, and FastAPI all call
  the same `colorize_image`/`load_model`/`validate_and_decode_upload`; no
  duplicated business logic.
- **Per-model inference lock** (`pipeline.py`) — guards against concurrent
  Streamlit/API requests hitting a cached, non-thread-safe `cv2.dnn.Net`.
- **Per-call SQLite connections** (`history.py`) — avoids cross-thread
  `sqlite3.ProgrammingError` under Streamlit's per-rerun threading model, and
  avoids leaking Windows file locks on the `.db` file.
- **GPU is opportunistic, not required** — `models.py` upgrades to CUDA only
  if OpenCV was built with CUDA support and a device is present; otherwise CPU
  inference with no configuration needed. No CUDA toolkit, PyTorch, or other
  GPU package is installed by this project.
- **Evaluation/benchmarking are on-demand only** — never run automatically on
  startup, tab switch, or model selection; only on explicit button click or
  CLI invocation, per this project's lightweight-by-default philosophy.
- **`opencv-python<5` pin** — the Caffe importer this project depends on is
  not available in newer OpenCV builds.

## 13. Suggested next steps

- Commit the current working-tree changes (the buildout described in §11) —
  the repository history does not yet reflect the actual state of the code.
- Decide whether to keep the 123MB `caffemodel` committed to git (see §10) or
  move to a download-on-setup model for all weights, `vibrant` included.
- Add screenshots to the README (placeholders currently unfilled).
- Optionally run a real evaluation against a larger personal dataset and drop
  the resulting numbers into the README's "Evaluation Results" table.
