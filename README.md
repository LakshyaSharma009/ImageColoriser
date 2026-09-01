# AI Image Colorizer

End-to-end CNN-based grayscale image colorization system with quantitative evaluation, batch inference, a REST API, experiment tracking, and CPU/GPU benchmarking.

## Project Overview

This project adds color to grayscale photos using a pretrained deep learning
model (Zhang, Isola & Efros, ECCV 2016), served through OpenCV's DNN module.
It started as a colorization demo and has grown into a small, deliberately
lightweight ML system: a shared core library, three interfaces (Tkinter
desktop, Streamlit web, and a FastAPI REST service), a PSNR/SSIM/LPIPS
evaluation pipeline, latency/throughput benchmarking, SQLite-backed
experiment tracking, and CI.

The entire stack runs on CPU with no required external services. GPU support
is opportunistic (used automatically if OpenCV was built with CUDA), never
required.

## Features

- CNN-based image colorization (LAB color-space pipeline, OpenCV DNN)
- Vibrant / Natural model registry, selectable per request
- CPU/GPU inference (automatic CUDA use when available, CPU otherwise)
- Single-image processing (Tkinter + Streamlit)
- Batch processing (folder-based in Tkinter, multi-upload + ZIP export in Streamlit)
- PSNR / SSIM evaluation against a ground-truth dataset (LPIPS optional)
- CPU/GPU latency and throughput benchmarking
- SQLite-backed run history and experiment tracking
- FastAPI REST API (`/health`, `/models`, `/colorize`, `/history`, `/experiments`)
- Streamlit UI, including an Evaluation dashboard (metrics, distributions, example predictions, experiment history)
- Docker + docker-compose (optional)
- GitHub Actions CI (syntax checks, unit/integration tests, Docker build check)
- Input validation and security controls for untrusted uploads

## Model Used

- Paper: *Colorful Image Colorization* (ECCV 2016)
- Authors: Richard Zhang, Phillip Isola, Alexei A. Efros
- Framework: Caffe model loaded through OpenCV DNN
- Input space: LAB color space, where the network predicts the missing `a` and `b` color channels from the grayscale `L` channel

## Architecture

```text
                 ┌──────────────────┐        ┌──────────────────┐
                 │   Streamlit UI   │        │     FastAPI      │
                 └────────┬─────────┘        └────────┬─────────┘
                          │                            │
                          └─────────────┬──────────────┘
                                        │
                              ┌──────────▼──────────┐
                              │   Colorizer Core     │
                              │                       │
                              │  LAB pipeline         │
                              │  Model registry/cache │
                              │  Validation           │
                              │  Evaluation           │
                              │  Benchmarking         │
                              └──────────┬────────────┘
                                        │
                        ┌───────────────┼────────────────┐
                        ▼               ▼                ▼
                  OpenCV DNN        SQLite            Logging
                  (Caffe model)   (runs + experiments)

Evaluation pipeline:

  Ground-truth color image -> grayscale -> colorize -> predicted image
                                                             │
                                          compare against original: PSNR / SSIM / LPIPS
                                                             │
                                                    CSV + JSON + SQLite experiment row
```

Both the Streamlit UI and the FastAPI service call the same `colorizer` core
library -- there is no duplicated business logic, validation, or model
loading between them. Within a single process, the model is loaded once and
cached (`functools.lru_cache`); if you run Streamlit and the API as separate
processes, each holds its own copy (that's normal OS process isolation, not
something this project tries to solve with a model-serving layer).

## Pipeline

```text
Image -> Normalize -> LAB -> Extract L -> CNN -> Predict AB -> Merge LAB -> BGR -> Output
```

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Note: the model depends on OpenCV's Caffe importer, so the project requires `opencv-python<5`.

Download the model files into a `Model/` folder in the project root:

- `colorization_deploy_v2.prototxt`
- `colorization_release_v2.caffemodel`
- `pts_in_hull.npy`

(An optional second model, `natural`, can be fetched with
`python scripts/download_models.py natural`; the app and CLIs only ever list
models whose weight files are actually present.)

## Usage

Run the desktop app:

```bash
python main.py
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Run the REST API:

```bash
uvicorn api.main:app --reload
```

Run the tests:

```bash
python -m unittest discover -s tests
```

In the Streamlit app:

- **Single image studio** shows the original image, the colorized output, and a before/after comparison slider.
- **Batch workspace** processes multiple files, shows progress, reports statistics, and lets you download a ZIP of the results.
- **Evaluation** scores colorization quality against your own dataset of color photos (see [Evaluation](#evaluation) below) and shows an experiment history table.

## REST API

Thin HTTP layer over the same core library -- no separate validation or
model-loading logic.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status, available model ids, device (CPU/CUDA) |
| GET | `/models` | Full model registry with per-model availability |
| POST | `/colorize` | Upload an image (`file`, `model`, `saturation`), get back a PNG |
| GET | `/history` | Recent colorization runs |
| GET | `/experiments` | Recent evaluation/benchmark experiments |

`/colorize` returns `X-Model` and `X-Processing-Time` response headers, and
maps validation failures to proper status codes (`400` invalid image/request,
`413` upload too large, `404` unknown/unavailable model, `500` unexpected
failure -- with the real exception logged server-side, never sent to the
client).

## Evaluation

Quantitative evaluation measures how close a colorized prediction gets to the
real color photo it was generated from:

```text
color image -> grayscale -> colorize -> prediction -> compare against original
                                                            (PSNR / SSIM / LPIPS)
```

PSNR and SSIM are always computed (via `scikit-image`). LPIPS is an optional
perceptual metric -- it requires the third-party `lpips` package (and
`torch`, which it depends on), neither of which is installed by default. When
they're absent, LPIPS is reported as unavailable and PSNR/SSIM continue to
work normally.

Bring your own dataset (see [`evaluation/README.md`](evaluation/README.md)
for the expected layout and supported formats), then run:

```bash
python scripts/evaluate.py --dataset evaluation/datasets/<dataset_name> --model vibrant
```

Flags: `--dataset` (required), `--model`, `--output`, `--saturation`, `--limit`.
This writes a per-image CSV and an aggregate JSON summary to
`evaluation/results/`, and logs the run to the SQLite experiment history
(also browsable from the Streamlit app's Evaluation tab, including PSNR/SSIM
distributions and side-by-side original / grayscale / prediction / difference
previews). Evaluation only ever runs when you explicitly invoke it -- never
automatically on startup, tab switch, or model selection.

### Evaluation Results

No numbers are hard-coded here -- run the evaluator against your own dataset
to fill this in:

| Model   | Images | PSNR | SSIM | LPIPS | Avg Latency |
| ------- | -----: | ---: | ---: | ----: | ----------: |
| Vibrant |      — |    — |    — |     — |           — |
| Natural |      — |    — |    — |     — |           — |

## Performance

Benchmark inference latency and throughput (separately from model load time):

```bash
python scripts/benchmark.py --model vibrant --images evaluation/datasets/<dataset_name> --iterations 3
```

- Reports CPU or CUDA based on the actual OpenCV build/runtime (no GPU required, none installed automatically).
- Warms up the model with one inference pass before timing begins, so the first (slower, JIT/cache-cold) call doesn't skew the numbers.
- Reports **model load time** separately from **inference latency**.
- Reports mean, median, and P95 inference latency, plus images/sec throughput.
- Runs only when explicitly invoked -- normal colorization requests are never benchmarked automatically.

## Security

Uploaded images are treated as untrusted input. `colorizer/validation.py` is
the single validation path shared by the Streamlit UI and the REST API (no
duplicated checks), and enforces:

- File-size limits (`IMAGECOLORIZER_MAX_UPLOAD_BYTES`, default 20MB)
- Pixel-count limits (`IMAGECOLORIZER_MAX_IMAGE_PIXELS`, default 25,000,000)
- File-extension allowlist (`.jpg`, `.jpeg`, `.png`, `.bmp`)
- Decoded-image validation (rejects files that aren't actually valid images, regardless of extension)
- No user-controlled filesystem paths -- output filenames/paths are always server-generated (UUIDs for history/evaluation output, derived-and-sanitized names for downloads), never taken from request input
- The REST API never returns internal filesystem paths (e.g. `output_path` is intentionally excluded from `/history` responses) and never leaks stack traces (exceptions are logged server-side, clients get a generic message + appropriate status code)
- Batch processing (Tkinter and Streamlit) validates and writes each file individually, so one bad or oversized file can't take down the whole batch or overwrite arbitrary paths
- Images are processed one at a time rather than loaded into memory as a full batch

## Configuration

All paths and limits are configurable via `IMAGECOLORIZER_*` environment
variables (see `colorizer/config.py`):

```text
IMAGECOLORIZER_MODEL_DIR
IMAGECOLORIZER_MAX_IMAGE_PIXELS
IMAGECOLORIZER_MAX_UPLOAD_BYTES
IMAGECOLORIZER_HISTORY_DIR
IMAGECOLORIZER_HISTORY_MAX_ENTRIES
IMAGECOLORIZER_LOG_DIR
IMAGECOLORIZER_LOG_LEVEL
IMAGECOLORIZER_EVALUATION_DIR
```

None of these require code changes -- no hard-coded machine-specific paths.

## Docker (optional)

Docker is never required -- `python main.py` and `streamlit run streamlit_app.py`
work directly. If you'd rather containerize:

```bash
docker compose up            # Streamlit UI on http://localhost:8501
docker compose --profile api up api   # FastAPI on http://localhost:8000
```

Model weights aren't baked into the image; mount your local `Model/` folder
(the compose file does this for you) or download them inside the container.

## Testing

```bash
python -m unittest discover -s tests
```

All tests use mocked model inference and tiny synthetic images -- none
require a GPU, internet access, or the 123MB model weights. Coverage
includes the colorization pipeline, model registry, run/experiment history
(SQLite), PSNR/SSIM evaluation, benchmarking, and the REST API (health,
models, validation errors, successful colorization, error handling).

CI (`.github/workflows/tests.yml`) runs syntax checks, import checks, the
full test suite, and a Docker build check on every push/PR to `main`.

## Project Structure

```text
ImageColoriser/
├── colorizer/              Shared core library
│   ├── __init__.py         Public API surface
│   ├── config.py           Paths & tunables, override-able via env vars
│   ├── models.py           Model registry, loading, GPU detection, caching
│   ├── pipeline.py         colorize_image(): the LAB colorization pipeline
│   ├── validation.py       Shared validation for untrusted image uploads
│   ├── evaluation.py       PSNR/SSIM/LPIPS evaluation against a dataset
│   ├── benchmarking.py     Latency/throughput benchmarking
│   ├── history.py          SQLite-backed run + experiment history
│   └── logging_setup.py    Idempotent rotating file + console logging
│
├── api/                    FastAPI REST API
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── evaluation/              Evaluation workspace (see evaluation/README.md)
│   ├── datasets/            Your own datasets (not committed)
│   ├── results/             CSV/JSON output (not committed)
│   ├── plots/                (not committed)
│   └── examples/             (not committed)
│
├── tests/                  unittest suite
│   ├── test_pipeline.py
│   ├── test_models.py
│   ├── test_history.py
│   ├── test_evaluation.py
│   ├── test_benchmarking.py
│   └── test_api.py
│
├── scripts/
│   ├── download_models.py
│   ├── evaluate.py
│   └── benchmark.py
│
├── Model/                  Pretrained weights
├── main.py                 Tkinter GUI entry point
├── streamlit_app.py        Streamlit UI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .github/workflows/tests.yml
└── README.md
```

## Credits

Model architecture and pretrained weights from ["Colorful Image Colorization"](https://richzhang.github.io/colorization/) by Zhang, Isola, and Efros.
