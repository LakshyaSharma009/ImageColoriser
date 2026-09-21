# AI Image Colorizer

Turn grayscale photos into vivid color with a CNN-based pipeline, exposed through three interfaces sharing one core library.

- **3 frontends, 1 core:** Tkinter desktop (`main.py`), Streamlit studio (`streamlit_app.py`), FastAPI REST (`api/main.py`) — all call `colorizer/`.
- **Model:** Zhang et al. 2016 colorization CNN via OpenCV DNN (Caffe), LAB color space, 2 switchable weights.
- **Beyond inference:** batch processing, saturation control, PSNR/SSIM/LPIPS evaluation, SQLite experiment history, latency benchmarking, Docker + CI.

> Model weights (`*.caffemodel`, ~120 MB) are **not** committed. Fetch them with `scripts/download_models.py` (see Setup).

---

## How it works

```text
BGR image -> LAB -> L resized to 224x224 -> Caffe CNN predicts ab (313-bin distribution)
  -> ab resized to original size (x saturation) -> reattach original L -> LAB2BGR -> PNG
```

1. Convert BGR to LAB and keep the original `L` (lightness) untouched.
2. Feed a `224x224` `L` channel (mean-centered by 50) through `colorization_deploy_v2.prototxt` + caffemodel.
3. The net outputs quantized `ab` chrominance, upsampled to full resolution.
4. Recombine original `L` + predicted `ab`, convert back to BGR.

Two weight variants (`colorizer/models.py`):

| ID | Label | File | Notes |
|---|---|---|---|
| `vibrant` (default) | Vibrant (rebalanced) | `Model/colorization_release_v2.caffemodel` | Class-rebalanced, punchier colors. Required. |
| `natural` | Natural (muted) | `Model/colorization_release_v2_norebal.caffemodel` | More conservative. Optional, fetched via download script. |

Shared small files stay tracked: `Model/colorization_deploy_v2.prototxt`, `Model/pts_in_hull.npy`.

## Setup

Requires Python 3.11, `opencv-python<5` (Caffe support).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python scripts/download_models.py vibrant
# optional second model:
python scripts/download_models.py natural
```

If the download host is down, manually place the file from `richzhang/colorization` (`models/fetch_release_models.sh`) into `Model/` with the exact filename above.

## Usage

### 1. Tkinter desktop app

```bash
python main.py
```

Open Image → Colorize → Save Result, or Batch Colorize Folder (writes `<folder>/colorized_output/*_colorized.png`). Model loads on a background thread; UI stays responsive.

### 2. Streamlit studio (recommended)

```bash
streamlit run streamlit_app.py
```

- **Single image studio:** upload → preview → Colorize → before/after slider → Download PNG.
- **Batch workspace:** multi-upload → Colorize all → Download ZIP + per-image previews + timing summary.
- **Evaluation tab:** pick dataset + model(s) → Run evaluation → PSNR/SSIM table, distribution charts, example predictions with difference view, experiment history.
- Global controls: model picker + saturation slider (scales predicted `ab`, `0.0` = grayscale, `1.0` = default, `2.0` = oversaturated).

### 3. FastAPI REST API

```bash
uvicorn api.main:app --reload
# docs: http://127.0.0.1:8000/docs
```

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Status + available models + device (`CPU`/`CUDA`). |
| `GET` | `/models` | All registry models with `available` flag. |
| `POST` | `/colorize` | Multipart `file` + form `model`, `saturation`. Returns `image/png` + `X-Model` / `X-Processing-Time` headers. |
| `GET` | `/history` | Past runs (`?limit=`). |
| `GET` | `/experiments` | Past evaluations (`?limit=`). |

Example:

```bash
curl -X POST http://127.0.0.1:8000/colorize \
  -F "file=@photo.jpg" -F "model=vibrant" -F "saturation=1.0" \
  --output colorized.png
```

Errors: `400` bad file, `404` unknown/missing model, `413` >20 MB upload, `500` inference failure.

## Evaluation

Provide your own ground-truth color photos (evaluator grayscales them itself — do not pre-convert):

```text
evaluation/datasets/<dataset_name>/
  image1.jpg
  image2.png
  subfolder/image3.webp
```

Supported: `.jpg` `.jpeg` `.png` `.webp`, recursive. See `evaluation/README.md`.

```bash
python scripts/evaluate.py --dataset evaluation/datasets/<dataset_name> --model vibrant
# options: --saturation 1.0 --limit 20 --output evaluation/results/custom.csv
```

Writes per-image CSV + aggregate JSON to `evaluation/results/`, records a row in SQLite history (visible in Streamlit + `/experiments`).

Metrics:

- **PSNR (dB):** pixel fidelity, higher is better.
- **SSIM (−1…1):** structural similarity, `1.0` = identical.
- **LPIPS:** perceptual distance, lower is better. Optional — needs `pip install lpips torch`; reported as `n/a` otherwise.
- Latency: mean / median / p95 per image.

## Benchmarking

```bash
python scripts/benchmark.py --images evaluation/datasets/<dataset_name> --model vibrant --iterations 3 --limit 10
```

Measures cold model-load time + warmup + N-pass inference latency (mean/median/p95) and throughput (images/sec). No disk writes; images decoded once up front.

## Docker (optional)

Docker is never required — `main.py` / `streamlit_app.py` run directly.

```bash
docker compose up            # Streamlit on ${STREAMLIT_PORT:-8501}
docker compose --profile api up  # also FastAPI on ${API_PORT:-8000}
docker build -t imagecoloriser .
```

Model/history/logs/evaluation are mounted as volumes; override paths with `IMAGECOLORIZER_*` env vars.

## Project structure

```text
main.py               Tkinter desktop UI
streamlit_app.py      Streamlit studio (single / batch / evaluation)
api/                  FastAPI layer (main.py, schemas.py) — no business logic
colorizer/            Shared core library
  config.py           Paths + limits (env-overridable)
  models.py           Registry, cached loader, CUDA auto-switch
  pipeline.py         colorize_image() + per-model thread lock
  validation.py       Upload validation (extension/size/decode/dimensions)
  evaluation.py       PSNR/SSIM/LPIPS dataset evaluation
  benchmarking.py     Latency/throughput harness
  history.py          SQLite runs + experiments
  logging_setup.py    Rotating file + console logging
scripts/              download_models.py, evaluate.py, benchmark.py
Model/                prototxt + pts_in_hull.npy tracked; *.caffemodel gitignored
evaluation/           datasets/ results/ plots/ examples/ (gitignored, see evaluation/README.md)
tests/                unittest suite (mocks inference, no GPU/net/weights needed)
Dockerfile, docker-compose.yml, .github/workflows/tests.yml
```

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `IMAGECOLORIZER_MODEL_DIR` | `<repo>/Model` | Model files location |
| `IMAGECOLORIZER_MAX_IMAGE_PIXELS` | `25000000` | Max pixels per image |
| `IMAGECOLORIZER_MAX_UPLOAD_BYTES` | `20971520` (20 MB) | Max upload bytes |
| `IMAGECOLORIZER_HISTORY_DIR` | `<repo>/history` | SQLite DB + outputs |
| `IMAGECOLORIZER_HISTORY_MAX_ENTRIES` | `200` | Prune cap for runs |
| `IMAGECOLORIZER_LOG_DIR` / `IMAGECOLORIZER_LOG_LEVEL` | `<repo>/logs` / `INFO` | Logging |
| `IMAGECOLORIZER_EVALUATION_DIR` | `<repo>/evaluation` | Datasets/results |

Uploads accept `.jpg` `.jpeg` `.png` `.bmp` (evaluation adds `.webp`).

## Tests & CI

```bash
python -m unittest discover -s tests
python -m py_compile main.py streamlit_app.py colorizer/*.py api/*.py scripts/*.py
```

CI (`.github/workflows/tests.yml`): pip install → syntax check → import/CLI help checks → full unittest suite → Docker build check. All tests mock `cv2.dnn.Net`; no GPU, internet, or weights required.

## Limitations

- Generative colorization is a guess — expect plausible but not always historically accurate colors, especially on unusual objects.
- CPU inference is seconds per image; GPU needs an OpenCV CUDA build.
- Large images are rejected at 25 MP to bound memory.
- LPIPS requires heavy optional `torch` dependency.

## Credits

Model and weights from Zhang, Isola & Efros, “Colorful Image Colorization” (ECCV 2016), via `richzhang/colorization`. Caffe inference through OpenCV DNN.
