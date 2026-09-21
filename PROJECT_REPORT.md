# AI Image Colorizer — Project Report

**Author:** Lakshya Sharma
**Repository:** [LakshyaSharma009/ImageColoriser](https://github.com/LakshyaSharma009/ImageColoriser)
**Date:** September 2026
**Status:** Working system. 54/54 unit tests passing (verified September 2026, ~4.7 s).

> This is the formal project report. For setup/usage see `README.md`; for interview preparation see `INTERVIEW_PREP.md`; for the evaluation workspace contract see `evaluation/README.md`.

---

## 1. Abstract

This project delivers an end-to-end grayscale-to-color image colorization system built on the pretrained convolutional model of Zhang, Isola & Efros (*Colorful Image Colorization*, ECCV 2016), executed through OpenCV's DNN module on a Caffe checkpoint. A single shared Python library (`colorizer/`) implements model management, the LAB-space inference pipeline, input validation, quantitative evaluation, latency benchmarking, experiment history, and logging. Three user-facing interfaces — a Tkinter desktop application, a Streamlit web studio, and a FastAPI REST service — are thin layers over that core, alongside three command-line tools. The system supports two model weight variants, single and batch colorization, saturation control, PSNR/SSIM/LPIPS evaluation, SQLite-backed run and experiment tracking, Docker packaging, and continuous integration. All 54 unit tests pass without requiring a GPU, network access, or model weights.

## 2. Objectives

1. Colorize grayscale photographs into plausible, vivid color with a pretrained deep model, without training infrastructure.
2. Serve the capability through desktop, web, and API interfaces without duplicating business logic.
3. Make results measurable: quantitative quality metrics, latency benchmarks, and persistent experiment records.
4. Harden the system for untrusted input (size, format, decode, and dimension validation) with correct HTTP semantics.
5. Keep the project reproducible and portable: pinned dependencies, environment-based configuration, Docker, and CI.

## 3. Background

Automatic colorization is ill-posed: many plausible colorings exist for a single grayscale input. Early approaches relied on user scribbles or reference images. Zhang et al. reframed colorization as a classification problem over 313 quantized `ab` bins in CIE LAB space rather than direct regression, preserving the multimodal nature of color (an apple may be red or green) instead of averaging to desaturated brown. Class rebalancing during training emphasizes rare, vivid colors. This project reuses that work's released Caffe artifacts (`colorization_deploy_v2.prototxt`, `colorization_release_v2*.caffemodel`, `pts_in_hull.npy`) and runs them with `cv2.dnn`, avoiding any PyTorch/TensorFlow dependency at inference time.

## 4. System overview

```text
BGR -> LAB -> L (224x224, mean-centered) -> CNN predicts ab distribution
  -> ab resized to full resolution (x saturation) -> original L + predicted ab
  -> LAB2BGR -> PNG

Evaluation: color -> grayscale -> colorize -> compare vs. original (PSNR/SSIM/LPIPS)
```

| Component | Path | Role |
|---|---|---|
| Shared core | `colorizer/` | Config, models, pipeline, validation, evaluation, benchmarking, history, logging |
| Desktop UI | `main.py` | Tkinter: open / colorize / save + batch folder mode |
| Web studio | `streamlit_app.py` | Single studio, batch workspace (ZIP), evaluation tab, before/after slider |
| REST API | `api/main.py`, `api/schemas.py` | `GET /health`, `GET /models`, `POST /colorize`, `GET /history`, `GET /experiments` |
| CLIs | `scripts/download_models.py`, `scripts/evaluate.py`, `scripts/benchmark.py` | Weight provisioning, dataset evaluation, latency benchmarking |
| Packaging | `Dockerfile`, `docker-compose.yml`, `.github/workflows/tests.yml` | Optional containers + CI |

## 5. Methodology and implementation

### 5.1 Colorization pipeline (`colorizer/pipeline.py`, `colorizer/models.py`)

`colorize_image(img, model_id, saturation)` requires a 3-channel BGR `uint8` image, then: normalizes to `[0,1]`, converts to LAB, resizes to `224x224`, extracts and mean-centers `L` (−50), builds a DNN blob, runs `net.forward()`, reshapes/upsamples the predicted `ab` to the original resolution, applies the scalar `saturation` multiplier (`0.0` = grayscale, `1.0` = default, `2.0` = oversaturated), recombines with the original full-resolution `L`, converts back to BGR, clips, and returns `uint8`.

`MODEL_REGISTRY` defines `vibrant` (class-rebalanced, required) and `natural` (muted, optional). `load_model()` is `lru_cache`d per id, injects the 313 `pts_in_hull` cluster centers into the `class8_ab` layer with the `2.606` rebalancing prior on `conv8_313_rh`, and opportunistically selects the CUDA DNN backend when OpenCV reports a CUDA device — CPU otherwise, with no mandatory GPU stack. `available_models()` filters by weights present on disk so no UI can offer a missing model. Inference is serialized per model with a dedicated lock because `cv2.dnn.Net.setInput/forward` is not documented as thread-safe and Streamlit/API sessions share the cached net.

### 5.2 Input validation (`colorizer/validation.py`)

One shared path for all entry points: filename extension check (`.jpg` `.jpeg` `.png` `.bmp`), non-empty bytes, 20 MB byte cap *before* decoding (denial-of-service guard), `cv2.imdecode`, then decodability, 3-channel, and 25 MP dimension checks *after* decoding (decompression-bomb guard). `PayloadTooLargeError` subclasses `ValidationError` so the API maps oversize to `413` and other bad input to `400`.

### 5.3 Interfaces

**Tkinter (`main.py`).** Model loads on a daemon thread at startup; controls disable until ready. Single mode previews via `cv2_to_tk` (fit-to-450, reference retained); batch mode walks a chosen folder on a worker thread into `colorized_output/*_colorized.png` with `root.after()` status updates and a success/failure summary dialog.

**Streamlit (`streamlit_app.py`).** Model picker from `available_models()` plus a global saturation slider; single studio (upload → preview → colorize → PNG download), batch workspace (multi-upload, progress bar, in-memory `ZIP_DEFLATED` export, expandable previews, mean/total/fastest/slowest summary), and an evaluation tab (dataset + multi-model selection, on-demand runs, results table, PSNR/SSIM distribution charts, original/grayscale/prediction/difference previews, experiment history). The before/after slider is a self-contained HTML/JS component (stacked data-URI images, `clip-path` driven by a range input) because `components.html` iframes cannot see page CSS. Logging setup is idempotent across Streamlit reruns.

**FastAPI (`api/`).** Stateless thin layer: `POST /colorize` accepts multipart `file` plus `model`/`saturation` forms, reuses core validation and pipeline, times inference, returns PNG with `X-Model` and `X-Processing-Time` headers. `400` invalid image, `413` oversized, `404` unknown/unavailable model, `500` unexpected failure (traceback logged server-side, generic message to client). `GET /` redirects to `/docs`; history endpoints intentionally omit server-local filesystem paths.

### 5.4 Evaluation (`colorizer/evaluation.py`, `scripts/evaluate.py`)

Protocol per image: ground-truth color → synthetic grayscale (3-channel) → colorize → score prediction against the untouched original. Metrics: PSNR (dB, higher better), SSIM (−1…1, `1.0` identical) via scikit-image, and optional LPIPS (lower better, `None`/`n/a` when `lpips`/`torch` absent — never a failure). `evaluate_dataset()` discovers `.jpg`/`.jpeg`/`.png`/`.webp` recursively, warms the model once, processes one image at a time, and aggregates mean/median plus mean/median/p95 latency; results persist as per-image CSV plus aggregate JSON under `evaluation/results/` and a SQLite experiment row (model, dataset, count, saturation, device, metrics, latency, git commit). Datasets live under `evaluation/datasets/<name>/` and are gitignored by design.

### 5.5 Benchmarking (`colorizer/benchmarking.py`, `scripts/benchmark.py`)

Cold model-load timing (cache cleared first) reported separately from inference; one warmup pass; then N iterations over a fixed pre-decoded image set with mean/median/p95 latency and images/sec throughput. No disk writes, no thread pools — same locked inference path as production.

### 5.6 History and configuration (`colorizer/history.py`, `colorizer/config.py`)

SQLite (WAL) stores `runs` (filename, model, saturation, dimensions, elapsed, output PNG path, timestamp) and `experiments` (evaluation aggregates plus git short hash, best-effort). Short-lived per-call connections avoid Streamlit cross-thread `ProgrammingError` and Windows `.db` lock leaks; `runs` auto-prunes to 200 entries including orphan PNG cleanup; tables use `CREATE TABLE IF NOT EXISTS` so no migrations are needed. All paths and limits are `IMAGECOLORIZER_*` environment-overridable for Docker.

## 6. Testing and CI

54 unit tests across `tests/` (`test_api`, `test_benchmarking`, `test_evaluation`, `test_history`, `test_models`, `test_pipeline`), verified passing in September 2026 (`Ran 54 tests — OK`). All DNN inference is mocked (zero-filled `forward`), so the suite needs no GPU, network, or weights; it covers validation, shape/dtype contracts, model-id routing, saturation invariance, metric correctness, discovery, CSV/JSON output, history round-trips/ordering/pruning/backward compatibility, percentile math, warmup behavior, and API success/error paths via `TestClient`. CI (`.github/workflows/tests.yml`) runs `py_compile`, import/CLI smoke checks, the full suite, and a Docker build check on every push/PR to `main`.

## 7. Results

- Functional: single and batch colorization work across all three interfaces; sample images in `images/batch/` (`lion.jpg`, `valley.jpg`, `gray.jpeg`) colorize successfully.
- Quality: evaluation is procedure-complete and dataset-dependent by design; no fixed scores are claimed here — run `python scripts/evaluate.py --dataset evaluation/datasets/<name> --model vibrant` to produce PSNR/SSIM/LPIPS numbers for any dataset, viewable in the CLI, Streamlit tab, and `/experiments`.
- Performance: latency is device-dependent (CPU vs CUDA auto-detect); run `python scripts/benchmark.py --images evaluation/datasets/<name> --model vibrant --iterations 3` for load time, p95 latency, and throughput on the host.
- Reliability: oversized, corrupt, and wrong-extension uploads are rejected with per-file messages (UI) or precise status codes (API); missing weights produce actionable errors naming the expected `Model/` files.

## 8. Limitations

Colorization is a plausible guess, not ground-truth recovery — unusual objects may receive incorrect but natural-looking hues. CPU inference is seconds per image; GPU requires an OpenCV CUDA build. Inputs are capped at 25 MP / 20 MB. LPIPS needs the heavy optional `torch` stack. The 2016 Caffe model trails modern GAN/diffusion colorizers on faces and fine textures.

## 9. Future work

Perceptual-loss fine-tuning or a modern backbone (e.g. DDColor) with ONNX export; face-aware priors; async batch job queue with IDs for the API; authentication/rate limiting; hosted demo; Playwright UI tests; larger curated evaluation dataset with published score table in the README.

## 10. Conclusion

The project meets its objectives: one tested core library powers three interfaces plus evaluation, benchmarking, history, packaging, and CI, with clean separation between ML inference, validation, presentation, and operations. The system is demonstrable offline after one-time weight download and extensible — a new model is a single `ModelSpec` entry.

## References

- Zhang, Isola & Efros, “Colorful Image Colorization,” ECCV 2016; artifacts via `richzhang/colorization`.
- OpenCV DNN Caffe inference; scikit-image PSNR/SSIM; optional `lpips`+`torch`; Streamlit; FastAPI; SQLite.
