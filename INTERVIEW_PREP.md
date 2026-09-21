# ImageColoriser — Interview Prep (Personal Notes)

> Private deep-dive for interviews. Public overview lives in `README.md`.
> Code references below match the current checkout (`main` branch).

---

## 1. Pitch scripts

### 30-second version
“I built an end-to-end AI image colorizer: a shared Python core wraps a Zhang-2016 colorization CNN running in OpenCV DNN, exposed through a Tkinter desktop app, a Streamlit studio with single/batch/evaluation modes, and a FastAPI REST API. On top I added quantitative evaluation with PSNR/SSIM/LPIPS, SQLite experiment tracking, latency benchmarking, validation, Docker, and CI.”

### 2-minute version
“The interesting part is that all three UIs share one `colorizer/` library with no duplicated logic. The pipeline converts BGR to LAB, feeds only the L channel resized to 224×224 into a Caffe model that predicts ab chrominance over 313 quantized bins, then recombines the original L with predicted ab. I support two weight variants — vibrant rebalanced and natural — with a saturation multiplier. Around that I built: shared upload validation, thread-safe cached inference, a FastAPI layer with health/models/colorize/history/experiments endpoints, a Streamlit evaluation tab that grayscales ground truth and scores predictions, a CLI evaluator and benchmark harness, SQLite history with pruning, env-based config, Docker Compose, and a GitHub Actions suite where all tests mock the DNN so no GPU or weights are needed.”

---

## 2. Architecture

```text
                    +-------------------+
                    |   colorizer/ core |
                    | config models     |
                    | pipeline valid.   |
                    | eval bench hist   |
                    +--------+----------+
                             |
        +--------------------+--------------------+
        |                    |                    |
   main.py (Tkinter)  streamlit_app.py      api/main.py (FastAPI)
   desktop + batch    single/batch/eval     /health /models
                                                /colorize /history
                                                /experiments
        |                    |                    |
        +--------------------+--------------------+
                             |
              scripts/download_models.py
              scripts/evaluate.py (CLI)
              scripts/benchmark.py (CLI)
```

Key design decision: **API and UIs are thin; all business logic lives in `colorizer/`**. Ask me: “why not put validation in the API?” — answer: uploads also come from Streamlit/Tkinter, so `colorizer/validation.py` is the single source of truth.

Request flows:

- **Tkinter:** button → `cv2.imread` → pixel-count guard → `colorize_image()` → display/save. Model preloaded on daemon thread (`load_model_async`), controls disabled until ready. Batch loops a folder on a worker thread, `root.after()` for status updates.
- **Streamlit:** `main()` → `available_models()` gate → model select + saturation slider → `load_model(model_id)` (cached) → radio workspace → `render_single/batch/evaluation_mode()`. Reruns whole script per interaction, hence idempotent logging and short-lived SQLite connections.
- **FastAPI:** `POST /colorize` → registry check → availability check → `await file.read()` → `validate_and_decode_upload()` → timed `colorize_image()` → `cv2.imencode('.png')` → PNG response with timing headers. `PayloadTooLargeError→413`, `ValidationError→400`, unknown/missing model→404.

---

## 3. ML / pipeline deep-dive (`colorizer/pipeline.py`, `models.py`)

### 3.1 Why LAB?
RGB entangles brightness and color. LAB splits them: `L` = lightness, `a/b` = chrominance. Grayscale input already gives us `L`; the model only must predict `ab`. Preserving the original `L` keeps details sharp and avoids brightness drift.

### 3.2 Exact steps (`pipeline.colorize_image`, line 22)
1. Assert 3-channel BGR `uint8`.
2. `models.load_model(model_id)` — cached `cv2.dnn.Net`.
3. `img/255 → BGR2LAB`, resize LAB to `224x224`, take `L`, subtract 50 (mean-centering the original paper used).
4. `cv2.dnn.blobFromImage(l_small)` → `net.setInput(blob)` → `net.forward()` → shape `(1,313,H,W)`-ish distribution over quantized ab bins.
5. Transpose/resize ab to original W×H, multiply by `saturation` (simple linear scaling in ab space: `0.0` = gray, `>1` = punchy).
6. Concatenate original full-res `L` + predicted `ab` → `LAB2BGR` → clip `[0,1]` → `×255 uint8`.

### 3.3 Why classification, not regression? (favorite interview Q)
Directly regressing `ab` with L2 loss produces desaturated brownish averages because color is multimodal (an apple could be red or green). Zhang et al. quantize ab space into **313 bins** and train a classifier; the soft distribution preserves multimodality. At inference the model takes an annealed mean of the distribution. The repo injects the cluster centers explicitly:

```python
pts = kernel.transpose().reshape(2, 313, 1, 1)  # pts_in_hull.npy
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, ...)]
```

`2.606` is the rebalancing temperature prior from the paper. The `natural` (norebal) weights skip class rebalancing → muted but sometimes more realistic.

### 3.4 Model loading (`models.py:80`)
- `@lru_cache` so the ~120 MB net loads once per process per `model_id`.
- `readNetFromCaffe(prototxt, caffemodel)`; guard for OpenCV builds without Caffe (`opencv-python<5` required).
- `_require_file` raises `FileNotFoundError` with the exact missing path — Streamlit/API surface this as “model files missing”.
- `_use_gpu_if_available()`: checks `cv2.cuda.getCudaEnabledDeviceCount()`, switches to `DNN_BACKEND_CUDA / DNN_TARGET_CUDA`. Never hard-requires CUDA.
- `available_models()` filters registry by files on disk so UIs never offer a broken choice.

### 3.5 Thread safety
`cv2.dnn.Net.setInput+forward` is not documented thread-safe. `pipeline.py:13` keeps a per-`model_id` `threading.Lock`; concurrent Streamlit sessions/API workers serialize only around those two calls, not the whole colorize. Benchmarking explicitly reuses this — “inference stays single-threaded, no extra pools”.

---

## 4. Module-by-module reference

| File | What to say |
|---|---|
| `colorizer/config.py` | Single place for `MODEL_DIR`, `VALID_EXTENSIONS=(.jpg,.jpeg,.png,.bmp)`, 25 MP / 20 MB caps, history/log/eval dirs. Everything env-overridable (`IMAGECOLORIZER_*`) for Docker. |
| `colorizer/validation.py` | Untrusted-input boundary. `validate_upload_bytes` (extension+empty+size) → `decode_image` (`cv2.imdecode`) → `validate_decoded_image` (decodable? 3ch? pixel cap). `PayloadTooLargeError` subclasses `ValidationError` so callers can map 413 vs 400. |
| `colorizer/evaluation.py` | Protocol: `color → gray(GRAY→BGR 3ch) → colorize → compare vs original`. Never mutates ground truth. `discover_images` recursive, eval exts add `.webp`. `calculate_psnr` (skimage, dB, inf=identical), `calculate_ssim` (`channel_axis=2`), `calculate_lpips` (optional; `None` if missing). `evaluate_dataset(limit, progress_callback)` warms model once, loops one-at-a-time (no batch OOM), aggregates mean/median/p95 via `statistics`+`numpy.percentile`. `write_csv`/`write_json` for artifacts. |
| `colorizer/history.py` | SQLite with WAL. `_connect` context manager: mkdir, connect, yield inside transaction, close (avoids Streamlit thread-reuse `ProgrammingError` + Windows file locks). `runs` (per colorize) + `experiments` (per eval, incl. `git_commit` via `git rev-parse --short HEAD` best-effort). `_prune` keeps last 200 runs and deletes orphan PNGs. `CREATE TABLE IF NOT EXISTS` = no migrations needed. |
| `colorizer/benchmarking.py` | `measure_model_load_time` clears `lru_cache` to time cold load. `benchmark_images(paths, iterations, warmup)` decodes once, warmups once, times each inference, reports mean/median/p95 + `images/sec`. Nothing written to disk. |
| `colorizer/logging_setup.py` | Idempotent (`_configured` flag) rotating 1 MB×3 file + console; safe under Streamlit reruns. |
| `api/main.py` + `schemas.py` | Pydantic response models. `GET /` redirects to `/docs`. Intentionally omits server-local `output_path` from history responses. |
| `streamlit_app.py` | Custom CSS hero/metrics/cards; `to_pil/to_png_bytes/to_data_uri/build_zip` helpers; `validate_upload` thin wrapper; `render_compare_slider` injects self-contained HTML/JS into `components.html` iframe (own `<style>` because iframe can’t see page CSS); batch shows progress bar + ZIP + expandable previews + summary metrics; eval tab caches results in `st.session_state`, writes history best-effort. |
| `main.py` (Tkinter) | `cv2_to_tk` resize-to-450 + BGR→RGB→PhotoImage (keep reference or image vanishes). Same `MAX_IMAGE_PIXELS`/`VALID_EXTENSIONS` constants as core. |
| `scripts/*` | Never imported by app. `argparse` CLIs with availability pre-checks and non-zero exits on bad dataset/model. |
| `Dockerfile` / `docker-compose.yml` | `python:3.11-slim` + `libgl1 libglib2.0-0` for OpenCV codecs. Model mounted `:ro`; history/logs/eval read-write. `api` behind `--profile api`. |
| `.github/workflows/tests.yml` | Checkout → Py3.11 → pip install → `py_compile` → `import colorizer; import api.main` + CLI `--help` → `unittest discover` → `docker build` check. |
| `evaluation/README.md` | Dataset layout + eval command contract. |

---

## 5. Metrics cheat sheet

- **PSNR:** `10·log10(MAX²/MSE)`, dB. ~30+ dB good for colorization; `inf` = identical. Sensitive to small shifts, blind to perception.
- **SSIM:** luminance/contrast/structure product, `1.0` identical. Better aligns with perceived structure than PSNR.
- **LPIPS (Alex):** deep-feature distance, lower better (~0.1–0.3 typical here). Needs `torch`; missing → `None`/`n/a`, never crashes eval.
- **Latency:** mean/median/p95 per image + throughput. Always state device (`device_name()` reports CPU/CUDA) because numbers are meaningless without it.
- Protocol nuance: we grayscale real color photos, so scores measure “reconstruction of known truth”, not absolute aesthetic quality. Good for regression tracking, not a beauty contest.

---

## 6. Correctness / robustness stories (use STAR)

1. **Thread-safety:** shared cached `Net` + concurrent Streamlit sessions → added per-model lock around `setInput/forward` only (minimal critical section).
2. **Streamlit SQLite crashes:** cached connection reused across ScriptRunner threads → switched to short-lived connections with explicit close.
3. **Broken model dropdown:** UIs listed models whose weights were missing → `available_models()` filters by file existence; API returns 404 with actionable message.
4. **Untrusted uploads:** one validation module for all entry points; byte-size check before decode (DoS guard), pixel check after decode (decompression-bomb guard), distinct 413 vs 400.
5. **Optional LPIPS:** eval works fully without `torch`; `lpips_available()` gate + `None` propagation + UI caption.
6. **Reproducibility:** every experiment stores model/dataset/count/saturation/device/metrics/latency/git_commit.

---

## 7. Tradeoffs & “what would you improve?”

- Chose OpenCV DNN + Caffe over PyTorch: zero heavy deps, CPU-friendly, but stuck with 2016 quality and `opencv<5` pin.
- Chose classification-in-LAB over newer GAN/diffusion colorizers (DeOldify, DDColor): deterministic, lightweight, explainable; less vibrant on faces.
- SQLite over Postgres: fine for single-node history; would move to Postgres/S3 if multi-user hosted.
- If I had more time: add input face/skin prior, perceptual-loss fine-tune, ONNX export, async batch queue with job IDs, auth/rate-limit on API, Playwright screenshot tests, hosted demo link.

---

## 8. Likely Q&A (short answers)

1. **Why LAB not HSV?** LAB is perceptually uniform; Euclidean distance in ab ≈ perceived color distance. HSV hue is circular and unstable at low saturation.
2. **Why 224×224?** Backbone (VGG-style) fixed input; ab is low-frequency so upsampling is fine; L stays full-res for detail.
3. **What does saturation do?** Linear scale on predicted ab. Cheap user control, no re-inference.
4. **Rebalanced vs not?** Rebalanced upsamples rare colors during training → vivid but can overshoot; norebal → safe/muted.
5. **How do you handle huge images?** Reject >25 MP pre/post decode; uploads >20 MB rejected pre-decode.
6. **CPU vs GPU?** Auto-detect; same code path, just DNN backend switch.
7. **How do you test without weights?** Mock `Net.forward` to return zeros `(1,2,56,56)`; assert shape/dtype/passthrough. See `tests/test_pipeline.py`.
8. **Why `lru_cache` + `cache_clear` in benchmark?** Cache for prod speed; clear to measure true cold load.
9. **WAL mode why?** Readers don’t block writer; safer for concurrent UI/API reads.
10. **Why record git commit per experiment?** Tie metric to exact code+weights run.
11. **PSNR vs SSIM vs LPIPS?** Pixel fidelity vs structure vs learned perception; report all three, note LPIPS optional.
12. **How does batch ZIP work?** `BytesIO` + `ZIP_DEFLATED` in memory; no temp files.
13. **Compare slider implementation?** Two data-URI `<img>` stacked, top clipped via `clip-path: inset()` driven by range input; self-contained CSS/JS in iframe.
14. **API error mapping?** Validation→400, too large→413, bad model→404, inference exception→500 with logged traceback, never leak internals.
15. **Docker volumes why?** Keep weights read-only shared, history/logs/eval persistent across rebuilds.
16. **CI steps?** Compile → import/CLI smoke → unit tests → docker build; mocked inference keeps it fast/offline.
17. **How to add a third model?** Add `ModelSpec` to `MODEL_REGISTRY`, drop weights, extend download URL; UIs/API pick it up automatically.
18. **Biggest limitation?** Ill-posed: model guesses plausible hues; no semantic guarantee (e.g. exact shirt color).
19. **How to demo offline?** `python main.py` needs only local weights; no internet after setup.
20. **What would you monitor in prod?** Inference p95, error rate by code, upload size histogram, GPU util, experiment metric drift.

---

## 9. Commands to memorize

```bash
pip install -r requirements.txt
python scripts/download_models.py vibrant
python main.py
streamlit run streamlit_app.py
uvicorn api.main:app --reload
python scripts/evaluate.py --dataset evaluation/datasets/<name> --model vibrant --limit 20
python scripts/benchmark.py --images evaluation/datasets/<name> --model vibrant --iterations 3
python -m unittest discover -s tests
docker compose up
docker compose --profile api up
```

## 10. Repo facts that impress

- No duplicated colorization logic: 3 UIs + 2 CLIs call `colorize_image()`.
- Tests need no weights/GPU/net (`tests/test_*.py` mock the net).
- `.gitignore` keeps repo push-safe: `*.caffemodel`, `*.db`, `history/`, `logs/`, datasets/results excluded; `prototxt` + `pts_in_hull.npy` tracked.
- Env-configured paths make Docker/CI trivial without code changes.
