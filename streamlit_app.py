import base64
import logging
import os
import time
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from colorizer import (
    DEFAULT_MODEL_ID,
    VALID_EXTENSIONS,
    available_models,
    colorize_image,
    configure_logging,
    load_model,
)
from colorizer import evaluation, history
from colorizer.config import EVALUATION_DATASETS_DIR
from colorizer.validation import ValidationError, decode_image, validate_decoded_image, validate_upload_bytes

configure_logging()
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="AI Image Colorizer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 194, 102, 0.18), transparent 32%),
            radial-gradient(circle at 85% 10%, rgba(73, 173, 255, 0.16), transparent 28%),
            linear-gradient(180deg, #0e1117 0%, #121826 42%, #161d2b 100%);
        color: #f3f6fb;
    }

    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }

    .hero {
        padding: 2.1rem 2.2rem 1.4rem 2.2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(14, 17, 23, 0.88), rgba(22, 29, 43, 0.7));
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #8ebdff;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2.2rem, 5vw, 4.6rem);
        line-height: 0.95;
        font-weight: 800;
        color: #f8fbff;
    }

    .hero p {
        color: rgba(243, 246, 251, 0.78);
        font-size: 1.05rem;
        line-height: 1.6;
        max-width: 900px;
        margin-top: 1rem;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 1.3rem;
    }

    .metric-card {
        border-radius: 22px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem 1.1rem;
        backdrop-filter: blur(14px);
    }

    .metric-label {
        color: rgba(243, 246, 251, 0.62);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .metric-value {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }

    .section-card {
        border-radius: 26px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(12, 16, 25, 0.72);
        box-shadow: 0 16px 50px rgba(0, 0, 0, 0.22);
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .section-title {
        color: #f7f9fc;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .section-copy {
        color: rgba(243, 246, 251, 0.66);
        margin-bottom: 1rem;
    }

    .image-frame {
        border-radius: 22px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.03);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    .download-note {
        color: rgba(243, 246, 251, 0.64);
        font-size: 0.9rem;
        margin-top: 0.6rem;
    }

    div[data-testid="stSlider"] label {
        color: rgba(243, 246, 251, 0.92) !important;
        font-weight: 600 !important;
    }

    .stButton > button {
        border-radius: 999px;
        border: none;
        padding: 0.7rem 1.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #66b3ff 0%, #8f7bff 100%);
        color: white;
        box-shadow: 0 16px 28px rgba(86, 131, 255, 0.25);
    }

    .stDownloadButton > button {
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 0.7rem 1.2rem;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.05);
        color: #f5f8ff;
    }
</style>
""",
    unsafe_allow_html=True,
)


def output_name_for(uploaded_name):
    return f"{os.path.splitext(uploaded_name)[0]}_colorized.png"


def to_pil(bgr_img):
    return Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))


def to_png_bytes(bgr_img):
    buffer = BytesIO()
    to_pil(bgr_img).save(buffer, format="PNG")
    return buffer.getvalue()


def to_data_uri(bgr_img):
    buffer = BytesIO()
    to_pil(bgr_img).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_zip(results):
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, image_bytes in results:
            archive.writestr(name, image_bytes)
    buffer.seek(0)
    return buffer.getvalue()


def validate_upload(uploaded_file):
    """Thin Streamlit-facing wrapper around the shared colorizer.validation checks."""
    if uploaded_file is None:
        return "No file was uploaded."
    try:
        validate_upload_bytes(uploaded_file.getvalue(), uploaded_file.name)
    except ValidationError as exc:
        return str(exc)
    return None


def validate_image(image, filename):
    try:
        validate_decoded_image(image)
    except ValidationError as exc:
        return f"{filename}: {exc}"
    return None


def colorize_safe(image, filename, model_id=DEFAULT_MODEL_ID, saturation=1.0):
    try:
        return colorize_image(image, model_id=model_id, saturation=saturation), None
    except Exception as exc:
        logger.exception("Colorization failed for %s", filename)
        return None, f"{filename}: colorization failed ({exc})."


def render_header():
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">AI image colorization</div>
            <h1>Turn grayscale photos into vivid color with a cleaner, faster workflow.</h1>
            <p>
                Upload a single image or a whole folder of black-and-white photos, preview the result with a proper
                before-and-after slider, and download PNG or ZIP outputs from one focused interface.
            </p>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Single upload</div>
                    <div class="metric-value">Interactive preview</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Batch mode</div>
                    <div class="metric-value">ZIP export</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Comparison</div>
                    <div class="metric-value">Proper slider</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Model</div>
                    <div class="metric-value">OpenCV DNN</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compare_slider(original_bgr, colorized_bgr):
    st.markdown(
        '<div class="section-title">Before / After</div><div class="section-copy">Drag the slider to compare the original grayscale photo with the restored color version.</div>',
        unsafe_allow_html=True,
    )
    grayscale = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    grayscale_bgr = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

    left_uri = to_data_uri(grayscale_bgr)
    right_uri = to_data_uri(colorized_bgr)

    img_height, img_width = colorized_bgr.shape[:2]
    aspect_ratio = img_width / img_height if img_height else 16 / 10
    assumed_width = 480
    component_height = int(min(680, max(260, assumed_width / aspect_ratio))) + 90

    # Rendered via components.html, which mounts this markup in its own sandboxed
    # iframe document, so it cannot see the page's <style> block above and needs
    # every rule it uses defined inline here.
    html = f"""
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        .compare-wrap {{
            position: relative;
            width: 100%;
            aspect-ratio: {img_width} / {img_height};
            overflow: hidden;
            border-radius: 22px;
            background: #0a0d12;
            border: 1px solid rgba(255, 255, 255, 0.08);
            user-select: none;
        }}
        .compare-layer {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            pointer-events: none;
        }}
        .compare-handle {{
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 48px;
            height: 48px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: 0 14px 35px rgba(0, 0, 0, 0.28);
            display: flex;
            align-items: center;
            justify-content: center;
            pointer-events: none;
            z-index: 2;
        }}
        .compare-handle::before {{
            content: "\\2194";
            color: #182033;
            font-size: 1.1rem;
            font-weight: 700;
        }}
        .compare-seam {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 2px;
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 0 0 12px rgba(0, 0, 0, 0.25);
            transform: translateX(-1px);
            pointer-events: none;
        }}
        .compare-label {{
            position: absolute;
            top: 12px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            background: rgba(10, 13, 18, 0.6);
            color: #f3f6fb;
            pointer-events: none;
        }}
        .compare-label.left {{ left: 12px; }}
        .compare-label.right {{ right: 12px; }}
        .compare-slider {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: ew-resize;
            z-index: 3;
            margin: 0;
        }}
        .download-note {{
            color: rgba(230, 235, 245, 0.8);
            font-size: 0.85rem;
            margin-top: 0.7rem;
        }}
    </style>
    <div class="compare-wrap" id="compare-wrap">
        <img class="compare-layer" src="{right_uri}" alt="Colorized image" />
        <span class="compare-label right">After</span>
        <img class="compare-layer" id="compare-top-img" src="{left_uri}" alt="Original image"
             style="clip-path: inset(0 50% 0 0);" />
        <span class="compare-label left">Before</span>
        <div class="compare-seam" id="compare-seam" style="left: 50%;"></div>
        <div class="compare-handle" id="compare-handle" style="left: 50%;"></div>
        <input id="compare-slider" class="compare-slider" type="range" min="0" max="100" value="50" step="0.1"
               aria-label="Before and after comparison slider" />
    </div>
    <div class="download-note">Drag inside the image, or use the slider, to reveal the restored color version.</div>
    <script>
        const slider = document.getElementById('compare-slider');
        const topImg = document.getElementById('compare-top-img');
        const seam = document.getElementById('compare-seam');
        const handle = document.getElementById('compare-handle');

        function updateCompare(value) {{
            topImg.style.clipPath = `inset(0 ${{100 - value}}% 0 0)`;
            seam.style.left = value + '%';
            handle.style.left = value + '%';
        }}

        slider.addEventListener('input', (event) => updateCompare(event.target.value));
        updateCompare(slider.value);
    </script>
    """
    components.html(html, height=component_height)


def render_single_mode(model_id, saturation):
    st.markdown(
        '<div class="section-title">Single image studio</div><div class="section-copy">Upload one image, inspect it, then colorize and compare.</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose a grayscale or color image",
        type=[ext.lstrip(".") for ext in VALID_EXTENSIONS],
        key="single_upload",
    )

    if uploaded is None:
        st.info("Drop in a JPG, PNG, or BMP to start.")
        return

    upload_error = validate_upload(uploaded)
    if upload_error:
        st.error(upload_error)
        return

    image = decode_image(uploaded.getvalue())
    image_error = validate_image(image, uploaded.name)
    if image_error:
        st.error(image_error)
        return

    preview_col, action_col = st.columns([1.05, 0.95], gap="large")
    with preview_col:
        st.markdown('<div class="section-title">Original preview</div>', unsafe_allow_html=True)
        st.image(to_pil(image), use_container_width=True)

    with action_col:
        st.markdown('<div class="section-title">Colorize</div>', unsafe_allow_html=True)
        st.write("Generate the colorized version and download it as a PNG.")
        if st.button("Colorize image", type="primary", use_container_width=True):
            with st.spinner("Colorizing image..."):
                result, error = colorize_safe(image, uploaded.name, model_id, saturation)

            if error:
                st.error(error)
                return

            st.markdown('<div class="section-title">Colorized result</div>', unsafe_allow_html=True)
            st.image(to_pil(result), use_container_width=True)
            st.download_button(
                "Download PNG",
                data=to_png_bytes(result),
                file_name=output_name_for(uploaded.name),
                mime="image/png",
                use_container_width=True,
            )
            st.markdown(
                '<div class="download-note">Use the slider below to compare the generated image with the original.</div>',
                unsafe_allow_html=True,
            )
            render_compare_slider(image, result)


def render_batch_mode(model_id, saturation):
    st.markdown(
        '<div class="section-title">Batch workspace</div><div class="section-copy">Process multiple files in one run and export a ZIP archive.</div>',
        unsafe_allow_html=True,
    )
    uploads = st.file_uploader(
        "Choose one or more images",
        type=[ext.lstrip(".") for ext in VALID_EXTENSIONS],
        accept_multiple_files=True,
        key="batch_uploads",
    )

    if not uploads:
        st.info("Upload multiple images to build a batch export.")
        return

    header_col, button_col = st.columns([0.7, 0.3], gap="medium")
    with header_col:
        st.caption(f"{len(uploads)} file(s) selected")
    with button_col:
        run_batch = st.button("Colorize all", type="primary", use_container_width=True)

    if not run_batch:
        st.markdown(
            '<div class="section-copy">Preview the selected files below, then run the batch job when ready.</div>',
            unsafe_allow_html=True,
        )
        for uploaded_file in uploads:
            st.write(uploaded_file.name)
        return

    results = []
    failures = []
    previews = []
    stats_rows = []
    total_started = time.perf_counter()
    progress = st.progress(0)
    status = st.empty()

    for index, uploaded_file in enumerate(uploads, start=1):
        status.write(f"Processing {index}/{len(uploads)}: {uploaded_file.name}")
        progress.progress((index - 1) / len(uploads))

        upload_error = validate_upload(uploaded_file)
        if upload_error:
            failures.append(f"{uploaded_file.name}: {upload_error}")
            continue

        image = decode_image(uploaded_file.getvalue())
        image_error = validate_image(image, uploaded_file.name)
        if image_error:
            failures.append(image_error)
            continue

        file_started = time.perf_counter()
        with st.spinner(f"Colorizing {uploaded_file.name}..."):
            result, error = colorize_safe(image, uploaded_file.name, model_id, saturation)

        if error:
            failures.append(error)
            continue

        results.append((output_name_for(uploaded_file.name), to_png_bytes(result)))
        stats_rows.append(
            {
                "name": uploaded_file.name,
                "width": image.shape[1],
                "height": image.shape[0],
                "time": time.perf_counter() - file_started,
            }
        )
        previews.append((uploaded_file.name, image, result))
        progress.progress(index / len(uploads))

    status.write(f"Finished {len(results)}/{len(uploads)} file(s).")

    if results:
        st.success(f"Processed {len(results)} image(s) successfully.")
        st.download_button(
            "Download ZIP",
            data=build_zip(results),
            file_name="colorized_results.zip",
            mime="application/zip",
            use_container_width=True,
        )

    if failures:
        st.warning("Some files could not be processed.")
        for failure in failures:
            st.write(failure)

    if previews:
        st.markdown('<div class="section-title">Batch previews</div>', unsafe_allow_html=True)
        for name, original, result in previews:
            with st.expander(name, expanded=False):
                left_col, right_col = st.columns(2)
                with left_col:
                    st.image(to_pil(original), caption="Original", use_container_width=True)
                with right_col:
                    st.image(to_pil(result), caption="Colorized", use_container_width=True)

    if stats_rows:
        total_time = time.perf_counter() - total_started
        average_time = total_time / len(stats_rows)
        fastest = min(stats_rows, key=lambda row: row["time"])
        slowest = max(stats_rows, key=lambda row: row["time"])
        first = stats_rows[0]

        st.markdown('<div class="section-title">Processing summary</div>', unsafe_allow_html=True)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Images processed", str(len(stats_rows)))
        metric_cols[1].metric("Average / image", f"{average_time:.2f} s")
        metric_cols[2].metric("Total time", f"{total_time:.1f} s")
        metric_cols[3].metric("Largest image", f"{first['width']}×{first['height']}")
        st.caption(
            f"First: {first['name']} | Fastest: {fastest['name']} ({fastest['time']:.2f} s) | Slowest: {slowest['name']} ({slowest['time']:.2f} s)"
        )


def _difference_image(original_bgr, predicted_bgr):
    """Amplified absolute-difference visualization; does not alter either input image."""
    diff = cv2.absdiff(original_bgr, predicted_bgr)
    return cv2.convertScaleAbs(diff, alpha=3.0)


def _available_dataset_names():
    if not EVALUATION_DATASETS_DIR.exists():
        return []
    return sorted(p.name for p in EVALUATION_DATASETS_DIR.iterdir() if p.is_dir())


def render_evaluation_mode():
    st.markdown(
        '<div class="section-title">Evaluation</div>'
        '<div class="section-copy">Measure colorization quality against a dataset of real color photos: '
        'each image is converted to grayscale, colorized, and compared back against the original with '
        'PSNR and SSIM (LPIPS too, if the optional dependency is installed). Runs only when you click '
        '"Run evaluation" below.</div>',
        unsafe_allow_html=True,
    )

    dataset_names = _available_dataset_names()
    if not dataset_names:
        st.info(
            f"No datasets found under `{EVALUATION_DATASETS_DIR}`. Create a folder there "
            "(e.g. `evaluation/datasets/sample/`) containing ground-truth color JPG/PNG/WEBP images, then reload this page."
        )
        return

    models = available_models()
    model_ids = [spec.id for spec in models]

    col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
    with col1:
        dataset_name = st.selectbox("Dataset", dataset_names, key="eval_dataset")
    with col2:
        selected_models = st.multiselect(
            "Model(s) to evaluate", model_ids, default=model_ids[:1], key="eval_models"
        )
    with col3:
        limit = st.number_input("Limit", min_value=0, value=20, step=5, key="eval_limit", help="0 = evaluate all images")

    eval_saturation = st.slider("Evaluation saturation", 0.0, 2.0, 1.0, 0.1, key="eval_saturation")

    if st.button("Run evaluation", type="primary"):
        if not selected_models:
            st.warning("Select at least one model to evaluate.")
        else:
            dataset_dir = EVALUATION_DATASETS_DIR / dataset_name
            results = {}
            for mid in selected_models:
                with st.spinner(f"Evaluating {mid} on {dataset_name}..."):
                    try:
                        result = evaluation.evaluate_dataset(
                            dataset_dir, model_id=mid, saturation=eval_saturation, limit=limit or None
                        )
                    except ValueError as exc:
                        st.error(f"{mid}: {exc}")
                        continue
                results[mid] = result
                try:
                    history.init_db()
                    history.record_experiment(
                        model=result.model,
                        dataset=result.dataset,
                        image_count=result.image_count,
                        saturation=eval_saturation,
                        device=result.device,
                        psnr_mean=result.psnr_mean,
                        ssim_mean=result.ssim_mean,
                        lpips_mean=result.lpips_mean,
                        latency_mean_seconds=result.latency_mean_seconds,
                        latency_median_seconds=result.latency_median_seconds,
                        latency_p95_seconds=result.latency_p95_seconds,
                    )
                except Exception:
                    logger.exception("Failed to record experiment")
            st.session_state["eval_results"] = results
            st.session_state["eval_results_dataset"] = dataset_name

    results = st.session_state.get("eval_results")
    if results and st.session_state.get("eval_results_dataset") == dataset_name:
        st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
        table_rows = [
            {
                "Model": mid,
                "Images": result.image_count,
                "PSNR (dB)": round(result.psnr_mean, 2),
                "SSIM": round(result.ssim_mean, 4),
                "LPIPS": round(result.lpips_mean, 4) if result.lpips_mean is not None else "n/a",
                "Avg latency (s)": round(result.latency_mean_seconds, 3),
            }
            for mid, result in results.items()
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
        if not evaluation.lpips_available():
            st.caption("LPIPS: unavailable (optional 'lpips'/'torch' packages not installed).")

        st.markdown('<div class="section-title">Metric distributions</div>', unsafe_allow_html=True)
        for mid, result in results.items():
            dist_col1, dist_col2 = st.columns(2)
            with dist_col1:
                st.caption(f"{mid} — PSNR per image (dB)")
                st.bar_chart({"psnr": [m.psnr for m in result.per_image]})
            with dist_col2:
                st.caption(f"{mid} — SSIM per image")
                st.bar_chart({"ssim": [m.ssim for m in result.per_image]})

        st.markdown('<div class="section-title">Example predictions</div>', unsafe_allow_html=True)
        preview_model = next(iter(results))
        dataset_dir = EVALUATION_DATASETS_DIR / dataset_name
        for path in evaluation.discover_images(dataset_dir)[:3]:
            original = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if original is None:
                continue
            grayscale = cv2.cvtColor(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            predicted, error = colorize_safe(grayscale, path.name, preview_model, eval_saturation)
            if error:
                st.warning(error)
                continue
            with st.expander(path.name, expanded=False):
                cols = st.columns(4)
                cols[0].image(to_pil(original), caption="Original", use_container_width=True)
                cols[1].image(to_pil(grayscale), caption="Grayscale input", use_container_width=True)
                cols[2].image(to_pil(predicted), caption="AI prediction", use_container_width=True)
                cols[3].image(to_pil(_difference_image(original, predicted)), caption="Difference", use_container_width=True)

    st.markdown('<div class="section-title">Experiment history</div>', unsafe_allow_html=True)
    try:
        history.init_db()
        experiments = history.list_experiments(limit=25)
    except Exception:
        logger.exception("Failed to load experiment history")
        experiments = []
    if experiments:
        st.dataframe(experiments, use_container_width=True, hide_index=True)
    else:
        st.caption("No experiments recorded yet. Run an evaluation above to populate this table.")


def render_sidebar():
    with st.sidebar:
        st.markdown("### About this build")
        st.write(
            "This frontend was rebuilt to be more focused, more visual, and easier to use on both desktop and laptop screens."
        )
        st.markdown("### Tips")
        st.write("Use the single-image studio for one-off edits and the batch workspace for folders of similar images.")
        st.write("Use the Evaluation tab to score colorization quality (PSNR/SSIM) against your own dataset of color photos.")
        st.markdown("### Requirements")
        st.write("The model needs OpenCV with Caffe support, so `opencv-python<5` is required.")


def main():
    models = available_models()
    if not models:
        st.error(
            "No model weights found. Make sure the Model/ folder contains at least one caffemodel file "
            "(see scripts/download_models.py)."
        )
        st.stop()

    render_sidebar()
    render_header()

    control_col, saturation_col = st.columns([0.6, 0.4], gap="large")
    with control_col:
        model_id = st.selectbox(
            "Model",
            options=[spec.id for spec in models],
            format_func=lambda spec_id: next(spec.label for spec in models if spec.id == spec_id),
            index=[spec.id for spec in models].index(DEFAULT_MODEL_ID)
            if DEFAULT_MODEL_ID in [spec.id for spec in models]
            else 0,
        )
    with saturation_col:
        saturation = st.slider("Saturation", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

    try:
        load_model(model_id)
    except FileNotFoundError:
        st.error(
            "Model files are missing. Make sure the Model/ folder contains the prototxt, caffemodel, and pts_in_hull.npy files."
        )
        st.stop()
    except Exception as exc:
        st.error(f"Model could not be loaded: {exc}")
        st.stop()

    mode = st.radio(
        "Choose a workspace",
        ["Single image studio", "Batch workspace", "Evaluation"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if mode == "Single image studio":
        render_single_mode(model_id, saturation)
    elif mode == "Batch workspace":
        render_batch_mode(model_id, saturation)
    else:
        render_evaluation_mode()
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()