import os
import base64
import time
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from colorizer_core import colorize_image, load_model


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MAX_IMAGE_PIXELS = 25_000_000


st.set_page_config(page_title="AI Image Colorizer", page_icon="🎨", layout="wide")


@st.cache_resource
def cached_model():
    return load_model()


def decode_image(file_bytes):
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def to_pil(bgr_img):
    return Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))


def to_png_bytes(bgr_img):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def to_data_uri(bgr_img):
    buffer = BytesIO()
    to_pil(bgr_img).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def to_gray_data_uri(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return to_data_uri(gray_bgr)


def render_before_after_slider(original_bgr, colorized_bgr):
    grayscale_uri = to_gray_data_uri(original_bgr)
    colorized_uri = to_data_uri(colorized_bgr)

    html = """
    <div style="width: 100%; max-width: 1000px; margin: 0 auto; font-family: sans-serif;">
        <div style="position: relative; width: 100%; aspect-ratio: 16 / 9; overflow: hidden; border-radius: 18px; box-shadow: 0 10px 30px rgba(0,0,0,0.12); background: #111;">
            <img src="{colorized_uri}" style="position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover;" />
            <div id="compare-mask" style="position: absolute; inset: 0; width: 50%; overflow: hidden;">
                <img src="{grayscale_uri}" style="position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; filter: grayscale(100%);" />
            </div>
            <div id="compare-line" style="position: absolute; top: 0; left: 50%; width: 3px; height: 100%; background: rgba(255,255,255,0.95); box-shadow: 0 0 0 1px rgba(0,0,0,0.12); transform: translateX(-1.5px);"></div>
            <div id="compare-knob" style="position: absolute; top: 50%; left: 50%; width: 42px; height: 42px; border-radius: 50%; transform: translate(-50%, -50%); background: rgba(255,255,255,0.96); box-shadow: 0 8px 20px rgba(0,0,0,0.22); display: flex; align-items: center; justify-content: center; pointer-events: none;">
                <div style="width: 18px; height: 18px; border-left: 2px solid #333; border-right: 2px solid #333;"></div>
            </div>
            <input id="compare-slider" type="range" min="0" max="100" value="50" style="position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; cursor: ew-resize;" />
        </div>
    </div>
    <script>
        const slider = document.getElementById('compare-slider');
        const mask = document.getElementById('compare-mask');
        const line = document.getElementById('compare-line');
        const knob = document.getElementById('compare-knob');
        slider.addEventListener('input', (event) => {{
            const value = event.target.value + '%';
            mask.style.width = value;
            line.style.left = value;
            knob.style.left = value;
        }});
    </script>
    """.format(grayscale_uri=grayscale_uri, colorized_uri=colorized_uri)
    components.html(html, height=540)


def build_zip(results):
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, image_bytes in results:
            archive.writestr(name, image_bytes)
    buffer.seek(0)
    return buffer.getvalue()


def validate_upload(uploaded_file):
    if uploaded_file is None:
        return "No file was uploaded."

    if not uploaded_file.name.lower().endswith(VALID_EXTENSIONS):
        return "Unsupported file extension. Please upload JPG, JPEG, PNG, or BMP images."

    if not uploaded_file.getvalue():
        return "The uploaded file is empty."

    return None


def validate_image(image, filename):
    if image is None:
        return f"{filename}: this file could not be decoded as an image."

    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        return f"{filename}: image is too large to process safely."

    return None


def colorize_safe(image, filename):
    try:
        return colorize_image(image), None
    except Exception as exc:
        return None, f"{filename}: colorization failed ({exc})."


try:
    cached_model()
except FileNotFoundError:
    st.error(
        "Model files are missing. Make sure the Model/ folder contains the prototxt, caffemodel, and pts_in_hull.npy files."
    )
    st.stop()
except Exception as exc:
    st.error(f"Model could not be loaded: {exc}")
    st.stop()


st.title("AI Image Colorizer")
st.caption("Upload a grayscale or color photo to colorize it in the browser.")

tab_single, tab_batch = st.tabs(["Single image", "Batch upload"])

with tab_single:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded is None:
        st.info("Upload an image to start. Supported formats: JPG, JPEG, PNG, BMP.")
    else:
        upload_error = validate_upload(uploaded)
        if upload_error:
            st.error(upload_error)
        else:
            image = decode_image(uploaded.getvalue())
            image_error = validate_image(image, uploaded.name)
            if image_error:
                st.error(image_error)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original")
                    st.image(to_pil(image), use_container_width=True)

                if st.button("Colorize image", type="primary"):
                    with st.spinner("Colorizing..."):
                        result, error = colorize_safe(image, uploaded.name)

                    if error:
                        st.error(error)
                    else:
                        with col2:
                            st.subheader("Colorized")
                            st.download_button(
                                "Download PNG",
                                data=to_png_bytes(result),
                                file_name=f"{os.path.splitext(uploaded.name)[0]}_colorized.png",
                                mime="image/png",
                            )

                        st.subheader("Before / After Comparison")
                        render_before_after_slider(image, result)

with tab_batch:
    uploads = st.file_uploader(
        "Choose images for batch colorization",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )

    if not uploads:
        st.info("Upload one or more images to process them in batch.")
    elif st.button("Colorize all images", type="primary"):
        results = []
        failures = []
        preview_rows = st.container()
        stats_rows = []
        total_started = time.perf_counter()
        progress_bar = st.progress(0)
        progress_text = st.empty()
        total_uploads = len(uploads)

        for index, uploaded_file in enumerate(uploads, start=1):
            progress_text.write(f"Processing... {index - 1}/{total_uploads} complete")
            progress_bar.progress((index - 1) / total_uploads)

            file_started = time.perf_counter()
            upload_error = validate_upload(uploaded_file)
            if upload_error:
                failures.append(f"{uploaded_file.name}: {upload_error}")
                progress_text.write(f"Processing... {index}/{total_uploads} complete")
                progress_bar.progress(index / total_uploads)
                continue

            image = decode_image(uploaded_file.getvalue())
            image_error = validate_image(image, uploaded_file.name)
            if image_error:
                failures.append(image_error)
                progress_text.write(f"Processing... {index}/{total_uploads} complete")
                progress_bar.progress(index / total_uploads)
                continue

            with st.spinner(f"Processing {uploaded_file.name}..."):
                result, error = colorize_safe(image, uploaded_file.name)

            if error:
                failures.append(error)
                progress_text.write(f"Processing... {index}/{total_uploads} complete")
                progress_bar.progress(index / total_uploads)
                continue

            results.append((uploaded_file.name, to_png_bytes(result)))
            stats_rows.append({
                "name": uploaded_file.name,
                "width": image.shape[1],
                "height": image.shape[0],
                "time": time.perf_counter() - file_started,
            })
            with preview_rows:
                st.write(uploaded_file.name)
                st.image(
                    [to_pil(image), to_pil(result)],
                    caption=["Original", "Colorized"],
                    use_container_width=True,
                )

            progress_text.write(f"Processing... {index}/{total_uploads} complete")
            progress_bar.progress(index / total_uploads)

        progress_text.write(f"Processing complete: {len(results)}/{total_uploads} image(s) processed")
        progress_bar.progress(1.0)

        if results:
            st.success(f"Processed {len(results)} image(s) successfully.")
            st.download_button(
                "Download all as ZIP",
                data=build_zip([(f"{os.path.splitext(name)[0]}_colorized.png", data) for name, data in results]),
                file_name="colorized_results.zip",
                mime="application/zip",
            )

        if failures:
            st.warning("Some files could not be processed.")
            for failure in failures:
                st.write(failure)

        if not results and not failures:
            st.warning("No files were processed.")

        if stats_rows:
            total_time = time.perf_counter() - total_started
            average_time = total_time / len(stats_rows)
            first = stats_rows[0]
            last = stats_rows[-1]

            st.subheader("Processing Statistics")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("Images Processed", str(len(stats_rows)))
            metric_col2.metric("Average Time / Image", f"{average_time:.2f} s")
            metric_col3.metric("Total Time", f"{total_time:.1f} s")
            metric_col4.metric("Resolution", f"{last['width']}×{last['height']}")

            st.caption(
                f"First processed: {first['name']} | Last processed: {last['name']} | "
                f"Fastest image: {min(stats_rows, key=lambda row: row['time'])['time']:.2f} s"
            )