"""FastAPI REST API for the AI Image Colorizer.

Thin HTTP layer over the same `colorizer` core library used by the Streamlit
and Tkinter UIs -- no separate business logic, validation, or model loading
is implemented here. Run locally with:

    uvicorn api.main:app --reload
"""
import logging
import time

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse, Response

from colorizer import configure_logging, history
from colorizer.models import DEFAULT_MODEL_ID, MODEL_REGISTRY, available_models, device_name
from colorizer.pipeline import colorize_image
from colorizer.validation import PayloadTooLargeError, ValidationError, validate_and_decode_upload

from .schemas import (
    ExperimentEntry,
    ExperimentsResponse,
    HealthResponse,
    HistoryEntry,
    HistoryResponse,
    ModelInfo,
    ModelsResponse,
)

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Image Colorizer API",
    description="REST API for CNN-based grayscale image colorization.",
    version="1.0.0",
)


def _encode_png(bgr_image) -> bytes:
    success, buffer = cv2.imencode(".png", bgr_image)
    if not success:
        raise RuntimeError("Failed to encode output image as PNG.")
    return buffer.tobytes()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Visiting the bare API URL in a browser is more useful pointed at the docs than a 404."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        models=[spec.id for spec in available_models()],
        device=device_name(),
    )


@app.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    available_ids = {spec.id for spec in available_models()}
    return ModelsResponse(
        models=[
            ModelInfo(id=spec.id, label=spec.label, available=spec.id in available_ids)
            for spec in MODEL_REGISTRY.values()
        ]
    )


@app.post("/colorize", responses={200: {"content": {"image/png": {}}}})
async def colorize(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL_ID),
    saturation: float = Form(1.0),
):
    if model not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model id: {model!r}.")
    if model not in {spec.id for spec in available_models()}:
        raise HTTPException(status_code=404, detail=f"Model '{model}' weights are not available on this server.")

    data = await file.read()
    try:
        image = validate_and_decode_upload(data, file.filename or "")
    except PayloadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = time.perf_counter()
    try:
        result = colorize_image(image, model_id=model, saturation=saturation)
    except Exception:
        logger.exception("Colorization failed for upload %s", file.filename)
        raise HTTPException(status_code=500, detail="Colorization failed unexpectedly.") from None
    elapsed = time.perf_counter() - started

    png_bytes = _encode_png(result)
    logger.info("Colorized %s with model=%s in %.3fs", file.filename, model, elapsed)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"X-Model": model, "X-Processing-Time": f"{elapsed:.3f}"},
    )


@app.get("/history", response_model=HistoryResponse)
def get_history(limit: int = Query(50, ge=1, le=500)) -> HistoryResponse:
    """List past colorization runs. `output_path` is a server-local filesystem
    path, so it's intentionally left out of HistoryEntry rather than returned."""
    history.init_db()
    rows = history.list_runs(limit=limit)
    return HistoryResponse(runs=[HistoryEntry(**row) for row in rows])


@app.get("/experiments", response_model=ExperimentsResponse)
def get_experiments(limit: int = Query(50, ge=1, le=500)) -> ExperimentsResponse:
    history.init_db()
    rows = history.list_experiments(limit=limit)
    return ExperimentsResponse(experiments=[ExperimentEntry(**row) for row in rows])
