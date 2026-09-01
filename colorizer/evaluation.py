"""Quantitative evaluation of colorization quality against ground-truth color images.

Pipeline for each image:

    color image -> grayscale -> AI colorization -> predicted image
                                                          |
                                        compare against the original (PSNR/SSIM/LPIPS)

The original ground-truth images are never modified.

LPIPS is an optional metric: it requires the third-party ``lpips`` package
(and ``torch``, which it depends on), neither of which is a required project
dependency. When those packages are not installed, LPIPS scores are simply
reported as unavailable (``None``) and PSNR/SSIM continue to work normally.
"""
from __future__ import annotations

import csv
import json
import statistics
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .models import DEFAULT_MODEL_ID, device_name, load_model
from .pipeline import colorize_image

try:
    import lpips as _lpips_pkg
    import torch as _torch
except ImportError:  # pragma: no cover - exercised by test that forces the fallback
    _lpips_pkg = None
    _torch = None

SUPPORTED_EVAL_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

_lpips_model = None
_lpips_model_lock = threading.Lock()


def lpips_available() -> bool:
    """Whether the optional ``lpips``/``torch`` packages are installed."""
    return _lpips_pkg is not None


def _get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        with _lpips_model_lock:
            if _lpips_model is None:
                _lpips_model = _lpips_pkg.LPIPS(net="alex")
    return _lpips_model


@dataclass(frozen=True)
class ImageMetrics:
    filename: str
    width: int
    height: int
    model: str
    saturation: float
    device: str
    psnr: float
    ssim: float
    lpips: Optional[float]
    inference_time_seconds: float


@dataclass(frozen=True)
class DatasetMetrics:
    model: str
    dataset: str
    image_count: int
    device: str
    per_image: List[ImageMetrics]
    psnr_mean: float
    psnr_median: float
    ssim_mean: float
    ssim_median: float
    lpips_mean: Optional[float]
    latency_mean_seconds: float
    latency_median_seconds: float
    latency_p95_seconds: float

    @property
    def metrics(self) -> dict:
        return {
            "psnr_mean": self.psnr_mean,
            "psnr_median": self.psnr_median,
            "ssim_mean": self.ssim_mean,
            "ssim_median": self.ssim_median,
            "lpips_mean": self.lpips_mean,
        }

    @property
    def latency(self) -> dict:
        return {
            "mean_seconds": self.latency_mean_seconds,
            "median_seconds": self.latency_median_seconds,
            "p95_seconds": self.latency_p95_seconds,
        }

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "image_count": self.image_count,
            "device": self.device,
            "metrics": self.metrics,
            "latency": self.latency,
        }


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, pct))


def discover_images(dataset_dir) -> List[Path]:
    """Recursively find supported image files under ``dataset_dir``."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EVAL_EXTENSIONS
    )


def calculate_psnr(original_bgr: np.ndarray, predicted_bgr: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB; higher is better (identical images -> inf)."""
    return float(peak_signal_noise_ratio(original_bgr, predicted_bgr, data_range=255))


def calculate_ssim(original_bgr: np.ndarray, predicted_bgr: np.ndarray) -> float:
    """Structural similarity in [-1, 1]; 1.0 means identical images."""
    return float(
        structural_similarity(original_bgr, predicted_bgr, channel_axis=2, data_range=255)
    )


def calculate_lpips(original_bgr: np.ndarray, predicted_bgr: np.ndarray) -> Optional[float]:
    """Learned perceptual similarity; lower is better. Returns None if the optional dependency is missing."""
    if not lpips_available():
        return None

    def to_tensor(bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32") / 127.5 - 1.0
        return _torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    model = _get_lpips_model()
    with _torch.no_grad():
        distance = model(to_tensor(original_bgr), to_tensor(predicted_bgr))
    return float(distance.item())


def evaluate_image(
    image_path,
    model_id: str = DEFAULT_MODEL_ID,
    saturation: float = 1.0,
) -> ImageMetrics:
    """Grayscale a ground-truth color image, colorize it, and score the result against the original."""
    import time

    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"Could not decode image: {image_path}")

    grayscale = cv2.cvtColor(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    started = time.perf_counter()
    predicted = colorize_image(grayscale, model_id=model_id, saturation=saturation)
    elapsed = time.perf_counter() - started

    height, width = original.shape[:2]
    return ImageMetrics(
        filename=Path(image_path).name,
        width=width,
        height=height,
        model=model_id,
        saturation=saturation,
        device=device_name(),
        psnr=calculate_psnr(original, predicted),
        ssim=calculate_ssim(original, predicted),
        lpips=calculate_lpips(original, predicted),
        inference_time_seconds=elapsed,
    )


def _aggregate(model_id: str, dataset_name: str, per_image: List[ImageMetrics]) -> DatasetMetrics:
    psnrs = [m.psnr for m in per_image]
    ssims = [m.ssim for m in per_image]
    lpips_scores = [m.lpips for m in per_image if m.lpips is not None]
    latencies = [m.inference_time_seconds for m in per_image]
    device = per_image[0].device if per_image else device_name()

    return DatasetMetrics(
        model=model_id,
        dataset=dataset_name,
        image_count=len(per_image),
        device=device,
        per_image=per_image,
        psnr_mean=float(statistics.mean(psnrs)),
        psnr_median=float(statistics.median(psnrs)),
        ssim_mean=float(statistics.mean(ssims)),
        ssim_median=float(statistics.median(ssims)),
        lpips_mean=float(statistics.mean(lpips_scores)) if lpips_scores else None,
        latency_mean_seconds=float(statistics.mean(latencies)),
        latency_median_seconds=float(statistics.median(latencies)),
        latency_p95_seconds=_percentile(latencies, 95),
    )


def evaluate_dataset(
    dataset_dir,
    model_id: str = DEFAULT_MODEL_ID,
    saturation: float = 1.0,
    limit: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, ImageMetrics], None]] = None,
) -> DatasetMetrics:
    """Evaluate every supported image in ``dataset_dir`` and return aggregate + per-image metrics.

    Runs only when called explicitly (e.g. from the evaluate CLI or a dashboard
    button) -- never on import or on a schedule. Images are processed one at a
    time rather than loaded into memory as a batch.
    """
    images = discover_images(dataset_dir)
    if limit:
        images = images[:limit]
    if not images:
        raise ValueError(f"No supported images ({', '.join(SUPPORTED_EVAL_EXTENSIONS)}) found in {dataset_dir}")

    load_model(model_id)  # warm the cache once so the first image's latency isn't skewed by model loading

    per_image: List[ImageMetrics] = []
    for index, path in enumerate(images, start=1):
        metrics = evaluate_image(path, model_id=model_id, saturation=saturation)
        per_image.append(metrics)
        if progress_callback:
            progress_callback(index, len(images), metrics)

    return _aggregate(model_id, Path(dataset_dir).name, per_image)


def write_csv(per_image: List[ImageMetrics], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["filename", "width", "height", "model", "saturation", "device",
             "psnr", "ssim", "lpips", "inference_time_seconds"]
        )
        for m in per_image:
            writer.writerow(
                [m.filename, m.width, m.height, m.model, m.saturation, m.device,
                 m.psnr, m.ssim, m.lpips if m.lpips is not None else "", m.inference_time_seconds]
            )


def write_json(result: DatasetMetrics, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
