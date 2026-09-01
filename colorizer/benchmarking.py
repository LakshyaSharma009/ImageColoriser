"""Latency/throughput benchmarking for the colorization pipeline.

Runs only when explicitly invoked (the benchmark CLI, or a future dashboard
button) -- never automatically. Inference stays single-threaded, reusing the
same per-model lock and cached ``cv2.dnn.Net`` as normal request handling; no
extra thread/process pools are introduced.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from .models import DEFAULT_MODEL_ID, device_name, load_model
from .pipeline import colorize_image


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    device: str
    image_count: int
    iterations: int
    model_load_seconds: float
    mean_inference_seconds: float
    median_inference_seconds: float
    p95_inference_seconds: float
    images_per_second: float
    per_image_seconds: List[float]

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "device": self.device,
            "image_count": self.image_count,
            "iterations": self.iterations,
            "model_load_seconds": self.model_load_seconds,
            "mean_inference_seconds": self.mean_inference_seconds,
            "median_inference_seconds": self.median_inference_seconds,
            "p95_inference_seconds": self.p95_inference_seconds,
            "images_per_second": self.images_per_second,
        }


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(list(values), pct))


def measure_model_load_time(model_id: str = DEFAULT_MODEL_ID) -> float:
    """Time a cold load of ``model_id``, distinct from inference time.

    ``load_model`` is cached, so this clears that cache first. Intended for
    dedicated benchmarking runs; not called during normal request handling.
    """
    load_model.cache_clear()
    started = time.perf_counter()
    load_model(model_id)
    return time.perf_counter() - started


def benchmark_images(
    image_paths: Sequence,
    model_id: str = DEFAULT_MODEL_ID,
    saturation: float = 1.0,
    iterations: int = 3,
    warmup: bool = True,
) -> BenchmarkResult:
    """Benchmark inference latency/throughput over a fixed set of images.

    Images are decoded once up front (benchmarking measures inference, not
    disk I/O) and reused across iterations; nothing is written to disk.
    """
    images = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)
    if not images:
        raise ValueError("No valid images to benchmark.")

    model_load_seconds = measure_model_load_time(model_id)

    if warmup:
        colorize_image(images[0], model_id=model_id, saturation=saturation)

    per_image_seconds: List[float] = []
    for _ in range(iterations):
        for image in images:
            started = time.perf_counter()
            colorize_image(image, model_id=model_id, saturation=saturation)
            per_image_seconds.append(time.perf_counter() - started)

    total_seconds = sum(per_image_seconds)
    images_per_second = len(per_image_seconds) / total_seconds if total_seconds > 0 else 0.0

    return BenchmarkResult(
        model=model_id,
        device=device_name(),
        image_count=len(images),
        iterations=iterations,
        model_load_seconds=model_load_seconds,
        mean_inference_seconds=float(statistics.mean(per_image_seconds)),
        median_inference_seconds=float(statistics.median(per_image_seconds)),
        p95_inference_seconds=_percentile(per_image_seconds, 95),
        images_per_second=images_per_second,
        per_image_seconds=per_image_seconds,
    )
