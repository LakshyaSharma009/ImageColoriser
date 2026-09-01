from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .config import MODEL_DIR

PROTO = MODEL_DIR / "colorization_deploy_v2.prototxt"
POINTS = MODEL_DIR / "pts_in_hull.npy"


def _require_file(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing required model file: {path}")
    return str(path)


def device_name() -> str:
    """Return 'CUDA' if OpenCV can run inference on a GPU, else 'CPU'. Never requires CUDA."""
    cuda_module = getattr(cv2, "cuda", None)
    if cuda_module is None:
        return "CPU"
    try:
        if cuda_module.getCudaEnabledDeviceCount() > 0:
            return "CUDA"
    except cv2.error:
        pass
    return "CPU"


def _use_gpu_if_available(net):
    """Switch the net to CUDA when OpenCV was built with CUDA support and a device is present."""
    if device_name() == "CUDA":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    description: str
    weights_filename: str
    required: bool = True

    @property
    def weights_path(self) -> Path:
        return MODEL_DIR / self.weights_filename


MODEL_REGISTRY = {
    "vibrant": ModelSpec(
        id="vibrant",
        label="Vibrant (rebalanced)",
        description="Class-rebalanced weights from the original release. Punchier, more saturated colors.",
        weights_filename="colorization_release_v2.caffemodel",
        required=True,
    ),
    "natural": ModelSpec(
        id="natural",
        label="Natural (muted)",
        description="Same architecture without class rebalancing. More conservative, natural-looking colors. "
        "Optional: fetch with scripts/download_models.py.",
        weights_filename="colorization_release_v2_norebal.caffemodel",
        required=False,
    ),
}

DEFAULT_MODEL_ID = "vibrant"


def available_models():
    """Model specs whose weights file actually exists on disk, so UIs never offer a broken choice."""
    return [spec for spec in MODEL_REGISTRY.values() if spec.weights_path.exists()]


@lru_cache(maxsize=None)
def load_model(model_id: str = DEFAULT_MODEL_ID):
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model id: {model_id!r}. Available: {list(MODEL_REGISTRY)}")

    if not hasattr(cv2.dnn, "readNetFromCaffe"):
        raise RuntimeError(
            "This OpenCV build does not support Caffe models. Install opencv-python<5 "
            "(for example, 4.11.0.86) and reinstall the project dependencies."
        )

    spec = MODEL_REGISTRY[model_id]
    net = cv2.dnn.readNetFromCaffe(_require_file(PROTO), _require_file(spec.weights_path))
    kernel = np.load(_require_file(POINTS))

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    _use_gpu_if_available(net)
    return net
