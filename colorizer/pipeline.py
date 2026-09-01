import threading

import cv2
import numpy as np

from . import models
from .models import DEFAULT_MODEL_ID

_locks_guard = threading.Lock()
_inference_locks = {}


def _lock_for(model_id):
    with _locks_guard:
        lock = _inference_locks.get(model_id)
        if lock is None:
            lock = threading.Lock()
            _inference_locks[model_id] = lock
        return lock


def colorize_image(img, model_id: str = DEFAULT_MODEL_ID, saturation: float = 1.0):
    """Takes a BGR uint8 image, returns a BGR uint8 colorized image."""
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("colorize_image expects a BGR image with three channels.")

    net = models.load_model(model_id)

    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab_img, (224, 224))
    l_small = cv2.split(resized)[0]
    l_small -= 50

    blob = cv2.dnn.blobFromImage(l_small)

    # cv2.dnn.Net is not documented as thread-safe; a shared cached net can be hit
    # concurrently by different Streamlit sessions, so serialize setInput+forward per model.
    with _lock_for(model_id):
        net.setInput(blob)
        ab_channel = net.forward()

    ab_channel = ab_channel[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

    if saturation != 1.0:
        ab_channel = ab_channel * saturation

    l_original = cv2.split(lab_img)[0]
    colorized = np.concatenate((l_original[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized.astype("float32"), cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    return (255 * colorized).astype("uint8")
