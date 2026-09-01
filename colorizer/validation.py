"""Shared validation for user-supplied image uploads.

Uploaded images are untrusted input (from the Streamlit UI, the REST API, or
anywhere else), so every entry point should route through this module instead
of re-implementing its own size/extension/decode checks.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES, VALID_EXTENSIONS


class ValidationError(ValueError):
    """Raised when untrusted image input fails validation."""


class PayloadTooLargeError(ValidationError):
    """Raised specifically when an upload exceeds the configured byte-size limit."""


def validate_filename(filename: str) -> None:
    if not filename or not filename.lower().endswith(VALID_EXTENSIONS):
        raise ValidationError(
            f"Unsupported file extension. Allowed: {', '.join(VALID_EXTENSIONS)}"
        )


def validate_upload_bytes(data: bytes, filename: str) -> None:
    """Validate raw upload bytes before attempting to decode them as an image."""
    validate_filename(filename)
    if not data:
        raise ValidationError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            f"File is too large ({len(data)} bytes). Maximum allowed is {MAX_UPLOAD_BYTES} bytes."
        )


def decode_image(data: bytes) -> Optional[np.ndarray]:
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def validate_decoded_image(image: Optional[np.ndarray]) -> None:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValidationError("The file could not be decoded as an image.")
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValidationError(
            f"Image is too large to process safely (max {MAX_IMAGE_PIXELS} pixels)."
        )


def validate_and_decode_upload(data: bytes, filename: str) -> np.ndarray:
    """Full validation pipeline for an untrusted uploaded file: size/extension, then decode, then dimensions."""
    validate_upload_bytes(data, filename)
    image = decode_image(data)
    validate_decoded_image(image)
    return image
