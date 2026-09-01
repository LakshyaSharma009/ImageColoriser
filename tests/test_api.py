import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from colorizer.models import MODEL_REGISTRY

client = TestClient(app)


def _png_bytes(size=(10, 10), color=(120, 120, 120)):
    img = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class HealthTests(unittest.TestCase):
    @patch("api.main.device_name", return_value="CPU")
    @patch("api.main.available_models")
    def test_health_reports_status_models_and_device(self, mock_available, mock_device):
        mock_available.return_value = [MODEL_REGISTRY["vibrant"]]

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "healthy")
        self.assertEqual(body["models"], ["vibrant"])
        self.assertEqual(body["device"], "CPU")

    @patch("api.main.available_models", return_value=[])
    def test_health_reports_no_models_when_none_available(self, mock_available):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["models"], [])


class ModelsTests(unittest.TestCase):
    @patch("api.main.available_models")
    def test_models_endpoint_reflects_availability_dynamically(self, mock_available):
        mock_available.return_value = [MODEL_REGISTRY["vibrant"]]

        response = client.get("/models")

        self.assertEqual(response.status_code, 200)
        availability = {m["id"]: m["available"] for m in response.json()["models"]}
        self.assertTrue(availability["vibrant"])
        self.assertFalse(availability["natural"])


class ColorizeValidationTests(unittest.TestCase):
    @patch("api.main.available_models")
    def test_colorize_requires_a_file(self, mock_available):
        mock_available.return_value = [MODEL_REGISTRY["vibrant"]]
        response = client.post("/colorize", data={"model": "vibrant"})
        self.assertEqual(response.status_code, 422)

    @patch("api.main.available_models")
    def test_colorize_rejects_unsupported_extension(self, mock_available):
        mock_available.return_value = [MODEL_REGISTRY["vibrant"]]
        response = client.post(
            "/colorize",
            files={"file": ("photo.txt", b"hello", "text/plain")},
            data={"model": "vibrant"},
        )
        self.assertEqual(response.status_code, 400)

    @patch("api.main.available_models")
    def test_colorize_rejects_undecodable_image(self, mock_available):
        mock_available.return_value = [MODEL_REGISTRY["vibrant"]]
        response = client.post(
            "/colorize",
            files={"file": ("photo.jpg", b"not-an-image", "image/jpeg")},
            data={"model": "vibrant"},
        )
        self.assertEqual(response.status_code, 400)

    @patch("api.main.available_models")
    def test_colorize_rejects_oversized_upload(self, mock_available):
        mock_available.return_value = [MODEL_REGISTRY["vibrant"]]
        oversized = b"\x00" * (21 * 1024 * 1024)  # exceeds the default 20MB limit
        response = client.post(
            "/colorize",
            files={"file": ("photo.jpg", oversized, "image/jpeg")},
            data={"model": "vibrant"},
        )
        self.assertEqual(response.status_code, 413)

    @patch("api.main.available_models")
    def test_colorize_rejects_unknown_model_id(self, mock_available):
        mock_available.return_value = [MODEL_REGISTRY["vibrant"]]
        response = client.post(
            "/colorize",
            files={"file": ("photo.jpg", _png_bytes(), "image/jpeg")},
            data={"model": "not-a-real-model"},
        )
        self.assertEqual(response.status_code, 404)

    @patch("api.main.available_models", return_value=[])
    def test_colorize_rejects_model_with_missing_weights(self, mock_available):
        response = client.post(
            "/colorize",
            files={"file": ("photo.jpg", _png_bytes(), "image/jpeg")},
            data={"model": "vibrant"},
        )
        self.assertEqual(response.status_code, 404)


class ColorizeSuccessTests(unittest.TestCase):
    @patch("api.main.colorize_image")
    @patch("api.main.available_models")
    def test_colorize_returns_png_with_metadata_headers(self, mock_available, mock_colorize):
        mock_available.return_value = [MODEL_REGISTRY["vibrant"]]
        mock_colorize.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        response = client.post(
            "/colorize",
            files={"file": ("photo.jpg", _png_bytes(), "image/jpeg")},
            data={"model": "vibrant", "saturation": "1.2"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.headers["x-model"], "vibrant")
        self.assertIn("x-processing-time", response.headers)
        mock_colorize.assert_called_once()

    @patch("api.main.colorize_image", side_effect=RuntimeError("boom"))
    @patch("api.main.available_models")
    def test_colorize_failure_returns_500_without_leaking_traceback(self, mock_available, mock_colorize):
        mock_available.return_value = [MODEL_REGISTRY["vibrant"]]

        response = client.post(
            "/colorize",
            files={"file": ("photo.jpg", _png_bytes(), "image/jpeg")},
            data={"model": "vibrant"},
        )

        self.assertEqual(response.status_code, 500)
        detail = response.json()["detail"]
        self.assertNotIn("Traceback", detail)
        self.assertNotIn("boom", detail)


class HistoryAndExperimentsEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_patch = patch("colorizer.history.HISTORY_DB_PATH", Path(self.tmp_dir.name) / "history.db")
        self.output_patch = patch("colorizer.history.HISTORY_OUTPUT_DIR", Path(self.tmp_dir.name) / "outputs")
        self.db_patch.start()
        self.output_patch.start()

    def tearDown(self):
        self.db_patch.stop()
        self.output_patch.stop()
        self.tmp_dir.cleanup()

    def test_history_endpoint_returns_empty_list_structure(self):
        response = client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"runs": []})

    def test_history_endpoint_never_exposes_output_path(self):
        from colorizer import history

        history.init_db()
        history.record_run(
            filename="photo.jpg", model_id="vibrant", saturation=1.0,
            width=10, height=10, elapsed_s=0.1, output_bytes=b"x",
        )

        response = client.get("/history")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["runs"]), 1)
        self.assertNotIn("output_path", body["runs"][0])

    def test_experiments_endpoint_returns_empty_list_structure(self):
        response = client.get("/experiments")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"experiments": []})

    def test_experiments_endpoint_returns_recorded_experiment(self):
        from colorizer import history

        history.init_db()
        history.record_experiment(
            model="vibrant", dataset="sample", image_count=5, saturation=1.0,
            device="CPU", psnr_mean=27.0, ssim_mean=0.9,
        )

        response = client.get("/experiments")

        self.assertEqual(response.status_code, 200)
        experiments = response.json()["experiments"]
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0]["model"], "vibrant")


if __name__ == "__main__":
    unittest.main()
