import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from colorizer import evaluation


def _write_image(path, size=(16, 16)):
    img = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return img


class MetricCalculationTests(unittest.TestCase):
    def test_psnr_is_infinite_for_identical_images(self):
        img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        self.assertTrue(math.isinf(evaluation.calculate_psnr(img, img)))

    def test_psnr_is_higher_for_more_similar_images(self):
        base = np.zeros((16, 16, 3), dtype=np.uint8)
        similar = base.copy()
        similar[0, 0] = 10
        different = np.full((16, 16, 3), 255, dtype=np.uint8)
        self.assertGreater(evaluation.calculate_psnr(base, similar), evaluation.calculate_psnr(base, different))

    def test_ssim_is_one_for_identical_images(self):
        img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        self.assertAlmostEqual(evaluation.calculate_ssim(img, img), 1.0, places=5)

    def test_ssim_is_lower_for_different_images(self):
        base = np.zeros((16, 16, 3), dtype=np.uint8)
        different = np.full((16, 16, 3), 255, dtype=np.uint8)
        self.assertLess(evaluation.calculate_ssim(base, different), 1.0)

    def test_lpips_returns_none_when_optional_package_missing(self):
        with patch.object(evaluation, "_lpips_pkg", None):
            self.assertFalse(evaluation.lpips_available())
            result = evaluation.calculate_lpips(
                np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)
            )
            self.assertIsNone(result)


class DatasetDiscoveryTests(unittest.TestCase):
    def test_discover_images_finds_supported_extensions_recursively(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            _write_image(root / "a.jpg")
            _write_image(root / "sub" / "b.png")
            (root / "notes.txt").write_text("not an image", encoding="utf-8")

            found = evaluation.discover_images(root)

            self.assertEqual({p.name for p in found}, {"a.jpg", "b.png"})

    def test_discover_images_raises_for_missing_directory(self):
        with self.assertRaises(FileNotFoundError):
            evaluation.discover_images(Path("this/does/not/exist"))


class EvaluateImageTests(unittest.TestCase):
    @patch("colorizer.evaluation.colorize_image")
    def test_evaluate_image_returns_metrics_for_a_perfect_prediction(self, mock_colorize):
        with tempfile.TemporaryDirectory() as tmp:
            # PNG is lossless, so re-reading it from disk reproduces the exact
            # array below -- unlike JPEG, which would make a "perfect" prediction
            # merely close rather than bit-identical, and PSNR non-infinite.
            path = Path(tmp) / "photo.png"
            _write_image(path, size=(20, 12))
            original = cv2.imread(str(path), cv2.IMREAD_COLOR)
            mock_colorize.return_value = original

            metrics = evaluation.evaluate_image(path, model_id="vibrant", saturation=1.0)

            self.assertEqual(metrics.filename, "photo.png")
            self.assertEqual(metrics.width, 20)
            self.assertEqual(metrics.height, 12)
            self.assertEqual(metrics.model, "vibrant")
            self.assertTrue(math.isinf(metrics.psnr))
            self.assertAlmostEqual(metrics.ssim, 1.0, places=4)
            self.assertGreaterEqual(metrics.inference_time_seconds, 0)
            mock_colorize.assert_called_once()

    def test_evaluate_image_raises_for_undecodable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jpg"
            path.write_bytes(b"not an image")
            with self.assertRaises(ValueError):
                evaluation.evaluate_image(path)


class EvaluateDatasetTests(unittest.TestCase):
    @patch("colorizer.evaluation.load_model")
    @patch("colorizer.evaluation.colorize_image")
    def test_evaluate_dataset_aggregates_metrics(self, mock_colorize, mock_load_model):
        mock_colorize.side_effect = lambda img, **kwargs: img
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(3):
                _write_image(root / f"img{i}.jpg", size=(10, 10))

            result = evaluation.evaluate_dataset(root, model_id="vibrant")

            self.assertEqual(result.image_count, 3)
            self.assertEqual(len(result.per_image), 3)
            self.assertEqual(result.model, "vibrant")
            self.assertIn("psnr_mean", result.metrics)
            self.assertIn("mean_seconds", result.latency)
            self.assertIsNone(result.lpips_mean)
            mock_load_model.assert_called_once_with("vibrant")

    def test_evaluate_dataset_raises_for_empty_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                evaluation.evaluate_dataset(tmp)

    @patch("colorizer.evaluation.load_model")
    @patch("colorizer.evaluation.colorize_image")
    def test_evaluate_dataset_respects_limit(self, mock_colorize, mock_load_model):
        mock_colorize.side_effect = lambda img, **kwargs: img
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(5):
                _write_image(root / f"img{i}.jpg", size=(8, 8))

            result = evaluation.evaluate_dataset(root, limit=2)

            self.assertEqual(result.image_count, 2)


class CsvJsonOutputTests(unittest.TestCase):
    @patch("colorizer.evaluation.load_model")
    @patch("colorizer.evaluation.colorize_image")
    def test_write_csv_and_json_produce_expected_files(self, mock_colorize, mock_load_model):
        mock_colorize.side_effect = lambda img, **kwargs: img
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            root.mkdir()
            _write_image(root / "a.jpg", size=(8, 8))
            result = evaluation.evaluate_dataset(root, model_id="vibrant")

            csv_path = Path(tmp) / "out.csv"
            json_path = Path(tmp) / "out.json"
            evaluation.write_csv(result.per_image, csv_path)
            evaluation.write_json(result, json_path)

            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("filename", csv_text.splitlines()[0])

            json_text = json_path.read_text(encoding="utf-8")
            self.assertIn('"psnr_mean"', json_text)
            self.assertIn('"mean_seconds"', json_text)


if __name__ == "__main__":
    unittest.main()
