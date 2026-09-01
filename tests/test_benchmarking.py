import statistics
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from colorizer import benchmarking


def _write_image(path, size=(10, 10)):
    img = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


class PercentileTests(unittest.TestCase):
    def test_percentile_matches_median_at_50th(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        self.assertAlmostEqual(benchmarking._percentile(values, 50), statistics.median(values), delta=1.0)

    def test_percentile_of_empty_sequence_is_zero(self):
        self.assertEqual(benchmarking._percentile([], 95), 0.0)


class MeasureModelLoadTimeTests(unittest.TestCase):
    @patch("colorizer.benchmarking.load_model")
    def test_measure_model_load_time_clears_cache_and_times_the_call(self, mock_load_model):
        elapsed = benchmarking.measure_model_load_time("vibrant")

        mock_load_model.cache_clear.assert_called_once()
        mock_load_model.assert_called_once_with("vibrant")
        self.assertGreaterEqual(elapsed, 0)


class BenchmarkImagesTests(unittest.TestCase):
    @patch("colorizer.benchmarking.measure_model_load_time")
    @patch("colorizer.benchmarking.colorize_image")
    def test_benchmark_images_reports_expected_fields(self, mock_colorize, mock_load_time):
        mock_load_time.return_value = 0.05
        mock_colorize.side_effect = lambda img, **kwargs: img

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for i in range(3):
                p = root / f"img{i}.jpg"
                _write_image(p)
                paths.append(p)

            result = benchmarking.benchmark_images(paths, model_id="vibrant", iterations=2)

            self.assertEqual(result.model, "vibrant")
            self.assertEqual(result.image_count, 3)
            self.assertEqual(result.iterations, 2)
            self.assertEqual(result.model_load_seconds, 0.05)
            self.assertGreaterEqual(result.mean_inference_seconds, 0)
            self.assertGreaterEqual(result.median_inference_seconds, 0)
            self.assertGreaterEqual(result.p95_inference_seconds, 0)
            self.assertGreater(result.images_per_second, 0)
            # 1 warmup call + 2 iterations * 3 images = 7 calls
            self.assertEqual(mock_colorize.call_count, 7)
            self.assertEqual(len(result.per_image_seconds), 6)

    @patch("colorizer.benchmarking.measure_model_load_time")
    @patch("colorizer.benchmarking.colorize_image")
    def test_benchmark_images_without_warmup_only_times_iterations(self, mock_colorize, mock_load_time):
        mock_load_time.return_value = 0.0
        mock_colorize.side_effect = lambda img, **kwargs: img

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "img.jpg"
            _write_image(p)

            benchmarking.benchmark_images([p], model_id="vibrant", iterations=3, warmup=False)

            self.assertEqual(mock_colorize.call_count, 3)

    def test_benchmark_images_raises_for_no_decodable_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken = root / "broken.jpg"
            broken.write_bytes(b"not an image")

            with self.assertRaises(ValueError):
                benchmarking.benchmark_images([broken], model_id="vibrant")


if __name__ == "__main__":
    unittest.main()
