import unittest
from unittest.mock import Mock, patch

import numpy as np

from colorizer.pipeline import colorize_image


def _dummy_net():
    net = Mock()
    net.getLayerId.side_effect = lambda name: 1
    layer = Mock()
    layer.blobs = []
    net.getLayer.return_value = layer
    net.forward.return_value = np.zeros((1, 2, 56, 56), dtype=np.float32)
    return net


class PipelineTests(unittest.TestCase):
    def test_colorize_image_rejects_non_bgr_input(self):
        with self.assertRaises(ValueError):
            colorize_image(np.zeros((10, 10), dtype=np.uint8))

    @patch("colorizer.pipeline.models.load_model")
    def test_colorize_image_returns_bgr_image(self, load_model):
        load_model.return_value = _dummy_net()

        image = np.zeros((32, 48, 3), dtype=np.uint8)
        result = colorize_image(image)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        load_model.return_value.setInput.assert_called_once()

    @patch("colorizer.pipeline.models.load_model")
    def test_colorize_image_passes_model_id_through(self, load_model):
        load_model.return_value = _dummy_net()

        image = np.zeros((16, 16, 3), dtype=np.uint8)
        colorize_image(image, model_id="natural")

        load_model.assert_called_once_with("natural")

    @patch("colorizer.pipeline.models.load_model")
    def test_saturation_does_not_change_output_shape_or_dtype(self, load_model):
        load_model.return_value = _dummy_net()

        image = np.zeros((16, 16, 3), dtype=np.uint8)
        result = colorize_image(image, saturation=1.4)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
