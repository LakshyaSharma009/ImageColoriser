import tempfile
import unittest
from pathlib import Path

from colorizer import models as core_models


class ModelsTests(unittest.TestCase):
    def test_require_file_returns_string_for_existing_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.bin"
            path.write_text("ok", encoding="utf-8")

            self.assertEqual(core_models._require_file(path), str(path))

    def test_require_file_raises_for_missing_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.bin"

            with self.assertRaises(FileNotFoundError):
                core_models._require_file(path)

    def test_load_model_rejects_unknown_id(self):
        with self.assertRaises(ValueError):
            core_models.load_model("not-a-real-model")

    def test_available_models_only_returns_specs_with_weights_on_disk(self):
        available = core_models.available_models()
        for spec in available:
            self.assertTrue(spec.weights_path.exists())
        available_ids = {spec.id for spec in available}
        self.assertIn("vibrant", available_ids)


if __name__ == "__main__":
    unittest.main()
