import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        # Uses fake specs/weights rather than the real Model/ directory, so this
        # passes the same way whether or not the actual (123MB, not committed
        # to git) caffemodel happens to be present on the machine running it.
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            present = core_models.ModelSpec(
                id="present", label="Present", description="", weights_filename="present.caffemodel"
            )
            missing = core_models.ModelSpec(
                id="missing", label="Missing", description="", weights_filename="missing.caffemodel"
            )
            (model_dir / present.weights_filename).write_bytes(b"fake weights")

            with patch.object(core_models, "MODEL_DIR", model_dir), \
                 patch.object(core_models, "MODEL_REGISTRY", {"present": present, "missing": missing}):
                available = core_models.available_models()

                for spec in available:
                    self.assertTrue(spec.weights_path.exists())
                self.assertEqual({spec.id for spec in available}, {"present"})


if __name__ == "__main__":
    unittest.main()
