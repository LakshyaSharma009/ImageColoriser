import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = Path(os.environ.get("IMAGECOLORIZER_MODEL_DIR", BASE_DIR / "Model"))

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MAX_IMAGE_PIXELS = int(os.environ.get("IMAGECOLORIZER_MAX_IMAGE_PIXELS", 25_000_000))
MAX_UPLOAD_BYTES = int(os.environ.get("IMAGECOLORIZER_MAX_UPLOAD_BYTES", 20 * 1024 * 1024))

HISTORY_DIR = Path(os.environ.get("IMAGECOLORIZER_HISTORY_DIR", BASE_DIR / "history"))
HISTORY_DB_PATH = HISTORY_DIR / "history.db"
HISTORY_OUTPUT_DIR = HISTORY_DIR / "outputs"
HISTORY_MAX_ENTRIES = int(os.environ.get("IMAGECOLORIZER_HISTORY_MAX_ENTRIES", 200))

LOG_DIR = Path(os.environ.get("IMAGECOLORIZER_LOG_DIR", BASE_DIR / "logs"))
LOG_FILE = LOG_DIR / "app.log"
LOG_LEVEL = os.environ.get("IMAGECOLORIZER_LOG_LEVEL", "INFO")

EVALUATION_DIR = Path(os.environ.get("IMAGECOLORIZER_EVALUATION_DIR", BASE_DIR / "evaluation"))
EVALUATION_DATASETS_DIR = EVALUATION_DIR / "datasets"
EVALUATION_RESULTS_DIR = EVALUATION_DIR / "results"
