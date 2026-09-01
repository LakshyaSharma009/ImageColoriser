import logging
from logging.handlers import RotatingFileHandler

from .config import LOG_DIR, LOG_FILE, LOG_LEVEL

_configured = False


def configure_logging():
    """Idempotent: safe to call on every Streamlit rerun without duplicating handlers."""
    global _configured
    if _configured:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    _configured = True
