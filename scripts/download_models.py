"""Standalone provisioning tool: fetches optional colorization model weights into Model/.

Not imported by the app itself. Run manually:
    python scripts/download_models.py [vibrant|natural]
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorizer.config import MODEL_DIR
from colorizer.models import MODEL_REGISTRY

BASE_URL = "http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2"


def download(model_id, model_dir):
    spec = MODEL_REGISTRY[model_id]
    dest = model_dir / spec.weights_filename

    if dest.exists():
        print(f"{spec.weights_filename} already present at {dest}, skipping.")
        return

    url = f"{BASE_URL}/{spec.weights_filename}"
    print(f"Downloading {spec.label} weights:\n  {url}\n  -> {dest}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, open(dest, "wb") as out_file:
            out_file.write(response.read())
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "You can fetch this file manually from the richzhang/colorization GitHub repo "
            "(see models/fetch_release_models.sh) and place it in the Model/ folder as "
            f"'{spec.weights_filename}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Saved {dest} ({dest.stat().st_size / 1_000_000:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download optional colorization model weights.")
    parser.add_argument(
        "model",
        choices=list(MODEL_REGISTRY),
        nargs="?",
        default="natural",
        help="Which model weights to download (default: natural, the optional one).",
    )
    parser.add_argument("--model-dir", default=None, help="Override the Model/ directory.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    download(args.model, model_dir)


if __name__ == "__main__":
    main()
