"""CLI: benchmark colorization inference latency and throughput on CPU or GPU.

Not imported by the app itself. Run manually, e.g.:
    python scripts/benchmark.py --model vibrant --images evaluation/datasets/sample --iterations 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorizer import configure_logging
from colorizer.benchmarking import benchmark_images
from colorizer.evaluation import discover_images
from colorizer.models import DEFAULT_MODEL_ID, MODEL_REGISTRY, available_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark colorization inference latency and throughput."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID, choices=list(MODEL_REGISTRY), help="Model id to benchmark."
    )
    parser.add_argument("--images", required=True, help="Folder of images to benchmark against.")
    parser.add_argument("--iterations", type=int, default=3, help="Number of passes over the image set.")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N images found.")
    return parser.parse_args()


def main():
    configure_logging()
    args = parse_args()

    images_dir = Path(args.images)
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    available_ids = {spec.id for spec in available_models()}
    if args.model not in available_ids:
        print(
            f"Model '{args.model}' weights are not available locally. Available: {sorted(available_ids) or 'none'}",
            file=sys.stderr,
        )
        sys.exit(1)

    image_paths = discover_images(images_dir)
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        print(f"No supported images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        result = benchmark_images(image_paths, model_id=args.model, iterations=args.iterations)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print("AI Image Colorizer Benchmark")
    print("=============================")
    print()
    print(f"Model: {result.model}")
    print(f"Device: {result.device}")
    print(f"Images: {result.image_count}")
    print(f"Iterations: {result.iterations}")
    print()
    print(f"Model load time: {result.model_load_seconds:.3f} sec")
    print()
    print("Inference latency")
    print(f"  Mean:   {result.mean_inference_seconds:.3f} sec")
    print(f"  Median: {result.median_inference_seconds:.3f} sec")
    print(f"  P95:    {result.p95_inference_seconds:.3f} sec")
    print()
    print(f"Throughput: {result.images_per_second:.2f} images/sec")
    print()
    print("Completed successfully.")


if __name__ == "__main__":
    main()
