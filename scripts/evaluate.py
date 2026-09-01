"""CLI: evaluate colorization quality (PSNR/SSIM/LPIPS) against a dataset of ground-truth color images.

Not imported by the app itself. Run manually, e.g.:
    python scripts/evaluate.py --dataset evaluation/datasets/sample --model vibrant
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorizer import configure_logging, history
from colorizer.config import EVALUATION_RESULTS_DIR
from colorizer.evaluation import evaluate_dataset, lpips_available, write_csv, write_json
from colorizer.models import DEFAULT_MODEL_ID, MODEL_REGISTRY, available_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate colorization quality (PSNR/SSIM/LPIPS) against a dataset of color images."
    )
    parser.add_argument("--dataset", required=True, help="Path to a folder of ground-truth color images.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID, choices=list(MODEL_REGISTRY), help="Model id to evaluate."
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write per-image CSV results (a .json summary is written alongside it).",
    )
    parser.add_argument("--saturation", type=float, default=1.0, help="AB channel saturation multiplier.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N images found.")
    return parser.parse_args()


def main():
    configure_logging()
    args = parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    available_ids = {spec.id for spec in available_models()}
    if args.model not in available_ids:
        print(
            f"Model '{args.model}' weights are not available locally. Available: {sorted(available_ids) or 'none'}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        result = evaluate_dataset(
            dataset_dir, model_id=args.model, saturation=args.saturation, limit=args.limit
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else EVALUATION_RESULTS_DIR / f"{args.model}.csv"
    write_csv(result.per_image, output_path)
    json_path = output_path.with_suffix(".json")
    write_json(result, json_path)

    try:
        history.init_db()
        history.record_experiment(
            model=result.model,
            dataset=result.dataset,
            image_count=result.image_count,
            saturation=args.saturation,
            device=result.device,
            psnr_mean=result.psnr_mean,
            ssim_mean=result.ssim_mean,
            lpips_mean=result.lpips_mean,
            latency_mean_seconds=result.latency_mean_seconds,
            latency_median_seconds=result.latency_median_seconds,
            latency_p95_seconds=result.latency_p95_seconds,
        )
    except Exception as exc:  # experiment logging should never block reporting results
        print(f"Warning: could not record experiment history ({exc})", file=sys.stderr)

    print("AI Image Colorizer Evaluation")
    print("=============================")
    print()
    print(f"Model: {result.model}")
    print(f"Dataset: {result.dataset}")
    print(f"Images: {result.image_count}")
    print()
    print("PSNR")
    print(f"  Mean:   {result.psnr_mean:.2f} dB")
    print(f"  Median: {result.psnr_median:.2f} dB")
    print()
    print("SSIM")
    print(f"  Mean:   {result.ssim_mean:.4f}")
    print(f"  Median: {result.ssim_median:.4f}")
    print()
    if not lpips_available():
        print("LPIPS: unavailable (optional 'lpips'/'torch' packages not installed)")
    elif result.lpips_mean is not None:
        print("LPIPS")
        print(f"  Mean:   {result.lpips_mean:.4f}")
    print()
    print("Latency")
    print(f"  Mean:   {result.latency_mean_seconds:.3f} sec")
    print(f"  Median: {result.latency_median_seconds:.3f} sec")
    print(f"  P95:    {result.latency_p95_seconds:.3f} sec")
    print()
    print(f"Per-image CSV: {output_path}")
    print(f"Summary JSON:  {json_path}")
    print()
    print("Completed successfully.")


if __name__ == "__main__":
    main()
