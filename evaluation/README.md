# Evaluation data

This folder is a workspace for quantitative evaluation. Nothing under it is
committed to the repository except this file and empty placeholder
directories -- datasets and results are generated locally, on demand.

```text
evaluation/
├── datasets/   Your own ground-truth color images (not committed)
├── results/    CSV/JSON output from scripts/evaluate.py (not committed)
├── plots/      Reserved for any plots you export (not committed)
└── examples/   Reserved for saved example visualizations (not committed)
```

## Providing a dataset

Create a folder under `evaluation/datasets/` and drop in your own color
photos (the evaluator grayscales them itself -- do not pre-convert them):

```text
evaluation/datasets/<dataset_name>/
    image1.jpg
    image2.png
    subfolder/image3.webp
    ...
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`. Images are discovered
recursively, so subfolders are fine. A few dozen varied photos are enough to
get a meaningful read on quality; there's no required minimum.

## Running an evaluation

```bash
python scripts/evaluate.py --dataset evaluation/datasets/<dataset_name> --model vibrant
```

This writes `evaluation/results/vibrant.csv` (per-image PSNR/SSIM/LPIPS/timing)
and `evaluation/results/vibrant.json` (aggregate summary), and records a row
in the SQLite experiment history (also visible in the Streamlit app's
Evaluation tab). See the main [README](../README.md#evaluation) for details
on the metrics and the `--output`, `--saturation`, and `--limit` flags.
