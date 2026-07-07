# AI Image Colorizer

AI Image Colorizer is a small application that adds color to grayscale photos using a pretrained deep learning model from Zhang, Isola, and Efros (ECCV 2016). It includes both a desktop Tkinter app and a Streamlit web app.

## Project Overview

This project solves the classic black-and-white photo colorization problem. You can open a single image, batch-process a set of images, preview the result, and download the output without needing to train a model yourself.

## Model Used

- Paper: *Colorful Image Colorization* (ECCV 2016)
- Authors: Richard Zhang, Phillip Isola, Alexei A. Efros
- Framework: Caffe model loaded through OpenCV DNN
- Input space: LAB color space, where the network predicts the missing `a` and `b` color channels from the grayscale `L` channel

## Pipeline

```text
Image -> Normalize -> LAB -> Extract L -> CNN -> Predict AB -> Merge LAB -> BGR -> Output
```

## Features

- Single image colorization
- Batch colorization
- Streamlit UI
- Download outputs as PNG or ZIP
- Before/after comparison slider in the Streamlit view
- Batch progress bar and processing statistics
- Graceful error handling for empty uploads, unsupported files, corrupted images, oversized images, and missing model files

## Installation

Install the Python dependencies:

```bash
pip install opencv-python numpy Pillow streamlit
```

Download the model files into a `Model/` folder in the project root:

- `colorization_deploy_v2.prototxt`
- `colorization_release_v2.caffemodel`
- `pts_in_hull.npy`

## Usage

Run the desktop app:

```bash
python main.py
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

In the Streamlit app:

- Single image mode shows the original image, the colorized output, and the comparison slider.
- Batch mode processes multiple images, shows progress, reports statistics, and lets you download a ZIP of the results.

## Screenshots

Add screenshots of these views here:

- Before
- After
- Streamlit UI
- Batch mode

## Project Structure

```text
ImageColoriser/
├── colorizer_core.py    Shared model loading and colorization logic
├── main.py              Tkinter GUI entry point
├── streamlit_app.py     Streamlit UI
├── Model/               Pretrained model files
└── README.md
```

## Credits

Model architecture and pretrained weights from ["Colorful Image Colorization"](https://richzhang.github.io/colorization/) by Zhang, Isola, and Efros.
