# Driver Vigilance Intelligence System

This project is an upgraded master’s-level driver drowsiness detection system using Python, PyTorch, OpenCV, NumPy, Pandas, transfer learning, calibrated decision thresholds, explainability, and a Streamlit dashboard for both images and videos.

## Features

- Advanced backbones: `EfficientNet-B0`, `ResNet18`, `ResNet50`, `MobileNetV3`
- Weighted sampling, focal loss, label smoothing, AdamW, cosine scheduling, and early stopping
- Threshold optimization using validation F1 or balanced accuracy
- ROC, PR, confusion matrix, and calibration-oriented metrics
- OpenCV face and eye analysis
- Grad-CAM visual explanations for image inference
- Video processing with frame sampling, temporal smoothing, and annotated output
- Streamlit dashboard for both image and video analysis
- Live webcam streaming with real-time score updates inside Streamlit

## Project Structure

```text
Credit/
├── data/
│   ├── raw/
│   └── processed/
│       ├── alert/
│       └── drowsy/
├── models/
├── outputs/
├── scripts/
├── src/
├── streamlit_app.py
└── requirements.txt
```

## 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Prepare your dataset

Put your images in this format:

```text
data/processed/
├── alert/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── drowsy/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

If you already have labeled folders, you can initialize the structure with:

```bash
python scripts/prepare_sample_dataset.py --output-dir data/processed
```

## 3. Train the model

```bash
python -m src.train --data-dir data --model-name efficientnet_b0 --epochs 12 --batch-size 32
```

For ResNet50:

```bash
python -m src.train --data-dir data --model-name resnet50 --epochs 12
```

The best checkpoint is saved in `models/` and reports are saved in `outputs/`.

You can also train directly from an already split dataset like:

```text
data/
├── train/
│   ├── awake/
│   └── sleepy/
├── val/
│   ├── awake/
│   └── sleepy/
└── test/
    ├── awake/
    └── sleepy/
```

If your downloaded dataset already looks like that, point `--data-dir` to that folder. Your current project already uses this layout.

## 4. Evaluate the model

```bash
python -m src.evaluate --model-path models/efficientnet_b0_best.pt --data-dir data
```

## 5. Run inference on one image

```bash
python -m src.inference --model-path models/efficientnet_b0_best.pt --image-path path/to/test.jpg
```

## 6. Run video analysis

```bash
python -m src.inference --model-path models/efficientnet_b0_best.pt --video-path path/to/drive.mp4 --output-video-path outputs/annotated_drive.mp4
```

## 7. Run webcam detection

```bash
python -m src.inference --model-path models/efficientnet_b0_best.pt --webcam
```

## 8. Launch the Streamlit app

```bash
streamlit run streamlit_app.py
```

For real-time webcam scoring inside Streamlit, use the `Live Monitor` mode in the sidebar and click `Start`.
The dashboard now enforces a 75% operating threshold, smooths live scores, tracks sustained high-risk sequences, and surfaces lighting and blur quality warnings.

## Suggested datasets

Look for driver alertness or eye-state datasets with classes like open/closed eyes, alert/drowsy, or yawning/no-yawning. If your source dataset uses different folder names, rename them to `alert` and `drowsy` before training.

## Graduate-level extensions included

- Stronger transfer-learning backbones
- Imbalance-aware optimization with weighted sampling and focal loss
- Threshold tuning for safety-oriented operation
- Explainability through Grad-CAM
- Video-level temporal smoothing and risk timeline analysis
- Evaluation artifacts suitable for a thesis demo or project defense

## Resume-ready summary

You can describe this project like this:

> Developed a graduate-level driver vigilance intelligence system using transfer learning with EfficientNet/ResNet backbones, imbalance-aware optimization, threshold calibration, Grad-CAM explainability, and video-level temporal risk analysis. Evaluated performance with ROC-AUC, precision-recall, F1-score, balanced accuracy, and confusion matrices for safety-critical monitoring.
