from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report

from src.dataset import load_data_bundle
from src.losses import FocalLoss
from src.model import build_model
from src.train import evaluate
from src.utils import (
    compute_binary_metrics,
    get_device,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained driver drowsiness classifier.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.model_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    threshold = checkpoint.get("threshold", 0.5)

    data_bundle = load_data_bundle(
        data_dir=args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=2,
        val_split=0.2,
        seed=42,
        use_weighted_sampler=False,
    )
    eval_loader = data_bundle.test_loader or data_bundle.val_loader

    device = get_device()
    model = build_model(checkpoint["model_name"], num_classes=len(class_names), freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    class_counts = checkpoint.get("class_counts", {name: 1 for name in class_names})
    class_weights = torch.tensor(
        [
            sum(class_counts.values()) / (len(class_names) * max(class_counts.get(name, 1), 1))
            for name in class_names
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = FocalLoss(gamma=2.0, weight=class_weights, label_smoothing=0.0)
    base_metrics, y_true, y_scores = evaluate(model, eval_loader, criterion, device)
    metrics = compute_binary_metrics(y_true, y_scores, threshold)
    metrics["loss"] = base_metrics["loss"]
    metrics["accuracy_argmax"] = base_metrics["accuracy_argmax"]

    save_json({key: value for key, value in metrics.items() if key != "y_pred"}, Path("outputs") / "evaluation_metrics.json")
    save_json(classification_report(y_true, metrics["y_pred"], zero_division=0, output_dict=True), Path("outputs") / "evaluation_report.json")
    plot_confusion_matrix(y_true, metrics["y_pred"], class_names, Path("outputs") / "evaluation_confusion_matrix.png")
    plot_roc_curve(y_true, y_scores, Path("outputs") / "evaluation_roc_curve.png")
    plot_precision_recall_curve(y_true, y_scores, Path("outputs") / "evaluation_pr_curve.png")

    print(
        f"loss={metrics['loss']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1_score']:.4f} "
        f"auc={metrics['roc_auc']:.4f} "
        f"threshold={threshold:.2f}"
    )


if __name__ == "__main__":
    main()
