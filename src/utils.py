from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def plot_confusion_matrix(y_true: list[int], y_pred: list[int], class_names: list[str], path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrRd", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compute_binary_metrics(y_true: list[int], y_scores: list[float], threshold: float) -> dict:
    y_pred = [1 if score >= threshold else 0 for score in y_scores]
    metrics = {
        "accuracy": float(np.mean(np.array(y_pred) == np.array(y_true))),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_scores),
        "average_precision": average_precision_score(y_true, y_scores),
        "brier_score": brier_score_loss(y_true, y_scores),
        "threshold": threshold,
        "y_pred": y_pred,
    }
    return metrics


def find_optimal_threshold(y_true: list[int], y_scores: list[float], metric: str = "f1") -> tuple[float, dict]:
    candidates = np.linspace(0.2, 0.8, 61)
    best_threshold = 0.5
    best_metrics = compute_binary_metrics(y_true, y_scores, best_threshold)
    best_score = best_metrics["f1_score"] if metric == "f1" else best_metrics["balanced_accuracy"]

    for threshold in candidates:
        metrics = compute_binary_metrics(y_true, y_scores, float(threshold))
        score = metrics["f1_score"] if metric == "f1" else metrics["balanced_accuracy"]
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score
    return best_threshold, best_metrics


def plot_training_curves(history: list[dict], path: Path) -> None:
    epochs = [item["epoch"] for item in history]
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, [item["train_loss"] for item in history], label="Train Loss", color="#9c6644")
    plt.plot(epochs, [item["val_loss"] for item in history], label="Val Loss", color="#d62828")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [item["train_accuracy"] for item in history], label="Train Acc", color="#386641")
    plt.plot(epochs, [item["val_f1_score"] for item in history], label="Val F1", color="#1d3557")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Performance Curves")

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_roc_curve(y_true: list[int], y_scores: list[float], path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}", color="#d62828")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_precision_recall_curve(y_true: list[int], y_scores: list[float], path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}", color="#1d3557")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
