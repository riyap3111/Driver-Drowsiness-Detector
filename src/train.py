from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

from src.config import TrainConfig
from src.dataset import detect_dataset_layout, load_data_bundle
from src.losses import FocalLoss
from src.model import SUPPORTED_MODELS, build_model, unfreeze_model
from src.utils import (
    compute_binary_metrics,
    find_optimal_threshold,
    get_device,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_training_curves,
    save_json,
)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_scores = []

    for images, labels in tqdm(loader, desc="Evaluation", leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_scores.extend(probabilities.cpu().tolist())

    metrics = {
        "loss": running_loss / total,
        "accuracy_argmax": correct / total,
    }
    return metrics, y_true, y_scores


def parse_args():
    parser = argparse.ArgumentParser(description="Train a graduate-level driver drowsiness detector.")
    parser.add_argument("--data-dir", type=Path, default=TrainConfig.data_dir)
    parser.add_argument("--model-name", type=str, default=TrainConfig.model_name, choices=SUPPORTED_MODELS)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--fine-tune-epoch", type=int, default=TrainConfig.fine_tune_epoch)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-freeze-backbone", action="store_true")
    parser.add_argument("--disable-weighted-sampler", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(
        data_dir=args.data_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        fine_tune_epoch=args.fine_tune_epoch,
        freeze_backbone=False if args.no_freeze_backbone else True,
        use_weighted_sampler=not args.disable_weighted_sampler,
    )
    if args.freeze_backbone:
        config.freeze_backbone = True

    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    data_bundle = load_data_bundle(
        data_dir=config.data_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
        seed=config.seed,
        use_weighted_sampler=config.use_weighted_sampler,
    )

    device = get_device()
    model = build_model(
        config.model_name,
        num_classes=len(data_bundle.class_names),
        freeze_backbone=config.freeze_backbone,
    ).to(device)

    class_weights = torch.tensor(
        [
            sum(data_bundle.class_counts.values()) / (len(data_bundle.class_names) * max(count, 1))
            for count in data_bundle.class_counts.values()
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = FocalLoss(
        gamma=config.focal_gamma,
        weight=class_weights,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_f1 = -1.0
    best_threshold = 0.5
    history = []
    epochs_without_improvement = 0
    backbone_unfrozen = not config.freeze_backbone
    dataset_layout = detect_dataset_layout(config.data_dir)
    best_model_path = config.model_dir / f"{config.model_name}_best.pt"

    for epoch in range(1, config.epochs + 1):
        if (
            config.freeze_backbone
            and not backbone_unfrozen
            and epoch >= config.fine_tune_epoch
        ):
            unfreeze_model(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate / 10, weight_decay=config.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs - epoch + 1, 1))
            backbone_unfrozen = True

        train_loss, train_acc = train_one_epoch(model, data_bundle.train_loader, criterion, optimizer, device, scaler)
        val_base_metrics, y_true, y_scores = evaluate(model, data_bundle.val_loader, criterion, device)
        threshold, val_metrics = find_optimal_threshold(y_true, y_scores, metric=config.threshold_metric)
        val_metrics["loss"] = val_base_metrics["loss"]
        val_metrics["accuracy_argmax"] = val_base_metrics["accuracy_argmax"]

        epoch_metrics = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "threshold": threshold,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_score": val_metrics["f1_score"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_average_precision": val_metrics["average_precision"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1_score']:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} threshold={threshold:.2f}"
        )

        scheduler.step()

        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            best_threshold = threshold
            epochs_without_improvement = 0
            report = classification_report(y_true, val_metrics["y_pred"], zero_division=0, output_dict=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": data_bundle.class_names,
                    "class_counts": data_bundle.class_counts,
                    "model_name": config.model_name,
                    "image_size": config.image_size,
                    "dataset_layout": dataset_layout,
                    "threshold": best_threshold,
                    "val_metrics": {key: value for key, value in val_metrics.items() if key != "y_pred"},
                },
                best_model_path,
            )
            save_json(report, config.output_dir / "classification_report.json")
            save_json({key: value for key, value in val_metrics.items() if key != "y_pred"}, config.output_dir / "best_val_metrics.json")
            plot_confusion_matrix(y_true, val_metrics["y_pred"], data_bundle.class_names, config.output_dir / "confusion_matrix.png")
            plot_roc_curve(y_true, y_scores, config.output_dir / "roc_curve.png")
            plot_precision_recall_curve(y_true, y_scores, config.output_dir / "precision_recall_curve.png")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    save_json({"history": history}, config.output_dir / "training_history.json")
    plot_training_curves(history, config.output_dir / "training_curves.png")
    print(f"Best model saved to {best_model_path} with threshold={best_threshold:.2f}")


if __name__ == "__main__":
    main()
