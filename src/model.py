from __future__ import annotations

import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = [
    "resnet18",
    "resnet50",
    "mobilenet_v3_small",
    "efficientnet_b0",
]


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def build_model(model_name: str, num_classes: int, freeze_backbone: bool = True, dropout: float = 0.3) -> nn.Module:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        backbone = model
        classifier = model.fc
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        backbone = model
        classifier = model.fc
    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, model.classifier[0].out_features),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        backbone = model.features
        classifier = model.classifier
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        backbone = model.features
        classifier = model.classifier
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    if freeze_backbone:
        _set_requires_grad(backbone, False)
        _set_requires_grad(classifier, True)

    return model


def unfreeze_model(model: nn.Module) -> None:
    _set_requires_grad(model, True)


def get_gradcam_target_layer(model_name: str, model: nn.Module) -> nn.Module:
    if model_name in {"resnet18", "resnet50"}:
        return model.layer4[-1]
    if model_name == "mobilenet_v3_small":
        return model.features[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Unsupported model_name: {model_name}")
