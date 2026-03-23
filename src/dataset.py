from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import datasets, transforms


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader | None
    class_names: list[str]
    class_counts: dict[str, int]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 24, image_size + 24)),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
            transforms.RandomAutocontrast(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _extract_targets(dataset) -> list[int]:
    if isinstance(dataset, datasets.ImageFolder):
        return list(dataset.targets)
    if isinstance(dataset, TransformedSubset):
        subset = dataset.subset
        return [subset.dataset.targets[i] for i in subset.indices]
    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def _compute_class_counts(targets: list[int], class_names: list[str]) -> dict[str, int]:
    counts = {class_name: 0 for class_name in class_names}
    for target in targets:
        counts[class_names[target]] += 1
    return counts


def _make_sampler(targets: list[int]) -> WeightedRandomSampler:
    class_counts = np.bincount(targets)
    sample_weights = [1.0 / class_counts[target] for target in targets]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def create_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
    use_weighted_sampler: bool,
) -> DataBundle:
    seed_everything(seed)
    train_transform, eval_transform = build_transforms(image_size)
    base_dataset = datasets.ImageFolder(data_dir, transform=None)

    val_size = int(len(base_dataset) * val_split)
    train_size = len(base_dataset) - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError("Dataset is too small for the requested validation split.")

    train_subset, val_subset = random_split(
        base_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_dataset = TransformedSubset(train_subset, train_transform)
    val_dataset = TransformedSubset(val_subset, eval_transform)
    train_targets = _extract_targets(train_dataset)
    sampler = _make_sampler(train_targets) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        class_names=base_dataset.classes,
        class_counts=_compute_class_counts(train_targets, base_dataset.classes),
    )


def create_split_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    use_weighted_sampler: bool,
) -> DataBundle:
    train_transform, eval_transform = build_transforms(image_size)

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=eval_transform)
    train_targets = _extract_targets(train_dataset)
    sampler = _make_sampler(train_targets) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=train_dataset.classes,
        class_counts=_compute_class_counts(train_targets, train_dataset.classes),
    )


def detect_dataset_layout(data_dir: Path) -> str:
    if all((data_dir / split).exists() for split in ["train", "val", "test"]):
        return "split"
    return "flat"


def load_data_bundle(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
    use_weighted_sampler: bool,
) -> DataBundle:
    dataset_layout = detect_dataset_layout(data_dir)
    if dataset_layout == "split":
        return create_split_dataloaders(
            data_dir=data_dir,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            use_weighted_sampler=use_weighted_sampler,
        )
    return create_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        seed=seed,
        use_weighted_sampler=use_weighted_sampler,
    )
