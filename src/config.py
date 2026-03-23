from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass
class TrainConfig:
    data_dir: Path = DATA_DIR
    model_dir: Path = MODEL_DIR
    output_dir: Path = OUTPUT_DIR
    model_name: str = "efficientnet_b0"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    freeze_backbone: bool = True
    fine_tune_epoch: int = 4
    val_split: float = 0.2
    seed: int = 42
    patience: int = 4
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    threshold_metric: str = "f1"
    use_weighted_sampler: bool = True
