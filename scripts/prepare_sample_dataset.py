from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def ensure_structure(output_dir: Path):
    for class_name in ["alert", "drowsy"]:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset folders for ImageFolder training.")
    parser.add_argument("--source-dir", type=Path, required=False, help="Optional source directory to copy from.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    ensure_structure(args.output_dir)

    if args.source_dir and args.source_dir.exists():
        for class_name in ["alert", "drowsy"]:
            source_class = args.source_dir / class_name
            target_class = args.output_dir / class_name
            if source_class.exists():
                for item in source_class.iterdir():
                    if item.is_file():
                        shutil.copy2(item, target_class / item.name)

    print(f"Dataset structure ready at {args.output_dir}")


if __name__ == "__main__":
    main()
