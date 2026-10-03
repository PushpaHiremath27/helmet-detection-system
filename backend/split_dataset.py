"""
Splits the flat data/helmet/images + labels dataset (produced by
dataset_loader.py) into train/val/test sets in YOLO layout, and rewrites
data.yaml with the final paths and class names.

Usage:
    python split_dataset.py --train 0.8 --val 0.1 --test 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "helmet"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
SPLITS = ("train", "val", "test")


def collect_flat_pairs() -> list[tuple[Path, Path]]:
    """Image/label pairs sitting directly in images/ and labels/ (not yet split)."""
    pairs = []
    for img_path in IMAGES_DIR.iterdir():
        if not img_path.is_file():
            continue
        label_path = LABELS_DIR / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            print(f"[split][warn] No label for image, skipping: {img_path.name}")
    return pairs


def primary_class(label_path: Path) -> int:
    """Used for rough stratification: the most frequent class in this label file."""
    counts: dict[int, int] = defaultdict(int)
    with open(label_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if parts:
                counts[int(parts[0])] += 1
    if not counts:
        return -1
    return max(counts, key=counts.get)


def stratified_split(pairs, train_ratio, val_ratio, test_ratio, seed):
    rng = random.Random(seed)
    by_class: dict[int, list] = defaultdict(list)
    for pair in pairs:
        by_class[primary_class(pair[1])].append(pair)

    splits = {"train": [], "val": [], "test": []}
    for cls, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train:n_train + n_val])
        splits["test"].extend(items[n_train + n_val:])

    for s in splits.values():
        rng.shuffle(s)
    return splits


def clear_split_dirs():
    for split in SPLITS:
        for base in (IMAGES_DIR, LABELS_DIR):
            d = base / split
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()


def write_split(splits: dict[str, list]) -> dict[str, int]:
    counts = {}
    for split, pairs in splits.items():
        for img_path, label_path in pairs:
            shutil.move(str(img_path), IMAGES_DIR / split / img_path.name)
            shutil.move(str(label_path), LABELS_DIR / split / label_path.name)
        counts[split] = len(pairs)
    return counts


def update_data_yaml():
    yaml_path = DATA_DIR / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    else:
        config = {}
    config.update({
        "path": str(DATA_DIR).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": config.get("nc", 2),
        "names": config.get("names", ["helmet", "no_helmet"]),
    })
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Split the helmet dataset into train/val/test")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratio_sum = args.train + args.val + args.test
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    pairs = collect_flat_pairs()
    if not pairs:
        print("[split][error] No flat image/label pairs found in "
              f"{IMAGES_DIR} / {LABELS_DIR}. Run dataset_loader.py first, "
              "or the dataset has already been split.")
        return

    print(f"[split] Found {len(pairs)} image/label pairs to split.")
    splits = stratified_split(pairs, args.train, args.val, args.test, args.seed)
    clear_split_dirs()
    counts = write_split(splits)
    update_data_yaml()

    print("\n===== Split Summary =====")
    for split in SPLITS:
        print(f"{split:<6}: {counts[split]} images")
    print(f"\nUpdated {DATA_DIR / 'data.yaml'}")
    print("Next step: python train_helmet.py --data data/helmet/data.yaml")


if __name__ == "__main__":
    main()
