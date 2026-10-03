"""
Universal dataset auto-loader for the Helmet Detection project.

Ingests a dataset from any of six source types, auto-detects the label
format (YOLO / Pascal VOC / COCO / CSV), converts everything to YOLO txt
format, remaps class names onto the target schema (helmet, no_helmet),
deduplicates and validates images, and writes the unified dataset into
data/helmet/images/ and data/helmet/labels/ (flat), plus data.yaml.

Usage:
    python dataset_loader.py --source "roboflow:API_KEY:workspace/project/version"
    python dataset_loader.py --source "kaggle:owner/dataset-slug"
    python dataset_loader.py --source "https://example.com/helmets.zip"
    python dataset_loader.py --source "gdrive:FILE_ID"
    python dataset_loader.py --source "/local/path/to/dataset"
    python dataset_loader.py --source "hf:username/helmet-dataset"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "helmet"
OUT_IMAGES = OUT_DIR / "images"
OUT_LABELS = OUT_DIR / "labels"

TARGET_CLASSES = ["helmet", "no_helmet"]

# Synonym map: lowercased source class name -> target class name.
# Anything not found here is reported as unmapped and skipped.
CLASS_SYNONYMS = {
    "helmet": "helmet",
    "with helmet": "helmet",
    "with_helmet": "helmet",
    "hardhat": "helmet",
    "hard hat": "helmet",
    "hard_hat": "helmet",
    "wearing helmet": "helmet",
    "helment": "helmet",  # common misspelling in public datasets
    "no-helmet": "no_helmet",
    "no_helmet": "no_helmet",
    "without helmet": "no_helmet",
    "without_helmet": "no_helmet",
    "no helmet": "no_helmet",
    "head": "no_helmet",
    "person": "no_helmet",  # bare person box (no hardhat) treated as violation
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------
# Source acquisition
# --------------------------------------------------------------------------

def download_roboflow(spec: str, dest: Path) -> Path:
    """spec format: API_KEY:workspace/project/version"""
    from roboflow import Roboflow

    try:
        api_key, path = spec.split(":", 1)
        workspace, project, version = path.split("/")
    except ValueError:
        raise ValueError(
            "Roboflow source must look like "
            "'roboflow:API_KEY:workspace/project/version'"
        )
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(int(version)).download("yolov8", location=str(dest))
    return Path(dataset.location)


def download_kaggle(slug: str, dest: Path) -> Path:
    """slug format: owner/dataset-name"""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    dest.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
    return dest


def download_url(url: str, dest: Path) -> Path:
    import requests
    from tqdm import tqdm

    dest.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1].split("?")[0] or "download.zip"
    archive_path = dest / filename

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(archive_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=f"Downloading {filename}"
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    extract_archive(archive_path, dest)
    return dest


def download_gdrive(file_id: str, dest: Path) -> Path:
    import gdown

    dest.mkdir(parents=True, exist_ok=True)
    output = dest / "gdrive_download"
    result = gdown.download(id=file_id, output=str(output) + ".zip", quiet=False)
    if result is None:
        # Might be a folder link instead of a file
        gdown.download_folder(id=file_id, output=str(dest), quiet=False)
        return dest
    extract_archive(Path(result), dest)
    return dest


def download_hf(repo_id: str, dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    local_path = snapshot_download(
        repo_id=repo_id, repo_type="dataset", local_dir=str(dest)
    )
    return Path(local_path)


def use_local(path_str: str, dest: Path) -> Path:
    src = Path(path_str)
    if not src.exists():
        raise FileNotFoundError(f"Local dataset path does not exist: {src}")
    if src.resolve() == dest.resolve():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / src.name
    if not target.exists():
        if src.is_dir():
            shutil.copytree(src, target)
        else:
            shutil.copy2(src, target)
            extract_archive(target, dest)
            return dest
    return target


def extract_archive(archive_path: Path, dest: Path) -> None:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest)
    # else: not an archive (already a raw file), leave as-is


def resolve_source(source: str) -> Path:
    """Detect source type from the --source string and download/locate it."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if source.startswith("roboflow:"):
        print("[loader] Detected source type: Roboflow")
        return download_roboflow(source[len("roboflow:"):], RAW_DIR / "roboflow")

    if source.startswith("kaggle:"):
        print("[loader] Detected source type: Kaggle")
        return download_kaggle(source[len("kaggle:"):], RAW_DIR / "kaggle")

    if source.startswith("gdrive:"):
        print("[loader] Detected source type: Google Drive")
        return download_gdrive(source[len("gdrive:"):], RAW_DIR / "gdrive")

    if source.startswith("hf:"):
        print("[loader] Detected source type: Hugging Face Hub")
        return download_hf(source[len("hf:"):], RAW_DIR / "hf")

    if source.startswith("http://") or source.startswith("https://"):
        if "drive.google.com" in source:
            print("[loader] Detected source type: Google Drive (URL)")
            file_id = extract_gdrive_id(source)
            return download_gdrive(file_id, RAW_DIR / "gdrive")
        print("[loader] Detected source type: Direct URL archive")
        return download_url(source, RAW_DIR / "url")

    print("[loader] Detected source type: Local folder")
    return use_local(source, RAW_DIR / "local")


def extract_gdrive_id(url: str) -> str:
    import re

    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url) or re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if not m:
        raise ValueError(f"Could not extract file id from Google Drive URL: {url}")
    return m.group(1)


# --------------------------------------------------------------------------
# Format auto-detection
# --------------------------------------------------------------------------

def detect_label_format(root: Path) -> str:
    """Returns one of: 'yolo', 'voc', 'coco', 'csv', 'unknown'."""
    files = list(root.rglob("*"))
    names = [f.name.lower() for f in files if f.is_file()]

    if any(n == "_annotations.coco.json" or n == "instances.json" for n in names):
        return "coco"
    json_files = [f for f in files if f.suffix.lower() == ".json"]
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys():
                return "coco"
        except Exception:
            continue

    xml_files = [f for f in files if f.suffix.lower() == ".xml"]
    if xml_files:
        return "voc"

    csv_files = [f for f in files if f.suffix.lower() == ".csv"]
    if csv_files:
        return "csv"

    txt_files = [f for f in files if f.suffix.lower() == ".txt" and f.name != "classes.txt"]
    if txt_files:
        return "yolo"

    return "unknown"


def find_classes_file(root: Path) -> Optional[list[str]]:
    for candidate in ("classes.txt", "obj.names", "labels.txt"):
        for f in root.rglob(candidate):
            with open(f, "r", encoding="utf-8") as fh:
                return [line.strip() for line in fh if line.strip()]
    for f in root.rglob("data.yaml"):
        with open(f, "r", encoding="utf-8") as fh:
            y = yaml.safe_load(fh)
        if isinstance(y, dict) and "names" in y:
            names = y["names"]
            if isinstance(names, dict):
                return [names[k] for k in sorted(names, key=lambda x: int(x))]
            return list(names)
    return None


def map_class(name: str, unmapped: set) -> Optional[str]:
    key = name.strip().lower()
    mapped = CLASS_SYNONYMS.get(key)
    if mapped is None:
        unmapped.add(name)
        return None
    return mapped


# --------------------------------------------------------------------------
# Converters: everything -> YOLO txt (class_id x_center y_center w h, normalized)
# --------------------------------------------------------------------------

def voc_to_yolo(root: Path, unmapped: set) -> dict[Path, list[tuple]]:
    """Returns {image_path: [(class_name, xc, yc, w, h), ...]}"""
    results: dict[Path, list[tuple]] = {}
    image_index = index_images(root)

    for xml_path in root.rglob("*.xml"):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            print(f"[loader][warn] Could not parse XML: {xml_path}")
            continue
        rt = tree.getroot()
        filename_tag = rt.find("filename")
        stem = xml_path.stem
        img_path = image_index.get(stem)
        if img_path is None and filename_tag is not None:
            img_path = image_index.get(Path(filename_tag.text).stem)
        if img_path is None:
            print(f"[loader][warn] No matching image for VOC label {xml_path.name}")
            continue

        size = rt.find("size")
        img_w = int(size.findtext("width", "0")) if size is not None else 0
        img_h = int(size.findtext("height", "0")) if size is not None else 0
        if img_w == 0 or img_h == 0:
            with Image.open(img_path) as im:
                img_w, img_h = im.size

        boxes = []
        for obj in rt.findall("object"):
            cls_name = obj.findtext("name", "")
            mapped = map_class(cls_name, unmapped)
            if mapped is None:
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
            xc = ((xmin + xmax) / 2) / img_w
            yc = ((ymin + ymax) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h
            boxes.append((mapped, xc, yc, w, h))
        if boxes:
            results[img_path] = boxes
    return results


def coco_to_yolo(root: Path, unmapped: set) -> dict[Path, list[tuple]]:
    results: dict[Path, list[tuple]] = {}
    coco_files = []
    for f in root.rglob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys():
                coco_files.append((f, data))
        except Exception:
            continue

    image_index = index_images(root)

    for json_path, data in coco_files:
        cat_map = {c["id"]: c["name"] for c in data["categories"]}
        img_map = {img["id"]: img for img in data["images"]}
        anns_by_image = defaultdict(list)
        for ann in data["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        for image_id, img_meta in img_map.items():
            file_name = Path(img_meta["file_name"]).name
            stem = Path(file_name).stem
            img_path = image_index.get(stem)
            if img_path is None:
                # try relative to the json's directory
                candidate = json_path.parent / file_name
                if candidate.exists():
                    img_path = candidate
            if img_path is None:
                continue

            img_w = img_meta.get("width", 0)
            img_h = img_meta.get("height", 0)
            if not img_w or not img_h:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size

            boxes = []
            for ann in anns_by_image.get(image_id, []):
                cls_name = cat_map.get(ann["category_id"], "")
                mapped = map_class(cls_name, unmapped)
                if mapped is None:
                    continue
                x, y, w, h = ann["bbox"]  # COCO: top-left x,y,width,height
                xc = (x + w / 2) / img_w
                yc = (y + h / 2) / img_h
                boxes.append((mapped, xc, yc, w / img_w, h / img_h))
            if boxes:
                results[img_path] = boxes
    return results


def csv_to_yolo(root: Path, unmapped: set) -> dict[Path, list[tuple]]:
    """Expects columns like: filename,width,height,class,xmin,ymin,xmax,ymax
    (falls back to normalized xc,yc,w,h if xmin/ymin/xmax/ymax absent)."""
    results: dict[Path, list[tuple]] = defaultdict(list)
    image_index = index_images(root)
    image_size_cache: dict[Path, tuple[int, int]] = {}

    for csv_path in root.rglob("*.csv"):
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                continue
            fields = {f.lower(): f for f in reader.fieldnames}
            for row in reader:
                fname = row.get(fields.get("filename", ""), "")
                stem = Path(fname).stem
                img_path = image_index.get(stem)
                if img_path is None:
                    continue
                cls_name = row.get(fields.get("class", fields.get("label", "")), "")
                mapped = map_class(cls_name, unmapped)
                if mapped is None:
                    continue

                if img_path not in image_size_cache:
                    with Image.open(img_path) as im:
                        image_size_cache[img_path] = im.size
                img_w, img_h = image_size_cache[img_path]

                if all(k in fields for k in ("xmin", "ymin", "xmax", "ymax")):
                    xmin = float(row[fields["xmin"]])
                    ymin = float(row[fields["ymin"]])
                    xmax = float(row[fields["xmax"]])
                    ymax = float(row[fields["ymax"]])
                    xc = ((xmin + xmax) / 2) / img_w
                    yc = ((ymin + ymax) / 2) / img_h
                    w = (xmax - xmin) / img_w
                    h = (ymax - ymin) / img_h
                else:
                    xc = float(row[fields["xc"]])
                    yc = float(row[fields["yc"]])
                    w = float(row[fields["w"]])
                    h = float(row[fields["h"]])
                results[img_path].append((mapped, xc, yc, w, h))
    return dict(results)


def yolo_passthrough(root: Path, unmapped: set) -> dict[Path, list[tuple]]:
    """Source labels are already YOLO txt. Remap class ids -> target names
    using a discovered classes list, then back to normalized boxes."""
    results: dict[Path, list[tuple]] = {}
    source_classes = find_classes_file(root) or TARGET_CLASSES
    image_index = index_images(root)

    for txt_path in root.rglob("*.txt"):
        if txt_path.name in ("classes.txt", "obj.names", "labels.txt"):
            continue
        stem = txt_path.stem
        img_path = image_index.get(stem)
        if img_path is None:
            continue
        boxes = []
        with open(txt_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, w, h = parts
                try:
                    cls_name = source_classes[int(cls_id)]
                except (IndexError, ValueError):
                    continue
                mapped = map_class(cls_name, unmapped)
                if mapped is None:
                    continue
                boxes.append((mapped, float(xc), float(yc), float(w), float(h)))
        if boxes:
            results[img_path] = boxes
    return results


def whole_image_negative(root: Path, class_name: str) -> dict[Path, list[tuple]]:
    """For unlabeled image collections used purely as negative examples
    (e.g. everyday portraits with no safety helmet) — labels every image
    with a single full-frame box of the given class. Used to teach the
    model what non-helmet headwear/backgrounds look like, reducing false
    positives like classifying a baseball cap as a helmet."""
    if class_name not in TARGET_CLASSES:
        raise ValueError(f"--as-negative-class must be one of {TARGET_CLASSES}, got {class_name!r}")
    results: dict[Path, list[tuple]] = {}
    for img_path in index_images(root).values():
        results[img_path] = [(class_name, 0.5, 0.5, 1.0, 1.0)]
    return results


def index_images(root: Path) -> dict[str, Path]:
    index = {}
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            index[f.stem] = f
    return index


# --------------------------------------------------------------------------
# Post-processing: dedupe, validate, write out
# --------------------------------------------------------------------------

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def write_unified_dataset(annotations: dict[Path, list[tuple]]) -> dict:
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    class_to_id = {name: i for i, name in enumerate(TARGET_CLASSES)}
    seen_hashes: set[str] = set()
    instance_counts = defaultdict(int)
    skipped_corrupt = 0
    skipped_duplicate = 0
    written = 0

    for img_path, boxes in annotations.items():
        if not img_path.exists():
            continue
        if not is_valid_image(img_path):
            skipped_corrupt += 1
            print(f"[loader][warn] Corrupt/unreadable image skipped: {img_path}")
            continue

        h = file_hash(img_path)
        if h in seen_hashes:
            skipped_duplicate += 1
            continue
        seen_hashes.add(h)

        dest_stem = f"img_{h[:16]}"
        dest_img = OUT_IMAGES / f"{dest_stem}{img_path.suffix.lower()}"
        dest_label = OUT_LABELS / f"{dest_stem}.txt"

        shutil.copy2(img_path, dest_img)
        with open(dest_label, "w", encoding="utf-8") as fh:
            for cls_name, xc, yc, w, h_box in boxes:
                cls_id = class_to_id[cls_name]
                instance_counts[cls_name] += 1
                fh.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h_box:.6f}\n")
        written += 1

    return {
        "written": written,
        "skipped_corrupt": skipped_corrupt,
        "skipped_duplicate": skipped_duplicate,
        "instance_counts": dict(instance_counts),
    }


def write_data_yaml() -> None:
    config = {
        "path": str(OUT_DIR).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES,
    }
    with open(OUT_DIR / "data.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Universal helmet dataset loader")
    parser.add_argument("--source", required=True, help="Dataset source (see module docstring)")
    parser.add_argument(
        "--as-negative-class", default=None, choices=TARGET_CLASSES,
        help="Treat every image in the source as an unlabeled negative example: "
             "labels the whole frame as this class (e.g. 'no_helmet') instead of "
             "detecting a label format. For unlabeled photo collections used to "
             "teach the model what non-helmet headwear/scenes look like.",
    )
    args = parser.parse_args()

    print(f"[loader] Resolving source: {args.source}")
    root = resolve_source(args.source)
    print(f"[loader] Source materialized at: {root}")

    unmapped: set = set()
    if args.as_negative_class:
        print(f"[loader] Treating all images as whole-frame '{args.as_negative_class}' negatives")
        annotations = whole_image_negative(root, args.as_negative_class)
    else:
        fmt = detect_label_format(root)
        print(f"[loader] Detected label format: {fmt}")
        if fmt == "voc":
            annotations = voc_to_yolo(root, unmapped)
        elif fmt == "coco":
            annotations = coco_to_yolo(root, unmapped)
        elif fmt == "csv":
            annotations = csv_to_yolo(root, unmapped)
        elif fmt == "yolo":
            annotations = yolo_passthrough(root, unmapped)
        else:
            print("[loader][error] Could not detect a supported label format "
                  "(YOLO txt / Pascal VOC XML / COCO JSON / CSV). Aborting.")
            sys.exit(1)

    if unmapped:
        print("\n[loader][WARNING] The following source class names could not be "
              "confidently mapped to the target schema and were SKIPPED:")
        for name in sorted(unmapped):
            print(f"    - {name!r}")
        print("Add them to CLASS_SYNONYMS in dataset_loader.py if they should map "
              "to 'helmet' or 'no_helmet'.\n")

    if not annotations:
        print("[loader][error] No usable annotations after class mapping. Aborting.")
        sys.exit(1)

    stats = write_unified_dataset(annotations)
    write_data_yaml()

    print("\n===== Dataset Load Summary =====")
    print(f"Images written:        {stats['written']}")
    print(f"Skipped (corrupt):     {stats['skipped_corrupt']}")
    print(f"Skipped (duplicate):   {stats['skipped_duplicate']}")
    print("Instances per class:")
    total_instances = sum(stats["instance_counts"].values())
    for cls in TARGET_CLASSES:
        count = stats["instance_counts"].get(cls, 0)
        pct = (count / total_instances * 100) if total_instances else 0
        print(f"    {cls:<12} {count:>6}  ({pct:.1f}%)")
    if unmapped:
        print(f"Unmapped class names skipped: {len(unmapped)} (see warnings above)")
    print(f"\nUnified dataset written to: {OUT_IMAGES} / {OUT_LABELS}")
    print(f"data.yaml written to:       {OUT_DIR / 'data.yaml'}")
    print("Next step: python split_dataset.py")


if __name__ == "__main__":
    main()
