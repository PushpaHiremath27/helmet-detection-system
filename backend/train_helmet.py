"""
Trains a YOLOv8 helmet detector on the prepared dataset.

Usage:
    python train_helmet.py --data data/helmet/data.yaml --epochs 100 --imgsz 640
    python train_helmet.py --resume
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs" / "detect"


def main():
    parser = argparse.ArgumentParser(description="Train the YOLOv8 helmet detector")
    parser.add_argument("--data", default="data/helmet/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="Base weights, e.g. yolov8n.pt / yolov8s.pt / yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Keep at 640+ — helmets are small objects, 320 hurts recall")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--name", default="train")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None, help="e.g. '0' for GPU 0, 'cpu' for CPU")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", default="ram", choices=["ram", "disk", "none"],
                         help="Dataloader image cache — 'ram' is much faster for small datasets")
    args = parser.parse_args()

    if args.resume:
        last_ckpt = RUNS_DIR / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"No checkpoint to resume from at {last_ckpt}")
        model = YOLO(str(last_ckpt))
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        train_kwargs = dict(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            project=str(RUNS_DIR),
            name=args.name,
            exist_ok=True,
            optimizer="auto",
            cos_lr=True,
            mosaic=1.0,
            mixup=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            scale=0.5,
            workers=args.workers,
            cache={"ram": True, "disk": "disk", "none": False}[args.cache],
        )
        if args.device is not None:
            train_kwargs["device"] = args.device
        try:
            results = model.train(**train_kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and args.batch > 1:
                new_batch = max(1, args.batch // 2)
                print(f"[train][warn] CUDA OOM at batch={args.batch}, retrying with batch={new_batch}")
                train_kwargs["batch"] = new_batch
                results = model.train(**train_kwargs)
            else:
                raise

    run_dir = RUNS_DIR / args.name
    best_weights = run_dir / "weights" / "best.pt"

    print(f"\n[train] Training complete. Best weights: {best_weights}")

    print("[train] Running validation on best weights...")
    val_model = YOLO(str(best_weights))
    metrics = val_model.val(data=args.data)

    summary = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n===== Validation Metrics =====")
    for k, v in summary.items():
        print(f"{k:<10}: {v:.4f}")
    print(f"Saved to {run_dir / 'metrics.json'}")
    print(f"results.csv available at {run_dir / 'results.csv'}")

    print("[train] Exporting to ONNX...")
    val_model.export(format="onnx")
    print(f"[train] ONNX model exported alongside {best_weights}")


if __name__ == "__main__":
    main()
