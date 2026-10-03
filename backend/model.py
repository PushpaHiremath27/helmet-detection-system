from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs" / "detect"
DATA_YAML = ROOT / "data" / "helmet" / "data.yaml"


def find_weight_files():
    return sorted(
        RUNS_DIR.glob("*/weights/best.pt")
    )


def load_model(weights_path: str):
    return YOLO(weights_path)