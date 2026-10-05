from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO


def normalize_class_name(model_class: str) -> str:
    """
    Convert different YOLO class names into the application's
    standard class names.
    """

    name = str(model_class).lower().strip().replace("-", "_")

    if name in {
        "helmet",
        "with_helmet",
        "with helmet",
        "wearing_helmet",
        "wearing helmet",
    }:
        return "With Helmet"

    if name in {
        "no_helmet",
        "no helmet",
        "without_helmet",
        "without helmet",
        "not_helmet",
        "not helmet",
    }:
        return "Without Helmet"

    return str(model_class).strip()


def draw_clean_detection(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    class_name: str,
) -> np.ndarray:
    """
    Draw a clean, professional detection box.

    The confidence percentage is intentionally NOT displayed
    on the image.
    """

    output = image.copy()

    height, width = output.shape[:2]

    # ---------------------------------------------------------
    # Add a little padding around the detected head
    # ---------------------------------------------------------
    box_width = x2 - x1
    box_height = y2 - y1

    # Slightly more vertical padding because the detection
    # should not look too tightly cropped around the head.
    pad_x = max(8, int(box_width * 0.12))
    pad_y = max(8, int(box_height * 0.15))

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(width - 1, int(x2 + pad_x))
    y2 = min(height - 1, int(y2 + pad_y))

    # ---------------------------------------------------------
    # Colors
    # ---------------------------------------------------------
    if class_name == "Without Helmet":
        # Red BGR
        box_color = (0, 0, 220)
    elif class_name == "With Helmet":
        # Green BGR
        box_color = (0, 180, 0)
    else:
        # Orange BGR
        box_color = (0, 165, 255)

    # ---------------------------------------------------------
    # Bounding box
    # ---------------------------------------------------------
    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        box_color,
        2,
        cv2.LINE_AA,
    )

    # ---------------------------------------------------------
    # Clean label
    # ---------------------------------------------------------
    label = class_name.upper()

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Small, professional font
    font_scale = 0.48
    text_thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        text_thickness,
    )

    # ---------------------------------------------------------
    # Position label above bounding box
    # ---------------------------------------------------------
    label_x = x1
    label_y = y1 - 7

    # If there isn't enough space above the box,
    # put the label inside the top of the box.
    if label_y - text_height - baseline < 0:
        label_y = y1 + text_height + 7

    background_x1 = label_x
    background_y1 = label_y - text_height - baseline - 4
    background_x2 = label_x + text_width + 8
    background_y2 = label_y + 3

    # Keep label background inside image
    background_x1 = max(0, background_x1)
    background_y1 = max(0, background_y1)
    background_x2 = min(width - 1, background_x2)
    background_y2 = min(height - 1, background_y2)

    # ---------------------------------------------------------
    # Label background
    # ---------------------------------------------------------
    cv2.rectangle(
        output,
        (background_x1, background_y1),
        (background_x2, background_y2),
        box_color,
        -1,
    )

    # ---------------------------------------------------------
    # White label text
    # ---------------------------------------------------------
    cv2.putText(
        output,
        label,
        (label_x + 4, label_y - 1),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )

    return output


def annotate_image(
    model: YOLO,
    image: np.ndarray,
    conf: float,
    iou: float,
):
    """
    Run YOLO detection and create a clean annotated image.

    Returns:
        annotated image
        detection rows
        class counts
    """

    # ---------------------------------------------------------
    # YOLO prediction
    # ---------------------------------------------------------
    results = model.predict(
        image,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    result = results[0]

    names = result.names

    # Work on a copy so the original image remains unchanged.
    annotated = image.copy()

    detection_rows = []

    class_counts = {
        "With Helmet": 0,
        "Without Helmet": 0,
    }

    # ---------------------------------------------------------
    # No detections
    # ---------------------------------------------------------
    if result.boxes is None or len(result.boxes) == 0:
        return annotated, detection_rows, class_counts

    # ---------------------------------------------------------
    # Process every detection
    # ---------------------------------------------------------
    for box in result.boxes:

        cls_id = int(box.cls[0])

        confidence = float(box.conf[0])

        # Original YOLO class name
        model_class = str(names[cls_id])

        # Convert to application class name
        class_name = normalize_class_name(model_class)

        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        x1 = round(float(x1), 1)
        y1 = round(float(y1), 1)
        x2 = round(float(x2), 1)
        y2 = round(float(y2), 1)

        # -----------------------------------------------------
        # Store detection information
        # -----------------------------------------------------
        detection_rows.append(
            {
                "class": class_name,
                "confidence": round(confidence, 3),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

        # -----------------------------------------------------
        # Count classes
        # -----------------------------------------------------
        class_counts[class_name] = (
            class_counts.get(class_name, 0) + 1
        )

        # -----------------------------------------------------
        # Draw clean detection
        # -----------------------------------------------------
        annotated = draw_clean_detection(
            annotated,
            x1,
            y1,
            x2,
            y2,
            class_name,
        )

    return annotated, detection_rows, class_counts


def create_safety_summary(class_counts: dict) -> dict:
    """
    Convert YOLO detection counts into a simple safety result.
    """

    helmets = class_counts.get("With Helmet", 0)

    without_helmet = class_counts.get(
        "Without Helmet",
        0,
    )

    # ---------------------------------------------------------
    # Without helmet
    # ---------------------------------------------------------
    if without_helmet > 0:

        return {
            "status": "NO HELMET DETECTED",
            "safety_status": "Potential Violation",
            "icon": "🔴",
            "message": (
                "The rider's head is visible, but a "
                "protective helmet could not be identified."
            ),
            "recommendation": (
                "Helmet use is required for this rider."
            ),
        }

    # ---------------------------------------------------------
    # Helmet detected
    # ---------------------------------------------------------
    if helmets > 0:

        return {
            "status": "HELMET DETECTED",
            "safety_status": "Compliant",
            "icon": "🟢",
            "message": (
                "A protective helmet was detected on the rider."
            ),
            "recommendation": (
                "No action required."
            ),
        }

    # ---------------------------------------------------------
    # Unclear
    # ---------------------------------------------------------
    return {
        "status": "RESULT UNCLEAR",
        "safety_status": "Requires Review",
        "icon": "🟡",
        "message": (
            "The system could not confidently determine "
            "whether the rider is wearing a helmet."
        ),
        "recommendation": (
            "Please review the image manually."
        ),
    }