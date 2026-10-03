"""Helmet compliance inference for the AI Helmet Safety Check system."""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO


# Classes from our trained model
WITH_HELMET = "With Helmet"
WITHOUT_HELMET = "Without Helmet"
LICENCE = "licence"


def annotate_image(
    model: YOLO,
    image: np.ndarray,
    conf: float,
    iou: float,
):
    """
    Run helmet detection on a single BGR image.

    Returns:
        annotated_bgr_image,
        detection_rows,
        class_counts
    """

    results = model.predict(
        image,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    result = results[0]
    names = result.names

    annotated = image.copy()

    detection_rows = []

    class_counts = {
        WITH_HELMET: 0,
        WITHOUT_HELMET: 0,
        LICENCE: 0,
    }

    for box in result.boxes:

        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = names[cls_id]

        x1, y1, x2, y2 = [
            round(value, 1)
            for value in box.xyxy[0].tolist()
        ]

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

        class_counts[class_name] = (
            class_counts.get(class_name, 0) + 1
        )

        # Decide annotation label
        label = f"{class_name} {confidence:.2f}"

        # Draw bounding box
        cv2.rectangle(
            annotated,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2,
        )

        # Draw label
        cv2.putText(
            annotated,
            label,
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated, detection_rows, class_counts


def create_safety_summary(class_counts: dict) -> dict:
    """
    Convert YOLO detections into a human-readable
    helmet safety result.
    """

    helmets = class_counts.get(WITH_HELMET, 0)
    without_helmet = class_counts.get(WITHOUT_HELMET, 0)

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

    if helmets > 0:

        return {
            "status": "HELMET DETECTED",
            "safety_status": "Compliant",
            "icon": "🟢",
            "message": (
                "A protective helmet was detected "
                "on the rider."
            ),
            "recommendation": (
                "No action required."
            ),
        }

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