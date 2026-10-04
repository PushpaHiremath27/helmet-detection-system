from pathlib import Path
import base64
import cv2
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from ultralytics import YOLO

from backend.gemini_verifier import verify_with_gemini


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend" / "web"
RUNS_DIR = ROOT / "runs" / "detect"


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# FIND YOLO MODEL
# ============================================================

MODEL_PATHS = list(
    RUNS_DIR.glob("*/weights/best.pt")
)

if not MODEL_PATHS:
    raise FileNotFoundError(
        "YOLO best.pt model was not found inside runs/detect/*/weights/"
    )

MODEL_PATH = MODEL_PATHS[0]

print(f"Loading YOLO model: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully.")
print("Classes:", model.names)


# ============================================================
# CONSTANTS
# ============================================================

WITH_HELMET = "With Helmet"
WITHOUT_HELMET = "Without Helmet"
LICENCE = "licence"


# ============================================================
# HELPER: YOLO DETECTION
# ============================================================

def run_yolo(image, confidence=0.40, iou=0.45):

    results = model.predict(
        image,
        conf=confidence,
        iou=iou,
        verbose=False
    )

    result = results[0]

    annotated = image.copy()

    detections = []

    counts = {
        WITH_HELMET: 0,
        WITHOUT_HELMET: 0,
        LICENCE: 0
    }

    highest_helmet_confidence = 0.0

    for box in result.boxes:

        class_id = int(box.cls[0])

        confidence_score = float(box.conf[0])

        class_name = result.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        x1 = round(x1, 1)
        y1 = round(y1, 1)
        x2 = round(x2, 1)
        y2 = round(y2, 1)

        detections.append({
            "class": class_name,
            "confidence": round(confidence_score, 3),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

        counts[class_name] = counts.get(class_name, 0) + 1

        if class_name in [WITH_HELMET, WITHOUT_HELMET]:

            highest_helmet_confidence = max(
                highest_helmet_confidence,
                confidence_score
            )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        if class_name == WITH_HELMET:

            box_color = (0, 200, 0)

        elif class_name == WITHOUT_HELMET:

            box_color = (0, 0, 255)

        else:

            box_color = (255, 180, 0)

        cv2.rectangle(
            annotated,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            box_color,
            2
        )

        label = f"{class_name} {confidence_score:.1%}"

        cv2.putText(
            annotated,
            label,
            (
                int(x1),
                max(25, int(y1) - 8)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2,
            cv2.LINE_AA
        )

    return (
        annotated,
        detections,
        counts,
        highest_helmet_confidence
    )


# ============================================================
# YOLO SAFETY SUMMARY
# ============================================================

def create_yolo_safety_summary(counts):

    helmets = counts.get(WITH_HELMET, 0)

    no_helmets = counts.get(WITHOUT_HELMET, 0)

    if no_helmets > 0:

        return {
            "status": "NO HELMET DETECTED",
            "safety_status": "Potential Violation",
            "icon": "🔴",
            "message": (
                "The rider's head is visible, "
                "but a protective helmet could not be identified."
            ),
            "recommendation": (
                "Helmet use is required for this rider."
            )
        }

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
            )
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
        )
    }


# ============================================================
# FRONTEND
# ============================================================

@app.route("/")
def index():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


@app.route("/style.css")
def style():

    return send_from_directory(
        FRONTEND_DIR,
        "style.css"
    )


@app.route("/script.js")
def script():

    return send_from_directory(
        FRONTEND_DIR,
        "script.js"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "model": str(MODEL_PATH),
        "classes": model.names
    })


# ============================================================
# IMAGE DETECTION API
# ============================================================

@app.route("/api/detect", methods=["POST"])
def detect():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded."
            }), 400

        uploaded_file = request.files["image"]

        file_bytes = uploaded_file.read()

        if not file_bytes:

            return jsonify({
                "error": "Uploaded image is empty."
            }), 400

        image_array = np.frombuffer(
            file_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return jsonify({
                "error": "Unable to read uploaded image."
            }), 400


        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        confidence = float(
            request.form.get(
                "confidence",
                0.40
            )
        )

        iou = float(
            request.form.get(
                "iou",
                0.45
            )
        )


        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        (
            annotated,
            detections,
            counts,
            highest_confidence
        ) = run_yolo(
            image,
            confidence,
            iou
        )


        yolo_safety = create_yolo_safety_summary(
            counts
        )


        # ----------------------------------------------------
        # Gemini Verification
        # ----------------------------------------------------

        gemini_result = verify_with_gemini(
            image,
            yolo_safety["status"]
        )


        gemini_status = gemini_result.get(
            "helmet_status",
            "UNCLEAR"
        )

        gemini_verification = gemini_result.get(
            "verification",
            "UNCLEAR"
        )

        gemini_confidence = float(
            gemini_result.get(
                "confidence",
                0
            )
        )

        gemini_reason = gemini_result.get(
            "reason",
            ""
        )


        # ----------------------------------------------------
        # FINAL DECISION
        # ----------------------------------------------------

        if (
            gemini_verification == "AGREE"
            and gemini_status == "HELMET_DETECTED"
        ):

            final_safety = {
                "status": "HELMET DETECTED",
                "safety_status": "Compliant",
                "icon": "🟢",
                "message": (
                    "A protective helmet was detected "
                    "and verified by AI."
                ),
                "recommendation": "No action required."
            }

            ai_verification = "AI VERIFIED"

        elif (
            gemini_verification == "AGREE"
            and gemini_status == "NO_HELMET_DETECTED"
        ):

            final_safety = {
                "status": "NO HELMET DETECTED",
                "safety_status": "Potential Violation",
                "icon": "🔴",
                "message": (
                    "The rider's head is visible, "
                    "but a protective helmet could not be identified."
                ),
                "recommendation": (
                    "Helmet use is required for this rider."
                )
            }

            ai_verification = "AI VERIFIED"

        elif gemini_verification == "DISAGREE":

            final_safety = {
                "status": "RESULT UNCLEAR",
                "safety_status": "Requires Review",
                "icon": "🟡",
                "message": (
                    "YOLO and Gemini produced different "
                    "helmet detection results."
                ),
                "recommendation": (
                    "Human review is recommended."
                )
            }

            ai_verification = "HUMAN REVIEW RECOMMENDED"

        else:

            final_safety = {
                "status": "RESULT UNCLEAR",
                "safety_status": "Requires Review",
                "icon": "🟡",
                "message": (
                    "The AI systems could not confidently "
                    "verify the helmet status."
                ),
                "recommendation": (
                    "Please review the image manually."
                )
            }

            ai_verification = "HUMAN REVIEW RECOMMENDED"


        # ----------------------------------------------------
        # Convert annotated image to PNG
        # ----------------------------------------------------

        success, encoded_image = cv2.imencode(
            ".png",
            annotated
        )

        if not success:

            return jsonify({
                "error": "Unable to encode result image."
            }), 500

        image_base64 = base64.b64encode(
            encoded_image.tobytes()
        ).decode("utf-8")


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "status": final_safety["status"],

            "icon": final_safety["icon"],

            "message": final_safety["message"],

            "safety_status": final_safety["safety_status"],

            "recommendation": final_safety["recommendation"],

            "ai_verification": ai_verification,

            "yolo_confidence": round(
                highest_confidence * 100,
                1
            ),

            "counts": {
                "With Helmet": counts.get(
                    WITH_HELMET,
                    0
                ),
                "Without Helmet": counts.get(
                    WITHOUT_HELMET,
                    0
                ),
                "licence": counts.get(
                    LICENCE,
                    0
                )
            },

            "total_detections": len(
                detections
            ),

            "detections": detections,

            "gemini": {
                "verification": gemini_verification,
                "helmet_status": gemini_status,
                "confidence": round(
                    gemini_confidence,
                    1
                ),
                "reason": gemini_reason
            },

            "image": (
                "data:image/png;base64,"
                + image_base64
            )
        })


    except Exception as error:

        print("ERROR:", error)

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ============================================================
# ANNOTATED IMAGE ENDPOINT
# ============================================================

@app.route("/api/detect-image", methods=["POST"])
def detect_image():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded."
            }), 400

        uploaded_file = request.files["image"]

        file_bytes = uploaded_file.read()

        image_array = np.frombuffer(
            file_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return jsonify({
                "error": "Invalid image."
            }), 400

        annotated, _, _, _ = run_yolo(
            image,
            0.40,
            0.45
        )

        success, encoded = cv2.imencode(
            ".png",
            annotated
        )

        if not success:

            return jsonify({
                "error": "Failed to encode image."
            }), 500

        return (
            encoded.tobytes(),
            200,
            {
                "Content-Type": "image/png"
            }
        )

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("")
    print("==============================================")
    print("       AI HELMET SAFETY CHECK SYSTEM")
    print("==============================================")
    print("")
    print("YOLO model:", MODEL_PATH)
    print("Frontend:", FRONTEND_DIR)
    print("")
    print("Open in browser:")
    print("http://127.0.0.1:5000")
    print("")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )