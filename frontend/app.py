import io
import sys
import time
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from backend.model import find_weight_files, load_model
from backend.inference_image import annotate_image, create_safety_summary
from backend.gemini_verifier import verify_helmet_with_gemini
from backend.infer_video import process_video


# ============================================================
# NORMALIZE STATUS
# ============================================================

def normalize_status(status):
    """
    Converts all possible status formats into one standard format.
    """

    value = str(status).upper().strip()

    value = value.replace("-", "_")
    value = value.replace(" ", "_")

    if value in [
        "HELMET_DETECTED",
        "HELMET",
        "WITH_HELMET",
    ]:
        return "HELMET_DETECTED"

    if value in [
        "NO_HELMET_DETECTED",
        "NO_HELMET",
        "WITHOUT_HELMET",
        "WITHOUT_HELMET_DETECTED",
    ]:
        return "NO_HELMET_DETECTED"

    return "UNCLEAR"


# ============================================================
# FINAL DECISION
# ============================================================

def make_final_decision(yolo_status, gemini_status):

    yolo_status = normalize_status(
        yolo_status
    )

    gemini_status = str(
        gemini_status
    ).upper().strip()

    # --------------------------------------------------------
    # Gemini unavailable / unclear
    # FALL BACK TO YOLO
    # --------------------------------------------------------

    if gemini_status in [
        "",
        "UNCLEAR",
        "UNAVAILABLE",
        "ERROR",
    ]:

        if yolo_status == "NO_HELMET_DETECTED":
            return "NOT_SAFE"

        if yolo_status == "HELMET_DETECTED":
            return "SAFE"

        return "REVIEW"

    # --------------------------------------------------------
    # Both agree
    # --------------------------------------------------------

    if (
        yolo_status == "HELMET_DETECTED"
        and gemini_status == "SAFE"
    ):
        return "SAFE"

    if (
        yolo_status == "NO_HELMET_DETECTED"
        and gemini_status == "NOT_SAFE"
    ):
        return "NOT_SAFE"

    # --------------------------------------------------------
    # They disagree
    # --------------------------------------------------------

    return "REVIEW"


# ============================================================
# RESULT DISPLAY
# ============================================================

def show_result(final_decision):

    if final_decision == "SAFE":

        st.success(
            "🟢 SAFE\n\n"
            "HELMET DETECTED\n\n"
            "The rider is wearing a helmet. Ride safely."
        )

    elif final_decision == "NOT_SAFE":

        st.error(
            "🔴 NOT SAFE\n\n"
            "HELMET NOT FOUND\n\n"
            "Please wear a helmet before riding."
        )

    else:

        st.warning(
            "🟡 PLEASE CHECK AGAIN\n\n"
            "RESULT UNCLEAR\n\n"
            "The helmet could not be verified confidently."
        )


# ============================================================
# LIVE CAMERA PROCESSOR
# ============================================================

class HelmetWebcamProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.model = None
        self.conf = 0.40
        self.iou = 0.45

        # Gemini is checked periodically,
        # not on every camera frame.
        self.gemini_interval = 15.0

        self.last_gemini_time = 0.0
        self.gemini_running = False

        self.gemini_status = "UNCLEAR"

        self.lock = threading.Lock()

    # --------------------------------------------------------
    # GEMINI BACKGROUND VERIFICATION
    # --------------------------------------------------------

    def run_gemini(
        self,
        image_bytes,
        yolo_status
    ):

        try:

            result = (
                verify_helmet_with_gemini(
                    image_bytes=image_bytes,
                    mime_type="image/jpeg",
                    yolo_result=(
                        f"YOLO status: {yolo_status}"
                    ),
                )
            )

            status = str(
                result.get(
                    "status",
                    "UNCLEAR"
                )
            ).upper()

            with self.lock:

                self.gemini_status = status

        except Exception:

            with self.lock:

                self.gemini_status = "UNCLEAR"

        finally:

            with self.lock:

                self.gemini_running = False

    # --------------------------------------------------------
    # FRAME PROCESSING
    # --------------------------------------------------------

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        if self.model is None:

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )

        try:

            results = self.model.predict(
                image,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )

            result = results[0]

            names = result.names

            has_helmet = False
            has_no_helmet = False

            for box in result.boxes:

                cls_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                raw_name = str(
                    names[cls_id]
                ).lower().strip()

                if raw_name in [
                    "helmet",
                    "with helmet",
                ]:

                    class_name = (
                        "WITH HELMET"
                    )

                    has_helmet = True

                    box_color = (
                        0,
                        180,
                        0
                    )

                elif raw_name in [
                    "no_helmet",
                    "no helmet",
                    "without helmet",
                ]:

                    class_name = (
                        "WITHOUT HELMET"
                    )

                    has_no_helmet = True

                    box_color = (
                        0,
                        0,
                        255
                    )

                else:

                    class_name = (
                        raw_name.upper()
                    )

                    box_color = (
                        255,
                        180,
                        0
                    )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # Bounding box
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2,
                )

                # Label
                font = (
                    cv2.FONT_HERSHEY_SIMPLEX
                )

                font_scale = 0.50

                thickness = 1

                (
                    text_width,
                    text_height
                ), baseline = cv2.getTextSize(
                    class_name,
                    font,
                    font_scale,
                    thickness,
                )

                label_y1 = max(
                    0,
                    y1
                    - text_height
                    - baseline
                    - 8,
                )

                label_y2 = (
                    label_y1
                    + text_height
                    + baseline
                    + 8
                )

                cv2.rectangle(
                    image,
                    (x1, label_y1),
                    (
                        x1
                        + text_width
                        + 10,
                        label_y2,
                    ),
                    box_color,
                    -1,
                )

                cv2.putText(
                    image,
                    class_name,
                    (
                        x1 + 5,
                        label_y2
                        - baseline
                        - 4,
                    ),
                    font,
                    font_scale,
                    (
                        255,
                        255,
                        255
                    ),
                    thickness,
                    cv2.LINE_AA,
                )

            # ------------------------------------------------
            # YOLO STATUS
            # ------------------------------------------------

            if has_no_helmet:

                yolo_status = (
                    "NO_HELMET_DETECTED"
                )

            elif has_helmet:

                yolo_status = (
                    "HELMET_DETECTED"
                )

            else:

                yolo_status = "UNCLEAR"

            # ------------------------------------------------
            # PERIODIC GEMINI
            # ------------------------------------------------

            now = time.time()

            should_run_gemini = (
                now
                - self.last_gemini_time
                >= self.gemini_interval
            )

            if (
                should_run_gemini
                and not self.gemini_running
                and yolo_status != "UNCLEAR"
            ):

                self.last_gemini_time = now

                self.gemini_running = True

                success, encoded = (
                    cv2.imencode(
                        ".jpg",
                        image,
                    )
                )

                if success:

                    image_bytes = (
                        encoded.tobytes()
                    )

                    thread = threading.Thread(
                        target=self.run_gemini,
                        args=(
                            image_bytes,
                            yolo_status,
                        ),
                        daemon=True,
                    )

                    thread.start()

            # ------------------------------------------------
            # GEMINI STATE
            # ------------------------------------------------

            with self.lock:

                gemini_status = (
                    self.gemini_status
                )

            # ------------------------------------------------
            # FINAL STATUS
            # ------------------------------------------------

            final_decision = (
                make_final_decision(
                    yolo_status,
                    gemini_status,
                )
            )

            if final_decision == "SAFE":

                status_text = (
                    "SAFE"
                )

                status_color = (
                    0,
                    180,
                    0
                )

            elif final_decision == "NOT_SAFE":

                status_text = (
                    "NOT SAFE"
                )

                status_color = (
                    0,
                    0,
                    255
                )

            else:

                status_text = (
                    "PLEASE CHECK"
                )

                status_color = (
                    0,
                    180,
                    255
                )

            # ------------------------------------------------
            # STATUS BAR
            # ------------------------------------------------

            cv2.rectangle(
                image,
                (10, 10),
                (360, 58),
                (
                    25,
                    25,
                    25
                ),
                -1,
            )

            cv2.putText(
                image,
                status_text,
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                status_color,
                2,
                cv2.LINE_AA,
            )

        except Exception:

            cv2.putText(
                image,
                "Camera inference error",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    0,
                    0,
                    255
                ),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SafeRide AI",
    page_icon="🪖",
    layout="wide",
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Detection Settings"
    )

    confidence_threshold = st.slider(
        "YOLO Confidence",
        0.10,
        0.90,
        0.40,
        0.05,
    )

    iou_threshold = st.slider(
        "IoU Threshold",
        0.10,
        0.90,
        0.45,
        0.05,
    )

    st.divider()

    st.subheader(
        "🤖 AI Pipeline"
    )

    st.write(
        "🪖 YOLO Helmet Detection"
    )

    st.write(
        "👁️ Gemini Vision Verification"
    )

    st.write(
        "🛡️ Safety Decision"
    )

    st.write(
        "👤 Recommended Action"
    )

    st.divider()

    st.subheader(
        "📚 Model"
    )

    weight_files = (
        find_weight_files()
    )

    if not weight_files:

        st.error(
            "No trained model found."
        )

        st.stop()

    selected_model = st.selectbox(
        "Select YOLO model",
        weight_files,
        format_func=lambda path: str(
            path
        ),
    )

    model = load_model(
        str(selected_model)
    )

    st.success(
        "Model loaded"
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🪖 SafeRide AI"
)

st.write(
    "Smart Helmet Safety"
)

st.write(
    "● SYSTEM READY"
)

st.header(
    "CHECK YOUR HELMET"
)

st.write(
    "AI-powered helmet safety verification"
)


# ============================================================
# TABS
# ============================================================

image_tab, video_tab, camera_tab = (
    st.tabs(
        [
            "📷 Check Photo",
            "🎥 Check Video",
            "📹 Live Camera",
        ]
    )
)


# ============================================================
# PHOTO TAB
# ============================================================

with image_tab:

    st.subheader(
        "📷 Check Rider Photo"
    )

    uploaded_image = st.file_uploader(
        "Upload a rider image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="image_uploader",
    )

    if uploaded_image is not None:

        image_bytes = (
            uploaded_image.getvalue()
        )

        original_pil = Image.open(
            io.BytesIO(
                image_bytes
            )
        ).convert("RGB")

        original_image = np.array(
            original_pil
        )

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        with st.spinner(
            "🪖 Analyzing rider image..."
        ):

            (
                annotated_image,
                detection_rows,
                class_counts,
            ) = annotate_image(
                model,
                original_image,
                confidence_threshold,
                iou_threshold,
            )

        summary = (
            create_safety_summary(
                class_counts
            )
        )

        # ----------------------------------------------------
        # COUNTS
        # ----------------------------------------------------

        helmet_count = class_counts.get(
            "With Helmet",
            0,
        )

        no_helmet_count = class_counts.get(
            "Without Helmet",
            0,
        )

        rider_count = (
            helmet_count
            + no_helmet_count
        )

        # ----------------------------------------------------
        # YOLO CONFIDENCE
        # ----------------------------------------------------

        confidences = [
            row["confidence"]
            for row in detection_rows
            if row["class"]
            in [
                "With Helmet",
                "Without Helmet",
            ]
        ]

        if confidences:

            yolo_confidence = max(
                confidences
            )

        else:

            yolo_confidence = 0.0

        # ----------------------------------------------------
        # YOLO STATUS
        #
        # IMPORTANT:
        # Normalize the actual summary status.
        # ----------------------------------------------------

        yolo_status = normalize_status(
            summary.get(
                "status",
                ""
            )
        )

        # ----------------------------------------------------
        # GEMINI
        # ----------------------------------------------------

        yolo_result_text = (
            f"YOLO status: "
            f"{yolo_status}\n"
            f"With Helmet: "
            f"{helmet_count}\n"
            f"Without Helmet: "
            f"{no_helmet_count}\n"
            f"YOLO confidence: "
            f"{yolo_confidence:.3f}"
        )

        with st.spinner(
            "🤖 Verifying helmet safety..."
        ):

            try:

                gemini_result = (
                    verify_helmet_with_gemini(
                        image_bytes=image_bytes,
                        mime_type=(
                            uploaded_image.type
                            or "image/jpeg"
                        ),
                        yolo_result=(
                            yolo_result_text
                        ),
                    )
                )

            except Exception:

                gemini_result = {
                    "status": "UNCLEAR",
                    "verification": "UNCLEAR",
                    "confidence": 0.0,
                    "reason": (
                        "Gemini is temporarily "
                        "unavailable."
                    ),
                }

        gemini_status = str(
            gemini_result.get(
                "status",
                "UNCLEAR",
            )
        ).upper().strip()

        gemini_verification = str(
            gemini_result.get(
                "verification",
                "UNCLEAR",
            )
        ).upper().strip()

        try:

            gemini_confidence = float(
                gemini_result.get(
                    "confidence",
                    0.0,
                )
            )

        except Exception:

            gemini_confidence = 0.0

        gemini_reason = str(
            gemini_result.get(
                "reason",
                "No explanation provided.",
            )
        )

        # ----------------------------------------------------
        # FINAL DECISION
        # ----------------------------------------------------

        final_decision = (
            make_final_decision(
                yolo_status,
                gemini_status,
            )
        )

        # ----------------------------------------------------
        # SAFETY RESULT
        # ----------------------------------------------------

        st.subheader(
            "🛡️ Safety Result"
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "👤 Rider",
                rider_count,
            )

        with col2:

            st.metric(
                "🪖 Helmet",
                helmet_count,
            )

        with col3:

            st.metric(
                "⚠️ No Helmet",
                no_helmet_count,
            )

        # ----------------------------------------------------
        # USER RESULT
        # ----------------------------------------------------

        if final_decision == "SAFE":

            st.success(
                "🟢 SAFE\n\n"
                "HELMET DETECTED\n\n"
                "The rider is wearing a helmet. "
                "Ride safely."
            )

        elif final_decision == "NOT_SAFE":

            st.error(
                "🔴 NOT SAFE\n\n"
                "HELMET NOT FOUND\n\n"
                "Please wear a helmet before riding."
            )

        else:

            st.warning(
                "🟡 PLEASE CHECK AGAIN\n\n"
                "RESULT UNCLEAR\n\n"
                "The AI checks produced different "
                "results. Please use a clearer image."
            )

        # ----------------------------------------------------
        # IMAGE ANALYSIS
        # ----------------------------------------------------

        st.subheader(
            "🖼️ Image Analysis"
        )

        image_col1, image_col2 = (
            st.columns(2)
        )

        with image_col1:

            st.caption(
                "Original"
            )

            st.image(
                original_pil,
                width="stretch",
            )

        with image_col2:

            st.caption(
                "🪖 Detection"
            )

            annotated_pil = Image.fromarray(
                annotated_image
            )

            st.image(
                annotated_pil,
                width="stretch",
            )

        # ----------------------------------------------------
        # MORE DETAILS
        # ----------------------------------------------------

        with st.expander(
            "⚙️ More Details"
        ):

            st.write(
                "YOLO Status:",
                yolo_status,
            )

            st.write(
                "YOLO Confidence:",
                f"{yolo_confidence * 100:.1f}%",
            )

            st.write(
                "Gemini Status:",
                gemini_status,
            )

            st.write(
                "Gemini Verification:",
                gemini_verification,
            )

            st.write(
                "Gemini Confidence:",
                f"{gemini_confidence * 100:.1f}%",
            )

            st.write(
                "Gemini Reason:",
                gemini_reason,
            )

            st.write(
                "YOLO Detections:"
            )

            st.json(
                detection_rows
            )

            st.write(
                "Model:",
                str(selected_model),
            )

            st.write(
                "YOLO Confidence Threshold:",
                confidence_threshold,
            )

            st.write(
                "IoU Threshold:",
                iou_threshold,
            )

            st.write(
                "Final Decision:",
                final_decision,
            )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        output_buffer = io.BytesIO()

        annotated_pil.save(
            output_buffer,
            format="PNG",
        )

        st.download_button(
            "⬇️ Download Detection Image",
            output_buffer.getvalue(),
            "helmet_detection_result.png",
            "image/png",
        )


# ============================================================
# VIDEO TAB
# ============================================================

with video_tab:

    st.subheader(
        "🎥 Check Rider Video"
    )

    uploaded_video = st.file_uploader(
        "Upload a rider video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
        ],
        key="video_uploader",
    )

    if uploaded_video is not None:

        st.video(
            uploaded_video
        )

        if st.button(
            "🚀 Analyze Video",
            type="primary",
        ):

            with st.spinner(
                "🎥 Processing video..."
            ):

                try:

                    video_result = (
                        process_video(
                            model=model,
                            video_bytes=(
                                uploaded_video.getvalue()
                            ),
                            conf=(
                                confidence_threshold
                            ),
                            iou=(
                                iou_threshold
                            ),
                        )
                    )

                    st.success(
                        "✅ Video processing completed."
                    )

                    col1, col2, col3, col4 = (
                        st.columns(4)
                    )

                    with col1:

                        st.metric(
                            "Total Frames",
                            video_result[
                                "total_frames"
                            ],
                        )

                    with col2:

                        st.metric(
                            "🪖 Helmet Frames",
                            video_result[
                                "frames_with_helmet"
                            ],
                        )

                    with col3:

                        st.metric(
                            "⚠️ No Helmet Frames",
                            video_result[
                                "frames_without_helmet"
                            ],
                        )

                    with col4:

                        st.metric(
                            "🟡 Unclear Frames",
                            video_result[
                                "frames_unclear"
                            ],
                        )

                    if (
                        video_result[
                            "frames_without_helmet"
                        ] > 0
                    ):

                        st.error(
                            "🔴 NO HELMET DETECTED IN VIDEO"
                        )

                    elif (
                        video_result[
                            "frames_with_helmet"
                        ] > 0
                    ):

                        st.success(
                            "🟢 HELMET DETECTED IN VIDEO"
                        )

                    else:

                        st.warning(
                            "🟡 RESULT UNCLEAR"
                        )

                    if (
                        "output_path"
                        in video_result
                    ):

                        with open(
                            video_result[
                                "output_path"
                            ],
                            "rb",
                        ) as f:

                            processed_video = (
                                f.read()
                            )

                        st.subheader(
                            "🎬 Processed Video"
                        )

                        st.video(
                            processed_video
                        )

                        st.download_button(
                            "⬇️ Download Processed Video",
                            processed_video,
                            "helmet_detection_video.mp4",
                            "video/mp4",
                        )

                except Exception as error:

                    st.error(
                        f"Video processing failed: {error}"
                    )


# ============================================================
# LIVE CAMERA
# ============================================================

with camera_tab:

    st.subheader(
        "📹 Live Helmet Safety Check"
    )

    st.write(
        "Start the camera and point it at the rider."
    )

    camera_col1, camera_col2 = (
        st.columns([3, 1])
    )

    with camera_col1:

        ctx = webrtc_streamer(
            key="saferide-live-camera-final-v4",

            video_processor_factory=(
                HelmetWebcamProcessor
            ),

            media_stream_constraints={
                "video": True,
                "audio": False,
            },

            async_processing=True,
        )

    with camera_col2:

        st.markdown(
            "### 🪖 Safety Guide"
        )

        st.write(
            "🟢 Helmet Detected"
        )

        st.write(
            "🔴 No Helmet Detected"
        )

        st.write(
            "🟡 Please Check Again"
        )

        st.caption(
            "YOLO checks the live camera continuously."
        )

        st.caption(
            "Gemini performs periodic verification."
        )

    if ctx.video_processor:

        ctx.video_processor.model = model

        ctx.video_processor.conf = (
            confidence_threshold
        )

        ctx.video_processor.iou = (
            iou_threshold
        )

    st.info(
        "Keep the rider's head clearly visible "
        "and use good lighting."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🪖 SafeRide AI • Smart Helmet Safety Detection"
)