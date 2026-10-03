import io
import sys
from pathlib import Path

# ---------------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image
import numpy as np

from backend.model import find_weight_files, load_model
from backend.inference_image import (
    annotate_image,
    create_safety_summary,
)
from backend.gemini_verifier import verify_helmet_with_gemini
from backend.infer_video import process_video


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Helmet Safety Check",
    page_icon="🛡️",
    layout="wide",
)


# ---------------------------------------------------------
# CUSTOM STYLE
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 38px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 25px;
        }

        .section-title {
            font-size: 25px;
            font-weight: 650;
            margin-top: 20px;
            margin-bottom: 12px;
        }

        .result-box {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #ddd;
            margin-bottom: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🛡️ Helmet Safety Check</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered helmet compliance detection and safety analysis"
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Detection Settings")

    confidence_threshold = st.slider(
        "YOLO Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
    )

    iou_threshold = st.slider(
        "IoU Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
    )

    st.divider()

    st.subheader("🤖 AI Pipeline")

    st.write("1. YOLOv8 Detection")
    st.write("2. Gemini Vision Verification")
    st.write("3. Safety Status")
    st.write("4. Recommended Action")

    st.divider()

    st.subheader("📚 Model")

    weight_files = find_weight_files()

    if not weight_files:

        st.error("No trained model found.")

        st.info(
            "Make sure best.pt exists inside "
            "runs/detect/helmet_baseline/weights/"
        )

        st.stop()

    selected_model = st.selectbox(
        "Select YOLO model",
        weight_files,
        format_func=lambda path: str(path),
    )

    model = load_model(str(selected_model))

    st.success("Model loaded")


# ---------------------------------------------------------
# MAIN TABS
# ---------------------------------------------------------

image_tab, video_tab = st.tabs(
    [
        "🖼️ Image Detection",
        "🎥 Video Detection",
    ]
)


# =========================================================
# IMAGE DETECTION
# =========================================================

with image_tab:

    st.markdown(
        '<div class="section-title">🖼️ Upload an Image</div>',
        unsafe_allow_html=True,
    )

    uploaded_image = st.file_uploader(
        "Choose a rider image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_image is not None:

        image_bytes = uploaded_image.getvalue()

        original_pil = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        original_image = np.array(
            original_pil
        )

        # -------------------------------------------------
        # YOLO DETECTION
        # -------------------------------------------------

        with st.spinner(
            "🔍 Running YOLO detection..."
        ):

            annotated_image, detection_rows, class_counts = (
                annotate_image(
                    model,
                    original_image,
                    confidence_threshold,
                    iou_threshold,
                )
            )

        summary = create_safety_summary(
            class_counts
        )

        # -------------------------------------------------
        # RIDER RESULT
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">👤 Rider Analysis</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "With Helmet",
                class_counts.get(
                    "With Helmet",
                    0
                ),
            )

        with col2:

            st.metric(
                "Without Helmet",
                class_counts.get(
                    "Without Helmet",
                    0
                ),
            )

        st.markdown(
            f"""
            <div class="result-box">
                <h2>
                    {summary["icon"]}
                    {summary["status"]}
                </h2>

                <p>
                    <b>What we found:</b>
                    {summary["message"]}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # YOLO CONFIDENCE
        # -------------------------------------------------

        no_helmet_confidences = [
            row["confidence"]
            for row in detection_rows
            if row["class"] == "Without Helmet"
        ]

        helmet_confidences = [
            row["confidence"]
            for row in detection_rows
            if row["class"] == "With Helmet"
        ]

        if no_helmet_confidences:

            yolo_confidence = max(
                no_helmet_confidences
            )

        elif helmet_confidences:

            yolo_confidence = max(
                helmet_confidences
            )

        else:

            yolo_confidence = 0.0

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "YOLO Confidence",
                f"{yolo_confidence * 100:.1f}%",
            )

        with col2:

            st.metric(
                "Total Detections",
                len(detection_rows),
            )

        with col3:

            st.metric(
                "Licence",
                class_counts.get(
                    "licence",
                    0
                ),
            )

        # -------------------------------------------------
        # GEMINI VERIFICATION
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">🤖 AI Verification</div>',
            unsafe_allow_html=True,
        )

        yolo_result_text = (
            f"YOLO detected: {summary['status']}\n"
            f"With Helmet: "
            f"{class_counts.get('With Helmet', 0)}\n"
            f"Without Helmet: "
            f"{class_counts.get('Without Helmet', 0)}\n"
            f"Licence: "
            f"{class_counts.get('licence', 0)}\n"
            f"YOLO confidence: "
            f"{yolo_confidence:.3f}"
        )

        with st.spinner(
            "🤖 Gemini is independently verifying the image..."
        ):

            gemini_result = verify_helmet_with_gemini(
                image_bytes=image_bytes,
                mime_type=uploaded_image.type,
                yolo_result=yolo_result_text,
            )

        verification = gemini_result.get(
            "verification",
            "UNCLEAR"
        )

        gemini_status = gemini_result.get(
            "helmet_status",
            "UNCLEAR"
        )

        gemini_confidence = float(
            gemini_result.get(
                "confidence",
                0.0
            )
        )

        gemini_reason = gemini_result.get(
            "reason",
            "No explanation provided."
        )

        if verification == "AGREE":

            st.success(
                "✅ AI VERIFIED — Gemini agrees with YOLO."
            )

        elif verification == "DISAGREE":

            st.warning(
                "⚠️ AI REVIEW — Gemini disagrees with YOLO."
            )

        else:

            st.info(
                "🟡 AI REVIEW — Gemini could not clearly verify the result."
            )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Gemini Helmet Status",
                gemini_status,
            )

        with col2:

            st.metric(
                "Gemini Confidence",
                f"{gemini_confidence * 100:.1f}%",
            )

        st.write(
            f"**Gemini Reason:** {gemini_reason}"
        )

        # -------------------------------------------------
        # FINAL SAFETY STATUS
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">🛡️ Safety Status</div>',
            unsafe_allow_html=True,
        )

        if (
            verification == "AGREE"
            and gemini_status == "NO_HELMET_DETECTED"
        ):

            final_status = "🔴 Potential Violation"

            recommendation = (
                "Helmet use is required for this rider."
            )

        elif (
            verification == "AGREE"
            and gemini_status == "HELMET_DETECTED"
        ):

            final_status = "🟢 Compliant"

            recommendation = (
                "No action required."
            )

        elif verification == "DISAGREE":

            final_status = "🟡 Requires Review"

            recommendation = (
                "The AI systems disagree. "
                "Please review the image manually."
            )

        else:

            final_status = "🟡 Requires Review"

            recommendation = (
                "Please review the image manually."
            )

        st.info(final_status)

        st.markdown(
            f"""
            <div class="result-box">
                <h3>Recommended Action</h3>
                <p>{recommendation}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # DETECTION SUMMARY
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">📊 Detection Summary</div>',
            unsafe_allow_html=True,
        )

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            st.metric(
                "With Helmet",
                class_counts.get(
                    "With Helmet",
                    0
                ),
            )

        with summary_col2:

            st.metric(
                "Without Helmet",
                class_counts.get(
                    "Without Helmet",
                    0
                ),
            )

        with summary_col3:

            st.metric(
                "Licence",
                class_counts.get(
                    "licence",
                    0
                ),
            )

        # -------------------------------------------------
        # IMAGE ANALYSIS
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">🖼️ Image Analysis</div>',
            unsafe_allow_html=True,
        )

        image_col1, image_col2 = st.columns(2)

        with image_col1:

            st.subheader("Original")

            st.image(
                original_pil,
                width="stretch",
            )

        with image_col2:

            st.subheader("YOLO Detection")

            annotated_pil = Image.fromarray(
                annotated_image
            )

            st.image(
                annotated_pil,
                width="stretch",
            )

        # -------------------------------------------------
        # TECHNICAL DETAILS
        # -------------------------------------------------

        with st.expander(
            "🔧 Technical Details"
        ):

            st.write(
                "Model:",
                selected_model
            )

            st.write(
                "YOLO Confidence Threshold:",
                confidence_threshold
            )

            st.write(
                "IoU Threshold:",
                iou_threshold
            )

            st.write(
                "YOLO Detections:",
                detection_rows
            )

            st.write(
                "Gemini Verification:",
                gemini_result
            )

        # -------------------------------------------------
        # DOWNLOAD
        # -------------------------------------------------

        output_buffer = io.BytesIO()

        annotated_pil.save(
            output_buffer,
            format="PNG"
        )

        st.download_button(
            label="⬇️ Download Detection Image",
            data=output_buffer.getvalue(),
            file_name="helmet_detection_result.png",
            mime="image/png",
        )


# =========================================================
# VIDEO DETECTION
# =========================================================

with video_tab:

    st.markdown(
        '<div class="section-title">🎥 Upload a Video</div>',
        unsafe_allow_html=True,
    )

    uploaded_video = st.file_uploader(
        "Choose a rider video",
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

        st.write(
            "Click the button below to run YOLO detection on the video."
        )

        process_button = st.button(
            "🚀 Analyze Video",
            type="primary",
        )

        if process_button:

            video_bytes = uploaded_video.getvalue()

            progress_text = st.empty()

            progress_text.info(
                "🎥 Processing video... Please wait."
            )

            try:

                video_result = process_video(
                    model=model,
                    video_bytes=video_bytes,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                )

                progress_text.success(
                    "✅ Video processing completed."
                )

                # -------------------------------------------------
                # VIDEO STATISTICS
                # -------------------------------------------------

                st.markdown(
                    '<div class="section-title">📊 Video Analysis</div>',
                    unsafe_allow_html=True,
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Total Frames",
                        video_result[
                            "total_frames"
                        ],
                    )

                with col2:

                    st.metric(
                        "Helmet Frames",
                        video_result[
                            "frames_with_helmet"
                        ],
                    )

                with col3:

                    st.metric(
                        "No Helmet Frames",
                        video_result[
                            "frames_without_helmet"
                        ],
                    )

                with col4:

                    st.metric(
                        "Unclear Frames",
                        video_result[
                            "frames_unclear"
                        ],
                    )

                # -------------------------------------------------
                # VIDEO SAFETY RESULT
                # -------------------------------------------------

                no_helmet_frames = video_result[
                    "frames_without_helmet"
                ]

                helmet_frames = video_result[
                    "frames_with_helmet"
                ]

                if no_helmet_frames > 0:

                    st.error(
                        "🔴 NO HELMET DETECTED IN VIDEO"
                    )

                    st.write(
                        "At least one video frame contained "
                        "a rider detected without a helmet."
                    )

                elif helmet_frames > 0:

                    st.success(
                        "🟢 HELMET DETECTED IN VIDEO"
                    )

                else:

                    st.warning(
                        "🟡 RESULT UNCLEAR"
                    )

                st.metric(
                    "Maximum No-Helmet Confidence",
                    (
                        f"{video_result['max_without_helmet_confidence'] * 100:.1f}%"
                    ),
                )

                # -------------------------------------------------
                # OUTPUT VIDEO
                # -------------------------------------------------

                st.markdown(
                    '<div class="section-title">🎬 Processed Video</div>',
                    unsafe_allow_html=True,
                )

                with open(
                    video_result["output_path"],
                    "rb",
                ) as video_file:

                    processed_video_bytes = (
                        video_file.read()
                    )

                st.video(
                    processed_video_bytes
                )

                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=processed_video_bytes,
                    file_name="helmet_detection_video.mp4",
                    mime="video/mp4",
                )

            except Exception as error:

                st.error(
                    f"Video processing failed: {error}"
                )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "🛡️ Helmet Safety Check • YOLOv8 + Gemini Vision"
)