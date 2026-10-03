import cv2
import tempfile

from ultralytics import YOLO


def process_video(
    model: YOLO,
    video_bytes: bytes,
    conf: float = 0.40,
    iou: float = 0.45,
):
    # Save uploaded video temporarily
    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    input_file.write(video_bytes)
    input_file.close()

    input_path = input_file.name

    # Open video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video
    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    output_path = output_file.name
    output_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the output video.")

    # Statistics
    frame_number = 0

    frames_with_helmet = 0
    frames_without_helmet = 0
    frames_unclear = 0

    max_without_helmet_confidence = 0.0

    # Process every frame
    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        results = model.predict(
            frame,
            conf=conf,
            iou=iou,
            verbose=False
        )

        result = results[0]

        names = result.names

        frame_has_helmet = False
        frame_has_no_helmet = False

        frame_no_helmet_confidence = 0.0

        # Process detections
        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = names[class_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Helmet
            if class_name == "With Helmet":

                frame_has_helmet = True

                label = (
                    f"With Helmet "
                    f"{confidence:.2f}"
                )

            # No helmet
            elif class_name == "Without Helmet":

                frame_has_no_helmet = True

                frame_no_helmet_confidence = max(
                    frame_no_helmet_confidence,
                    confidence
                )

                label = (
                    f"Without Helmet "
                    f"{confidence:.2f}"
                )

            # Licence
            else:

                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw label
            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # Determine frame status
        if frame_has_no_helmet:

            frames_without_helmet += 1

            max_without_helmet_confidence = max(
                max_without_helmet_confidence,
                frame_no_helmet_confidence
            )

            frame_status = "NO HELMET DETECTED"

        elif frame_has_helmet:

            frames_with_helmet += 1

            frame_status = "HELMET DETECTED"

        else:

            frames_unclear += 1

            frame_status = "RESULT UNCLEAR"

        # Draw status banner
        cv2.rectangle(
            frame,
            (10, 10),
            (430, 55),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            frame_status,
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Write processed frame
        writer.write(frame)

    # Cleanup
    cap.release()
    writer.release()

    return {
        "output_path": output_path,
        "total_frames": frame_number,
        "frames_with_helmet": frames_with_helmet,
        "frames_without_helmet": frames_without_helmet,
        "frames_unclear": frames_unclear,
        "max_without_helmet_confidence": max_without_helmet_confidence,
        "fps": fps,
        "width": width,
        "height": height,
    }