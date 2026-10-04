# 🪖 AI Helmet Safety Detection System

An AI-powered helmet compliance detection system that analyzes rider images and videos using **YOLOv8** and **Gemini Vision**.

The system detects helmet compliance, performs a secondary AI verification, and provides a clear safety status such as **Compliant**, **Potential Violation**, or **Requires Review**.

---

## 🚀 Project Overview

Helmet safety is an important part of road safety. This project uses computer vision and AI to automatically analyze whether a rider is wearing a helmet.

The system combines:

- **YOLOv8** for object detection
- **Gemini Vision** for secondary visual verification
- **Python** for the application logic
- **Streamlit** for the user interface

The goal is not only to detect objects but also to provide an understandable safety decision for the user.

---

## 🔄 AI Detection Pipeline

```text
              Rider Image / Video
                       │
                       ▼
              ┌─────────────────┐
              │    YOLOv8       │
              │ Object Detection│
              └────────┬────────┘
                       │
                       ▼
              Detection Results
                       │
                       ▼
              ┌─────────────────┐
              │ Gemini Vision   │
              │ AI Verification │
              └────────┬────────┘
                       │
                       ▼
             Compare AI Results
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
       Agreement                Disagreement
          │                         │
          ▼                         ▼
   Safety Decision           Human Review
```

---

## ✨ Key Features

### 🪖 Helmet Detection
Uses YOLOv8 to identify helmet-related objects in rider images.

### 🤖 AI Verification
Gemini Vision provides an additional visual analysis of the uploaded image.

### 🔍 Dual-AI Validation
The application compares the YOLO detection with the Gemini analysis.

### ⚠️ Safety Decision

The application can return:

- ✅ **Compliant** — helmet detected
- ⚠️ **Potential Violation** — no helmet detected
- 🔍 **Requires Review** — AI systems disagree or the result is unclear

### 📷 Image Analysis
Upload an image and receive an AI-generated safety analysis.

### 🎥 Video Analysis
The application also supports helmet detection on video input.

### 🎯 Configurable Detection
The interface provides controls for:

- YOLO confidence threshold
- IoU threshold

---

## 🧠 AI Models

### YOLOv8

The project uses **Ultralytics YOLOv8** for object detection.

YOLO performs the initial computer vision analysis and produces:

- Detected objects
- Bounding boxes
- Confidence scores
- Class information

The project uses the Ultralytics implementation rather than implementing the YOLO architecture from scratch.

### Gemini Vision

Gemini Vision is used as a secondary verification layer.

It receives the uploaded image along with the YOLO detection result and performs an independent visual assessment.

If the two AI systems disagree, the application does not force a final decision and instead recommends **Human Review**.

---

## 🏗️ Project Structure

```text
helmet-detection-system/
│
├── backend/
│   ├── gemini_verifier.py
│   ├── inference_image.py
│   ├── infer_video.py
│   └── model.py
│
├── data/
│   └── helmet/
│
├── frontend/
│   └── app.py
│
├── runs/
│   └── detect/
│
├── assets/
│
├── samples/
│
├── main.py
├── requirements.txt
├── requirements-space.txt
├── packages.txt
├── HF_SPACE_README.md
├── .gitignore
└── README.md
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Ultralytics YOLOv8 | Object detection |
| PyTorch | Deep learning framework |
| OpenCV | Image/video processing |
| Streamlit | Web interface |
| Google Gemini Vision | Secondary AI verification |

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/PushpaHiremath27/helmet-detection-system.git
cd helmet-detection-system
```

### 2. Create a Python virtual environment

```bash
py -3.11 -m venv .venv
```

### 3. Activate the environment

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Gemini API Configuration

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

**Do not commit your `.env` file or API key to GitHub.**

The API key is loaded by the backend Gemini verification module.

---

## ▶️ Run the Application

After activating the virtual environment:

```bash
python main.py
```

The application starts locally at:

```text
http://localhost:8501
```

Open the address in a web browser.

---

## 🔎 Image Detection Workflow

When a user uploads a rider image:

### Step 1 — Image Upload

The user uploads an image through the Streamlit interface.

### Step 2 — YOLO Detection

YOLOv8 analyzes the image and detects the relevant objects.

The application records:

- Detected classes
- Number of detections
- Confidence scores

### Step 3 — Gemini Verification

The image is also sent to Gemini Vision for an independent visual analysis.

### Step 4 — Result Comparison

The application compares the YOLO result with the Gemini result.

### Step 5 — Safety Status

The system determines the final status.

For example:

```text
Safety Status: Potential Violation
Recommended Action: Human Review
```

When the AI systems disagree or the result is unclear, the application intentionally returns:

```text
Safety Status: Requires Review
```

This avoids presenting an uncertain AI prediction as a definite safety violation.

---

## 🎯 Example Result

```text
AI VERIFICATION

Rider: 1

Detection:
No Helmet

Vision Analysis:
The rider's head is clearly visible, but
no helmet-like protective headgear is detected.

Confidence:
54.5%

Safety Status:
⚠️ Potential Violation
```

If YOLO and Gemini produce conflicting results:

```text
RESULT UNCLEAR

YOLO and Gemini produced different
helmet detection results.

Safety Status:
Requires Review

Recommended Action:
Human Review
```

---

## 🧩 Backend Components

### `backend/model.py`

Responsible for locating and loading trained YOLO model weights.

### `backend/inference_image.py`

Handles image inference and converts YOLO predictions into application-level detection results and safety information.

### `backend/infer_video.py`

Handles video-based helmet detection.

### `backend/gemini_verifier.py`

Handles communication with the Gemini Vision API and performs secondary AI verification.

### `frontend/app.py`

Provides the Streamlit user interface and connects the detection pipeline to the user.

### `main.py`

Starts the Streamlit application.

---

## 🧪 Testing Approach

The application can be tested using:

- Images containing riders wearing helmets
- Images containing riders without helmets
- Multiple riders in the same image
- Images with unclear visibility
- Video input
- Cases where YOLO and Gemini agree
- Cases where YOLO and Gemini disagree

The **Requires Review** state is intentionally used for uncertain or conflicting results.

---

## 🛡️ Safety-Oriented Design

The project is designed as an **AI-assisted verification system**, not as a replacement for human judgment.

When the AI result is uncertain, the application recommends human review instead of automatically treating the result as a confirmed violation.

This makes the system more suitable for real-world safety workflows where incorrect classifications can have consequences.

---

## 📌 Project Scope

This project demonstrates the integration of:

1. Computer vision
2. Object detection
3. Deep learning inference
4. LLM / vision-model verification
5. Image and video processing
6. Backend processing
7. Streamlit frontend development
8. AI-based decision logic

---

## 👩‍💻 Project Implementation

The application was organized into separate frontend, backend, model, inference, and verification components.

The YOLO architecture itself is provided through the **Ultralytics** framework. The project focuses on applying the trained detection model to the helmet-safety problem and integrating its output with Gemini Vision and the application interface.

---

## 🚧 Future Improvements

Possible future improvements include:

- Real-time camera detection
- Improved model accuracy
- More diverse training data
- Automatic violation reporting
- Rider tracking across video frames
- Database integration
- Cloud deployment
- GPU-accelerated inference
- Advanced analytics dashboard

---

## 📄 License

This project is intended for educational, portfolio, and demonstration purposes.