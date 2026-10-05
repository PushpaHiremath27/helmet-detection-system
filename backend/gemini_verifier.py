from pathlib import Path
import json
import os
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# PATH / ENVIRONMENT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
).strip()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = None

if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        client = None


# ============================================================
# DEFAULT RESULT
# ============================================================

def default_result(
    status: str = "UNCLEAR",
    reason: str = "Gemini verification is currently unavailable."
) -> Dict[str, Any]:

    return {
        "status": status,
        "verification": status,
        "confidence": 0.0,
        "reason": reason,
    }


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_response_text(response) -> str:

    try:
        text = response.text

        if text:
            return text.strip()

    except Exception:
        pass

    return ""


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text: str) -> Optional[Dict[str, Any]]:

    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences if Gemini returns them.
    if text.startswith("```"):
        lines = text.splitlines()

        if len(lines) >= 3:
            lines = lines[1:]

            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]

            text = "\n".join(lines).strip()

    # Direct JSON
    try:
        data = json.loads(text)

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    # Try to find JSON object inside the response.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            data = json.loads(candidate)

            if isinstance(data, dict):
                return data

        except Exception:
            pass

    return None


# ============================================================
# NORMALIZE GEMINI RESULT
# ============================================================

def normalize_result(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:

    if not isinstance(data, dict):
        return default_result(
            "UNCLEAR",
            "Gemini returned an invalid verification response."
        )

    raw_status = str(
        data.get("status")
        or data.get("verification")
        or data.get("result")
        or "UNCLEAR"
    ).strip().upper()

    # Normalize possible Gemini responses.
    if raw_status in {
        "SAFE",
        "WITH_HELMET",
        "HELMET",
        "HELMET_DETECTED"
    }:
        status = "SAFE"

    elif raw_status in {
        "NOT_SAFE",
        "NO_HELMET",
        "WITHOUT_HELMET",
        "NO_HELMET_DETECTED"
    }:
        status = "NOT_SAFE"

    else:
        status = "UNCLEAR"

    try:
        confidence = float(
            data.get("confidence", 0)
        )
    except Exception:
        confidence = 0.0

    # Gemini may return 0–100 instead of 0–1.
    if confidence > 1:
        confidence = confidence / 100.0

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    reason = str(
        data.get("reason")
        or data.get("explanation")
        or data.get("message")
        or "No explanation provided."
    ).strip()

    return {
        "status": status,
        "verification": status,
        "confidence": confidence,
        "reason": reason,
    }


# ============================================================
# GEMINI PROMPT
# ============================================================

GEMINI_PROMPT = """
You are an AI safety verification system for motorcycle helmet compliance.

Analyze the rider image carefully.

Your task is ONLY to determine whether the visible rider is wearing a
protective motorcycle helmet.

IMPORTANT:
- Do not assume a helmet is present just because the rider is on a motorcycle.
- Hair, head shape, cap, hood, shadows, or background objects are NOT helmets.
- If the rider's head is clearly visible and there is no protective helmet,
  return NOT_SAFE.
- If a clear protective helmet is visible on the rider's head,
  return SAFE.
- If the rider's head is blocked, too small, heavily blurred, or impossible
  to determine, return UNCLEAR.
- Do not rely on the YOLO answer blindly.
- Independently inspect the image.

Return ONLY valid JSON in exactly this structure:

{
  "status": "SAFE",
  "confidence": 0.95,
  "reason": "A protective helmet is clearly visible on the rider's head."
}

Allowed status values:

SAFE
NOT_SAFE
UNCLEAR

Confidence must be a number between 0 and 1.
"""


# ============================================================
# GEMINI IMAGE VERIFICATION
# ============================================================

def verify_helmet_with_gemini(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    yolo_result: str = ""
) -> Dict[str, Any]:

    if not image_bytes:
        return default_result(
            "UNCLEAR",
            "No image was provided to Gemini."
        )

    if client is None:
        return default_result(
            "UNCLEAR",
            "Gemini API client is not configured."
        )

    # --------------------------------------------------------
    # Give Gemini the YOLO result as context only.
    # Gemini must still inspect the image independently.
    # --------------------------------------------------------

    context_text = ""

    if yolo_result:
        context_text = f"""

YOLO PRELIMINARY RESULT:
{yolo_result}

This is only a preliminary computer-vision result.
Do NOT automatically agree with it.
Independently inspect the image.
"""

    prompt = GEMINI_PROMPT + context_text

    contents = [
        types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        ),
        prompt,
    ]

    # --------------------------------------------------------
    # Retry temporary server errors.
    #
    # We intentionally keep retries small so the application
    # does not become slow when Gemini is under heavy load.
    # --------------------------------------------------------

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents
            )

            response_text = extract_response_text(response)

            if not response_text:
                return default_result(
                    "UNCLEAR",
                    "Gemini returned an empty response."
                )

            parsed = extract_json(response_text)

            if parsed is None:
                return default_result(
                    "UNCLEAR",
                    "Gemini returned a response that could not be parsed."
                )

            return normalize_result(parsed)

        except Exception as exc:

            error_text = str(exc)
            error_upper = error_text.upper()

            # ------------------------------------------------
            # Temporary Gemini capacity/service problems.
            # ------------------------------------------------

            temporary_error = (
                "503" in error_upper
                or "UNAVAILABLE" in error_upper
                or "SERVICE UNAVAILABLE" in error_upper
                or "HIGH DEMAND" in error_upper
                or "429" in error_upper
                or "RESOURCE EXHAUSTED" in error_upper
            )

            if temporary_error and attempt < max_attempts - 1:

                # Small increasing delay:
                # attempt 1 -> 2 seconds
                # attempt 2 -> 4 seconds

                time.sleep(2 ** (attempt + 1))
                continue

            # ------------------------------------------------
            # Authentication problem
            # ------------------------------------------------

            if "401" in error_upper or "403" in error_upper:

                return default_result(
                    "UNCLEAR",
                    "Gemini authentication failed. Please check the API key."
                )

            # ------------------------------------------------
            # Model unavailable
            # ------------------------------------------------

            if "404" in error_upper or "NOT_FOUND" in error_upper:

                return default_result(
                    "UNCLEAR",
                    f"Gemini model '{GEMINI_MODEL}' is unavailable."
                )

            # ------------------------------------------------
            # Temporary server problem after retries.
            # ------------------------------------------------

            if temporary_error:

                return default_result(
                    "UNCLEAR",
                    "Gemini is temporarily unavailable or experiencing high demand. "
                    "The YOLO result can still be used."
                )

            # ------------------------------------------------
            # Any other unexpected problem.
            # ------------------------------------------------

            return default_result(
                "UNCLEAR",
                f"Gemini verification failed: {error_text[:300]}"
            )

    return default_result(
        "UNCLEAR",
        "Gemini verification could not be completed."
    )


# ============================================================
# OPENCV / IMAGE ARRAY COMPATIBILITY FUNCTION
# ============================================================

def verify_with_gemini(
    image,
    yolo_status: str = ""
) -> Dict[str, Any]:

    try:

        import cv2

        success, encoded = cv2.imencode(
            ".jpg",
            image
        )

        if not success:
            return default_result(
                "UNCLEAR",
                "Could not encode the image for Gemini."
            )

        image_bytes = encoded.tobytes()

        return verify_helmet_with_gemini(
            image_bytes=image_bytes,
            mime_type="image/jpeg",
            yolo_result=yolo_status
        )

    except Exception as exc:

        return default_result(
            "UNCLEAR",
            f"Could not prepare image for Gemini: {str(exc)[:300]}"
        )


# ============================================================
# MODULE TEST INFORMATION
# ============================================================

if __name__ == "__main__":

    print("========================================")
    print(" Gemini Verifier")
    print("========================================")

    print(
        "API KEY LOADED:",
        bool(GEMINI_API_KEY)
    )

    print(
        "MODEL:",
        GEMINI_MODEL
    )

    print(
        "CLIENT:",
        bool(client)
    )

    print("========================================")