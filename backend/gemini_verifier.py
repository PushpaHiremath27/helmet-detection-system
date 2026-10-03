import os
import json
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ---------------------------------------------------------
# LOAD API KEY
# ---------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found in the .env file."
    )


# ---------------------------------------------------------
# GEMINI CLIENT
# ---------------------------------------------------------

client = genai.Client(
    api_key=API_KEY
)

MODEL_NAME = "gemini-3.8-flash"


# ---------------------------------------------------------
# GEMINI VERIFICATION
# ---------------------------------------------------------

def verify_helmet_with_gemini(
    image_bytes: bytes,
    mime_type: str,
    yolo_result: str,
):

    prompt = f"""
You are an AI safety verification system.

Analyze the provided motorcycle rider image.

Determine whether the rider is:

1. Wearing a protective helmet
2. Not wearing a protective helmet
3. Unclear

YOLO produced this result:

{yolo_result}

Independently inspect the image.

Return ONLY valid JSON in this exact format:

{{
    "verification": "AGREE",
    "helmet_status": "HELMET_DETECTED",
    "confidence": 0.90,
    "reason": "Short explanation"
}}

Allowed verification values:

AGREE
DISAGREE
UNCLEAR

Allowed helmet_status values:

HELMET_DETECTED
NO_HELMET_DETECTED
UNCLEAR

confidence must be between 0.0 and 1.0.

Do not use markdown.
Do not use code fences.
"""


    # -----------------------------------------------------
    # RETRY TEMPORARY GEMINI ERRORS
    # -----------------------------------------------------

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    types.Part.from_text(
                        text=prompt
                    ),
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type,
                    ),
                ],
            )

            response_text = response.text.strip()

            # Remove accidental markdown code fences
            if response_text.startswith("```"):

                response_text = (
                    response_text
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

            result = json.loads(
                response_text
            )

            return result


        except Exception as error:

            error_text = str(error)

            # Temporary server overload
            if "503" in error_text or "UNAVAILABLE" in error_text:

                if attempt < max_attempts - 1:

                    time.sleep(2)

                    continue

            # Other errors or final retry failure
            return {
                "verification": "UNCLEAR",
                "helmet_status": "UNCLEAR",
                "confidence": 0.0,
                "reason": (
                    f"Gemini verification failed: "
                    f"{error}"
                ),
            }


    return {
        "verification": "UNCLEAR",
        "helmet_status": "UNCLEAR",
        "confidence": 0.0,
        "reason": "Gemini verification was temporarily unavailable.",
    }