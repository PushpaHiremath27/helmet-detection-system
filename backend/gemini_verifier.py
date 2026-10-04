import os
import time
import json

from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    genai = None


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.8-flash"


# ============================================================
# GEMINI CLIENT
# ============================================================

client = None

if genai is not None and API_KEY:

    try:
        client = genai.Client(
            api_key=API_KEY
        )

    except Exception as error:

        print(
            "Gemini client initialization failed:",
            error
        )

        client = None


# ============================================================
# GEMINI HELMET VERIFICATION
# ============================================================

def verify_helmet_with_gemini(
    image_bytes=None,
    mime_type="image/jpeg",
    yolo_status="",
    **kwargs
):

    # --------------------------------------------------------
    # Support alternative argument names
    # --------------------------------------------------------

    if image_bytes is None:
        image_bytes = kwargs.get("image")

    if image_bytes is None:
        image_bytes = kwargs.get("bytes")

    if not yolo_status:
        yolo_status = kwargs.get(
            "yolo_result",
            kwargs.get("status", "")
        )

    if not mime_type:
        mime_type = "image/jpeg"


    # --------------------------------------------------------
    # Check Gemini configuration
    # --------------------------------------------------------

    if client is None:

        return {
            "verification": "UNCLEAR",
            "helmet_status": "UNCLEAR",
            "confidence": 0,
            "reason": (
                "Gemini verification is unavailable. "
                "Please check GEMINI_API_KEY and Gemini setup."
            )
        }


    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if image_bytes is None:

        return {
            "verification": "UNCLEAR",
            "helmet_status": "UNCLEAR",
            "confidence": 0,
            "reason": "No image was provided to Gemini."
        }


    # --------------------------------------------------------
    # Gemini prompt
    # --------------------------------------------------------

    prompt = f"""
You are an AI safety verification system.

Analyze the uploaded rider image and determine whether
the rider is wearing a protective helmet.

YOLO preliminary result:
{yolo_status}

Return ONLY valid JSON using exactly this structure:

{{
    "helmet_status": "HELMET_DETECTED",
    "verification": "AGREE",
    "confidence": 90,
    "reason": "Short explanation"
}}

Allowed helmet_status values:

HELMET_DETECTED
NO_HELMET_DETECTED
UNCLEAR

Allowed verification values:

AGREE
DISAGREE
UNCLEAR

Rules:

1. HELMET_DETECTED means a protective helmet is clearly visible.

2. NO_HELMET_DETECTED means the rider's head is visible
   and no protective helmet is visible.

3. UNCLEAR means the image is ambiguous or the rider/head
   cannot be reliably evaluated.

4. Compare your result with the YOLO preliminary result.

5. Use AGREE when your helmet status agrees with YOLO.

6. Use DISAGREE when your helmet status conflicts with YOLO.

7. Use UNCLEAR when you cannot confidently evaluate the image.

8. Confidence must be a number between 0 and 100.

9. Keep reason short and factual.
"""


    # --------------------------------------------------------
    # Gemini request with retries
    # --------------------------------------------------------

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_bytes
                        }
                    },
                    prompt
                ]
            )


            # ------------------------------------------------
            # Read response
            # ------------------------------------------------

            text = response.text.strip()


            # ------------------------------------------------
            # Remove Markdown JSON fences
            # ------------------------------------------------

            if text.startswith("```"):

                text = text.replace(
                    "```json",
                    ""
                )

                text = text.replace(
                    "```",
                    ""
                )

                text = text.strip()


            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            result = json.loads(text)


            # ------------------------------------------------
            # Read values
            # ------------------------------------------------

            helmet_status = str(
                result.get(
                    "helmet_status",
                    "UNCLEAR"
                )
            ).upper()


            verification = str(
                result.get(
                    "verification",
                    "UNCLEAR"
                )
            ).upper()


            confidence = float(
                result.get(
                    "confidence",
                    0
                )
            )


            reason = str(
                result.get(
                    "reason",
                    ""
                )
            )


            # ------------------------------------------------
            # Validate helmet status
            # ------------------------------------------------

            if helmet_status not in [
                "HELMET_DETECTED",
                "NO_HELMET_DETECTED",
                "UNCLEAR"
            ]:

                helmet_status = "UNCLEAR"


            # ------------------------------------------------
            # Validate verification
            # ------------------------------------------------

            if verification not in [
                "AGREE",
                "DISAGREE",
                "UNCLEAR"
            ]:

                verification = "UNCLEAR"


            # ------------------------------------------------
            # Validate confidence
            # ------------------------------------------------

            confidence = max(
                0,
                min(
                    100,
                    confidence
                )
            )


            # ------------------------------------------------
            # Return result
            # ------------------------------------------------

            return {
                "verification": verification,
                "helmet_status": helmet_status,
                "confidence": confidence,
                "reason": reason
            }


        except Exception as error:

            print(
                f"Gemini attempt {attempt + 1} failed:",
                error
            )

            if attempt < 2:
                time.sleep(2)


    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    return {
        "verification": "UNCLEAR",
        "helmet_status": "UNCLEAR",
        "confidence": 0,
        "reason": (
            "Gemini verification could not be completed."
        )
    }