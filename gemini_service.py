import os
import time

from dotenv import load_dotenv
from google import genai


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured."
    )


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# GEMINI CONNECTION TEST
# =========================================================

def test_gemini():

    response = client.models.generate_content(

        model="gemini-3.1-flash-lite",

        contents=(
            "Respond with exactly: "
            "InkSense Gemini API is working!"
        )
    )

    return response.text


# =========================================================
# HANDWRITING DIGITIZATION
# =========================================================

def digitize_handwriting(
    image_bytes,
    language="English",
    mime_type="image/jpeg"
):
    """
    Send a handwriting image to Gemini and return
    only the extracted handwritten text.

    Supports images coming from:
    - Normal file upload
    - Browser camera
    - Live Camera / clip-on camera
    - JPEG
    - PNG
    - WebP
    """

    # -----------------------------------------------------
    # Validate image
    # -----------------------------------------------------

    if not image_bytes:

        raise ValueError(
            "Image data is empty."
        )

    # -----------------------------------------------------
    # Normalize MIME type
    # -----------------------------------------------------

    mime_type = (
        mime_type
        or "image/jpeg"
    ).lower().split(";")[0].strip()

    # Common browser MIME aliases
    mime_aliases = {

        "image/jpg":
            "image/jpeg",

        "image/pjpeg":
            "image/jpeg",

    }

    mime_type = mime_aliases.get(
        mime_type,
        mime_type
    )

    # -----------------------------------------------------
    # Validate supported image types
    # -----------------------------------------------------

    allowed_types = {

        "image/jpeg",
        "image/png",
        "image/webp"

    }

    if mime_type not in allowed_types:

        raise ValueError(
            f"Unsupported image MIME type: {mime_type}"
        )

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""
You are the handwriting digitization engine for InkSense AI.

Analyze the provided handwritten document image.

The document contains {language} handwriting.

This image may have been captured directly using a camera
attached to a sheet of paper.

Focus ONLY on the handwritten content visible on the paper.

Ignore:

- Camera frame
- Background
- Paper edges
- Shadows
- Fingers
- Hands
- Clip-on device
- Camera/device components
- Other objects that are not handwritten content

Your task is to accurately transcribe ONLY the handwritten
content visible in the image.

IMPORTANT RULES:

1. Convert the handwriting into clean digital text.

2. Preserve the original meaning.

3. Preserve the original wording as much as possible.

4. Preserve paragraph breaks and line structure where possible.

5. Read all clearly visible handwritten lines.

6. Read the handwriting even if it is slightly tilted,
   photographed at an angle, or captured from a camera.

7. Do not summarize the document.

8. Do not generate a title.

9. Do not generate key points.

10. Do not classify the document.

11. Do not add explanations.

12. Do not invent information.

13. If a word is unclear, make the best possible
    interpretation based only on what is visible.

14. Return ONLY the extracted handwritten text.

15. Do NOT return JSON.

16. Do NOT use Markdown code fences.

17. Do NOT describe the image.

18. Do NOT mention the camera or device in the output.

Return only the transcription.
"""

    # -----------------------------------------------------
    # Gemini request
    # -----------------------------------------------------

    max_retries = 3

    last_error = None

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(

                model="gemini-3.1-flash-lite",

                contents=[

                    {
                        "text": prompt
                    },

                    {
                        "inline_data": {

                            "mime_type":
                                mime_type,

                            "data":
                                image_bytes

                        }
                    }

                ]

            )

            # -------------------------------------------------
            # Validate Gemini response
            # -------------------------------------------------

            if not response:

                raise RuntimeError(
                    "Gemini returned no response."
                )

            response_text = getattr(
                response,
                "text",
                None
            )

            if not response_text:

                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            extracted_text = (
                response_text
                .strip()
            )

            if not extracted_text:

                raise RuntimeError(
                    "Gemini returned empty transcription."
                )

            return extracted_text

        except Exception as error:

            last_error = error

            print(
                f"GEMINI DIGITIZATION ERROR "
                f"(attempt {attempt + 1}/{max_retries}):",
                str(error)
            )

            # -------------------------------------------------
            # Retry
            # -------------------------------------------------

            if attempt < max_retries - 1:

                # Small delay before retry
                time.sleep(
                    2 ** attempt
                )

            else:

                raise RuntimeError(
                    f"Gemini digitization failed: "
                    f"{str(last_error)}"
                )