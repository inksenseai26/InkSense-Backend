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
You are an advanced AI handwriting recognition and digitization engine for InkSense AI.

Your primary task is to accurately extract and transcribe handwritten content from the provided document image.

The document contains {language} handwriting.

The image may have been captured using a camera attached to a sheet of ordinary paper. Therefore, the image may not always be clean, properly aligned, or captured under ideal conditions.

## PRIMARY OBJECTIVE

Extract ALL readable handwritten content from the paper and convert it into accurate, clean digital text.

Your priority is to maximize handwriting recognition accuracy, even when the input image has imperfections.

## 1. IMAGE QUALITY AND ORIENTATION HANDLING

Before transcribing, carefully analyze the entire image.

The input image may contain one or more of the following conditions:

- Low-resolution or slightly blurry images.
- Rotated images (90°, 180°, or 270°).
- Slightly tilted or skewed handwriting.
- Perspective distortion caused by camera positioning.
- Uneven lighting or exposure.
- Shadows falling across the handwritten content.
- Faint, light-colored, or partially faded handwriting.
- Low contrast between handwriting and paper.
- Noisy or grainy images.
- Slightly folded, curved, or wrinkled paper.
- Handwriting that appears compressed or stretched due to camera angle.
- Different handwriting sizes and writing styles.

IMPORTANT:

Do not reject an image simply because its quality is imperfect.

Do not assume that the handwriting must be horizontally aligned.

Automatically interpret the document in its correct reading orientation, regardless of its original rotation.

Mentally compensate for minor rotation, skew, perspective distortion, lighting variations, and image imperfections when interpreting the handwriting.

## 2. HANDWRITING RECOGNITION

Examine the handwritten content carefully, including:

- Individual letters and their shapes.
- Connections between cursive letters.
- Words with overlapping or closely spaced characters.
- Numbers, mathematical symbols, and punctuation.
- Uppercase and lowercase letters.
- Faint or partially visible characters.
- Words written near the paper edges.

Use the surrounding words and sentence context to interpret unclear characters, but never invent information that is not reasonably supported by the visible handwriting.

If a word is partially unclear, make the most reasonable interpretation based on its visible characters and surrounding context.

If a word cannot be confidently identified, represent it as [unclear] rather than generating an unrelated or fabricated word.

## 3. CONTENT EXTRACTION

Focus ONLY on handwritten content physically visible on the paper.

Ignore all non-handwritten elements, including:

- Camera frame and background.
- Paper edges and shadows.
- Fingers and hands.
- Clip-on devices and camera components.
- Other objects surrounding the paper.
- Printed text, labels, or unrelated visual elements.

Do not mistake background objects or visual noise for handwritten content.

## 4. TRANSCRIPTION RULES

Follow these rules strictly:

1. Convert handwritten content into clean digital text.
2. Preserve the original meaning.
3. Preserve the original wording as closely as possible.
4. Maintain the original language and writing script.
5. Preserve paragraph breaks and line structure wherever possible.
6. Preserve headings, lists, numbering, and bullet points if they are handwritten.
7. Preserve mathematical expressions, numbers, and symbols accurately.
8. Correctly interpret handwriting even when letters are connected or slightly distorted.
9. Do not unnecessarily correct the author's grammar, spelling, or sentence structure.
10. Do not summarize, paraphrase, or rewrite the content.
11. Do not omit readable words or lines.
12. Do not add information that is not present in the image.
13. Do not generate content to fill missing or unreadable portions.

## 5. ACCURACY PRIORITY

When interpreting difficult handwriting, follow this priority:

1. Recognize the visible character shapes.
2. Identify possible words from the visible characters.
3. Use surrounding sentence context to resolve ambiguity.
4. Select the most probable interpretation supported by the handwriting.
5. If the text remains unreadable, use [unclear].

Always prioritize transcription accuracy over grammatical perfection.

Make every reasonable effort to recover readable handwriting from imperfect images.

## 6. STRICT OUTPUT REQUIREMENTS

Return ONLY the extracted handwritten text.

DO NOT:

- Generate a title.
- Generate a summary.
- Generate key points.
- Classify the document.
- Add explanations or observations.
- Describe image quality.
- Mention image rotation or camera conditions.
- Return JSON or XML.
- Use Markdown code fences.
- Add introductory or concluding statements.
- Include instructions or recommendations.

If no handwritten content can be reliably identified, return:

[No readable handwritten content detected]

## FINAL INSTRUCTION

Carefully inspect the entire document before responding.

Your goal is to produce the most accurate possible transcription of the handwritten content, even when the image is rotated, slightly blurry, low-resolution, poorly illuminated, skewed, or captured at an angle.

Never sacrifice accuracy by inventing unreadable content.

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