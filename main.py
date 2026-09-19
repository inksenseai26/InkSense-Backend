from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Header,
)

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client, Client
from uuid import UUID, uuid4

from gemini_service import (
    digitize_handwriting,
    test_gemini,
)

import os
import hashlib


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL:
    raise RuntimeError(
        "SUPABASE_URL is not configured."
    )

if not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_KEY is not configured."
    )


# =========================================================
# OPTIONAL ADMIN EMAIL WHITELIST
# =========================================================

# Add your 4 authorized admin emails to .env like:
#
# ADMIN_EMAILS=admin1@gmail.com,admin2@gmail.com,admin3@gmail.com,admin4@gmail.com
#
# If ADMIN_EMAILS is empty, any authenticated Supabase
# user is allowed.
#
# For your final production setup, I recommend filling
# this with your exact 4 admin emails.

ADMIN_EMAILS_RAW = os.getenv(
    "ADMIN_EMAILS",
    ""
)

ADMIN_EMAILS = {
    email.strip().lower()
    for email in ADMIN_EMAILS_RAW.split(",")
    if email.strip()
}


# =========================================================
# SUPABASE CLIENT
# =========================================================

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

STORAGE_BUCKET = "document-images"


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="InkSense API",
    description="Backend API for InkSense AI",
    version="4.1.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://inksense-ai.vercel.app",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =========================================================
# PYDANTIC MODELS
# =========================================================

class DocumentUpdate(BaseModel):

    title: Optional[str] = None

    text: Optional[str] = None

    language: Optional[str] = None


# =========================================================
# AUTHENTICATION
# =========================================================

def get_authenticated_user(
    authorization: Optional[str]
):
    """
    Validate the Supabase access token sent by the frontend.

    Expected header:

        Authorization: Bearer <supabase_access_token>

    Returns the authenticated Supabase user.
    """

    # -----------------------------------------------------
    # Check Authorization header
    # -----------------------------------------------------

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authentication required."
        )

    # -----------------------------------------------------
    # Validate Bearer format
    # -----------------------------------------------------

    if not authorization.startswith(
        "Bearer "
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header."
        )

    access_token = (
        authorization[len("Bearer "):]
        .strip()
    )

    if not access_token:
        raise HTTPException(
            status_code=401,
            detail="Access token is missing."
        )

    # -----------------------------------------------------
    # Ask Supabase to validate the token
    # -----------------------------------------------------

    try:

        response = supabase.auth.get_user(
            access_token
        )

        user = getattr(
            response,
            "user",
            None
        )

        if user is None:

            # Some supabase-py versions expose
            # the response through .data
            data = getattr(
                response,
                "data",
                None
            )

            if data is not None:
                user = getattr(
                    data,
                    "user",
                    None
                )

        if user is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired authentication token."
            )

        # -------------------------------------------------
        # Check admin whitelist
        # -------------------------------------------------

        user_email = (
            getattr(
                user,
                "email",
                None
            )
            or ""
        ).lower().strip()

        if ADMIN_EMAILS:

            if user_email not in ADMIN_EMAILS:

                raise HTTPException(
                    status_code=403,
                    detail="This account is not authorized to access InkSense."
                )

        return user

    except HTTPException:
        raise

    except Exception as error:

        print(
            "AUTHENTICATION ERROR:",
            str(error)
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token."
        )


# =========================================================
# AUTHENTICATED USER ID
# =========================================================

def get_authenticated_user_id(
    authorization: Optional[str]
):
    """
    Return the UUID of the authenticated Supabase user.
    """

    user = get_authenticated_user(
        authorization
    )

    user_id = getattr(
        user,
        "id",
        None
    )

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Unable to determine authenticated user."
        )

    return str(user_id)


# =========================================================
# HELPERS
# =========================================================

def now_iso():

    return datetime.now(
        timezone.utc
    ).isoformat()


def calculate_hash(data):

    return hashlib.sha256(
        data
    ).hexdigest()


def create_signed_image_url(
    image_path,
    expires_in=3600
):
    """
    Generate a temporary signed URL for a private
    Supabase Storage image.
    """

    if not image_path:
        return None

    try:

        result = (
            supabase.storage
            .from_(STORAGE_BUCKET)
            .create_signed_url(
                image_path,
                expires_in
            )
        )

        if isinstance(result, dict):

            return (
                result.get("signedURL")
                or result.get("signedUrl")
                or result.get("signed_url")
            )

        return None

    except Exception as error:

        print(
            "SIGNED IMAGE URL ERROR:",
            str(error)
        )

        return None


def get_document_pages(
    document_id
):

    response = (
        supabase
        .table("document_pages")
        .select("*")
        .eq(
            "document_id",
            document_id
        )
        .order(
            "page_number"
        )
        .execute()
    )

    return response.data or []


def document_to_dict(
    document,
    pages=None
):

    if pages is None:

        pages = get_document_pages(
            document["id"]
        )

    # -----------------------------------------------------
    # Combine page text
    # -----------------------------------------------------

    extracted_text_parts = []

    for page in pages:

        page_text = page.get(
            "extracted_text"
        )

        if page_text:

            extracted_text_parts.append(
                page_text
            )

    combined_text = "\n\n".join(
        extracted_text_parts
    )

    # -----------------------------------------------------
    # First page image
    # -----------------------------------------------------

    image_path = None
    image_url = None

    if pages:

        image_path = pages[0].get(
            "image_path"
        )

        if image_path:

            image_url = (
                create_signed_image_url(
                    image_path
                )
            )

    # -----------------------------------------------------
    # Return document
    # -----------------------------------------------------

    return {

        "id":
            document["id"],

        "title":
            document["title"],

        "text":
            combined_text,

        "language":
            document.get(
                "language",
                "English"
            ),

        "type":
            document.get(
                "file_type",
                "PNG"
            ),

        "file_type":
            document.get(
                "file_type",
                "PNG"
            ),

        "status":
            document.get(
                "status",
                "digitised"
            ),

        "image":
            image_url,

        "image_path":
            image_path,

        "date":
            document.get(
                "created_at"
            ),

        "created_at":
            document.get(
                "created_at"
            ),

        "updated":
            document.get(
                "updated_at"
            ),

        "updated_at":
            document.get(
                "updated_at"
            ),

        "user_id":
            document.get(
                "user_id"
            ),

        "pages": [

            {
                "id":
                    page["id"],

                "page_number":
                    page["page_number"],

                "image_path":
                    page["image_path"],

                "image_url":
                    create_signed_image_url(
                        page.get(
                            "image_path"
                        )
                    ),

                "extracted_text":
                    page.get(
                        "extracted_text"
                    ),

                "original_extracted_text":
                    page.get(
                        "original_extracted_text"
                    ),

                "language":
                    page.get(
                        "language"
                    ),

                "source_type":
                    page.get(
                        "source_type"
                    ),

                "status":
                    page.get(
                        "page_status"
                    )
            }

            for page in pages

        ]

    }


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def home():

    return {

        "message":
            "InkSense Backend is running!"

    }


# =========================================================
# BACKEND TEST
# =========================================================

@app.get("/api/test")
def test():

    return {

        "status":
            "success",

        "message":
            "InkSense API is working!"

    }


# =========================================================
# SUPABASE TEST
# =========================================================

@app.get("/api/supabase-test")
def supabase_test():

    try:

        response = (
            supabase
            .table("documents")
            .select("id")
            .limit(1)
            .execute()
        )

        return {

            "success":
                True,

            "message":
                "Supabase connection is working.",

            "rows_found":
                len(
                    response.data or []
                )

        }

    except Exception as error:

        return {

            "success":
                False,

            "error":
                str(error)

        }


# =========================================================
# GEMINI TEST
# =========================================================

@app.get("/api/gemini-test")
def gemini_test():

    try:

        result = test_gemini()

        return {

            "success":
                True,

            "message":
                result

        }

    except Exception as error:

        return {

            "success":
                False,

            "error":
                str(error)

        }


# =========================================================
# DIGITISE HANDWRITING
# =========================================================

@app.post("/api/digitize")
async def digitize(

    image: UploadFile = File(...),

    language: str = Form("English")

):

    try:

        # -------------------------------------------------
        # Read image
        # -------------------------------------------------

        image_bytes = await image.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty."
            )

        # -------------------------------------------------
        # Normalize browser / camera MIME types
        # -------------------------------------------------

        content_type = (
            image.content_type
            or "image/jpeg"
        ).lower().split(";")[0].strip()

        mime_aliases = {

            "image/jpg":
                "image/jpeg",

            "image/pjpeg":
                "image/jpeg",

        }

        content_type = mime_aliases.get(
            content_type,
            content_type
        )

        # -------------------------------------------------
        # Supported image types
        # -------------------------------------------------

        allowed_types = {

            "image/jpeg",
            "image/png",
            "image/webp"

        }

        if content_type not in allowed_types:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported image type. "
                    "Please send JPG, JPEG, PNG or WebP."
                )
            )

        # -------------------------------------------------
        # Gemini
        # -------------------------------------------------

        extracted_text = digitize_handwriting(

            image_bytes,

            language,

            content_type

        )

        if not extracted_text:

            raise HTTPException(
                status_code=422,
                detail="No handwritten text was detected."
            )

        # -------------------------------------------------
        # Return result
        # -------------------------------------------------

        return {

            "success":
                True,

            "text":
                extracted_text,

            "filename":
                image.filename,

            "content_type":
                content_type,

            "language":
                language

        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            "DIGITIZATION ERROR:",
            str(error)
        )

        return {

            "success":
                False,

            "error":
                str(error)

        }


# =========================================================
# CREATE / SAVE DOCUMENT
# =========================================================

@app.post("/api/documents")
async def create_document(

    image: UploadFile = File(...),

    text: str = Form(...),

    title: str = Form(...),

    language: str = Form("English"),

    authorization: Optional[str] = Header(
        default=None
    )

):

    document_id = None
    storage_path = None

    try:

        # -------------------------------------------------
        # AUTHENTICATION
        # -------------------------------------------------

        authenticated_user_id = (
            get_authenticated_user_id(
                authorization
            )
        )

        # -------------------------------------------------
        # Read image
        # -------------------------------------------------

        image_bytes = await image.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Image is empty."
            )

        # -------------------------------------------------
        # MIME type
        # -------------------------------------------------

        content_type = (
            image.content_type
            or "image/jpeg"
        )

        content_type = (
            content_type
            .lower()
            .split(";")[0]
            .strip()
        )

        if content_type == "image/jpg":
            content_type = "image/jpeg"

        if content_type not in [
            "image/jpeg",
            "image/png",
            "image/webp"
        ]:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Only JPG, JPEG, PNG and WebP "
                    "images are supported."
                )
            )

        # -------------------------------------------------
        # File type
        # -------------------------------------------------

        if content_type == "image/png":

            file_type = "PNG"
            extension = "png"

        elif content_type == "image/webp":

            file_type = "WEBP"
            extension = "webp"

        else:

            file_type = "JPG"
            extension = "jpg"

        # -------------------------------------------------
        # Document ID
        # -------------------------------------------------

        document_id = str(
            uuid4()
        )

        # -------------------------------------------------
        # Storage filename
        # -------------------------------------------------

        safe_filename = (
            image.filename
            or f"document.{extension}"
        )

        storage_path = (
            f"{authenticated_user_id}/"
            f"{document_id}/"
            f"{safe_filename}"
        )

        # -------------------------------------------------
        # Upload image
        # -------------------------------------------------

        (
            supabase.storage
            .from_(STORAGE_BUCKET)
            .upload(

                path=storage_path,

                file=image_bytes,

                file_options={

                    "content-type":
                        content_type,

                    "cache-control":
                        "3600",

                    "upsert":
                        "false"

                }

            )
        )

        # -------------------------------------------------
        # Create document
        # -------------------------------------------------

        timestamp = now_iso()

        document_response = (
            supabase
            .table("documents")
            .insert({

                "id":
                    document_id,

                "user_id":
                    authenticated_user_id,

                "title":
                    (
                        title.strip()
                        or
                        "Untitled Handwritten Document"
                    ),

                "language":
                    language,

                "file_type":
                    file_type,

                "status":
                    "digitised",

                "created_at":
                    timestamp,

                "updated_at":
                    timestamp

            })
            .select("*")
            .execute()
        )

        if not document_response.data:

            raise Exception(
                "Document could not be created."
            )

        # -------------------------------------------------
        # Insert page
        # -------------------------------------------------

        page_response = (
            supabase
            .table("document_pages")
            .insert({

                "document_id":
                    document_id,

                "page_number":
                    1,

                "image_path":
                    storage_path,

                "image_hash":
                    calculate_hash(
                        image_bytes
                    ),

                "extracted_text":
                    text,

                "original_extracted_text":
                    text,

                "text_hash":
                    calculate_hash(
                        text.encode(
                            "utf-8"
                        )
                    ),

                "language":
                    language,

                "source_type":
                    "image",

                "page_status":
                    "digitised"

            })
            .select("*")
            .execute()
        )

        if not page_response.data:

            raise Exception(
                "Document page could not be created."
            )

        # -------------------------------------------------
        # Return document
        # -------------------------------------------------

        document = (
            document_response.data[0]
        )

        pages = page_response.data

        return {

            "success":
                True,

            "document":
                document_to_dict(
                    document,
                    pages
                )

        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            "CREATE DOCUMENT ERROR:",
            str(error)
        )

        # -------------------------------------------------
        # Cleanup storage
        # -------------------------------------------------

        if storage_path:

            try:

                (
                    supabase.storage
                    .from_(STORAGE_BUCKET)
                    .remove([
                        storage_path
                    ])
                )

            except Exception as cleanup_error:

                print(
                    "STORAGE CLEANUP ERROR:",
                    str(cleanup_error)
                )

        return {

            "success":
                False,

            "error":
                str(error)

        }


# =========================================================
# ADD PAGE TO EXISTING DOCUMENT
# =========================================================

@app.post(
    "/api/documents/{document_id}/pages"
)
async def add_document_page(

    document_id: str,

    image: UploadFile = File(...),

    text: str = Form(...),

    language: str = Form("English"),

    authorization: Optional[str] = Header(
        default=None
    )

):

    storage_path = None

    try:

        # -------------------------------------------------
        # AUTHENTICATION
        # -------------------------------------------------

        authenticated_user_id = (
            get_authenticated_user_id(
                authorization
            )
        )

        # -------------------------------------------------
        # Validate document ID
        # -------------------------------------------------

        try:

            UUID(document_id)

        except ValueError:

            raise HTTPException(
                status_code=400,
                detail="Invalid document_id."
            )

        # -------------------------------------------------
        # Get document
        # -------------------------------------------------

        document_response = (
            supabase
            .table("documents")
            .select("*")
            .eq(
                "id",
                document_id
            )
            .eq(
                "user_id",
                authenticated_user_id
            )
            .maybe_single()
            .execute()
        )

        document = (
            document_response.data
        )

        if not document:

            raise HTTPException(
                status_code=404,
                detail="Document not found."
            )

        # -------------------------------------------------
        # Read image
        # -------------------------------------------------

        image_bytes = await image.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Image is empty."
            )

        # -------------------------------------------------
        # Validate type
        # -------------------------------------------------

        content_type = (
            image.content_type
            or "image/jpeg"
        )

        content_type = (
            content_type
            .lower()
            .split(";")[0]
            .strip()
        )

        if content_type == "image/jpg":
            content_type = "image/jpeg"

        if content_type not in [
            "image/jpeg",
            "image/png",
            "image/webp"
        ]:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Only JPG, JPEG, PNG and WebP "
                    "images are supported."
                )
            )

        # -------------------------------------------------
        # Extension
        # -------------------------------------------------

        if content_type == "image/png":

            extension = "png"

        elif content_type == "image/webp":

            extension = "webp"

        else:

            extension = "jpg"

        # -------------------------------------------------
        # Existing pages
        # -------------------------------------------------

        pages = get_document_pages(
            document_id
        )

        # -------------------------------------------------
        # Next page number
        # -------------------------------------------------

        if pages:

            page_numbers = [

                page["page_number"]

                for page in pages

                if page.get(
                    "page_number"
                ) is not None

            ]

            max_page_number = max(
                page_numbers
            )

            next_page_number = (
                max_page_number + 1
            )

        else:

            next_page_number = 1

        # -------------------------------------------------
        # Storage filename
        # -------------------------------------------------

        safe_filename = (
            image.filename
            or
            f"page-{next_page_number}.{extension}"
        )

        storage_path = (
            f"{authenticated_user_id}/"
            f"{document_id}/"
            f"page-{next_page_number}-"
            f"{safe_filename}"
        )

        # -------------------------------------------------
        # Upload
        # -------------------------------------------------

        (
            supabase.storage
            .from_(STORAGE_BUCKET)
            .upload(

                path=storage_path,

                file=image_bytes,

                file_options={

                    "content-type":
                        content_type,

                    "cache-control":
                        "3600",

                    "upsert":
                        "false"

                }

            )
        )

        # -------------------------------------------------
        # Insert page
        # -------------------------------------------------

        page_response = (
            supabase
            .table("document_pages")
            .insert({

                "document_id":
                    document_id,

                "page_number":
                    next_page_number,

                "image_path":
                    storage_path,

                "image_hash":
                    calculate_hash(
                        image_bytes
                    ),

                "extracted_text":
                    text,

                "original_extracted_text":
                    text,

                "text_hash":
                    calculate_hash(
                        text.encode(
                            "utf-8"
                        )
                    ),

                "language":
                    language,

                "source_type":
                    "image",

                "page_status":
                    "digitised"

            })
            .select("*")
            .execute()
        )

        if not page_response.data:

            raise Exception(
                "Document page could not be created."
            )

        # -------------------------------------------------
        # Update document timestamp
        # -------------------------------------------------

        (
            supabase
            .table("documents")
            .update({
                "updated_at":
                    now_iso()
            })
            .eq(
                "id",
                document_id
            )
            .eq(
                "user_id",
                authenticated_user_id
            )
            .execute()
        )

        # -------------------------------------------------
        # Get updated document
        # -------------------------------------------------

        updated_document_response = (
            supabase
            .table("documents")
            .select("*")
            .eq(
                "id",
                document_id
            )
            .eq(
                "user_id",
                authenticated_user_id
            )
            .maybe_single()
            .execute()
        )

        updated_document = (
            updated_document_response.data
        )

        all_pages = get_document_pages(
            document_id
        )

        return {

            "success":
                True,

            "message":
                (
                    f"Page {next_page_number} "
                    "added successfully."
                ),

            "page":
                page_response.data[0],

            "document":
                document_to_dict(
                    updated_document,
                    all_pages
                )

        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            "ADD DOCUMENT PAGE ERROR:",
            str(error)
        )

        # -------------------------------------------------
        # Cleanup uploaded image
        # -------------------------------------------------

        if storage_path:

            try:

                (
                    supabase.storage
                    .from_(STORAGE_BUCKET)
                    .remove([
                        storage_path
                    ])
                )

            except Exception as cleanup_error:

                print(
                    "STORAGE CLEANUP ERROR:",
                    str(cleanup_error)
                )

        return {

            "success":
                False,

            "error":
                str(error)

        }


# =========================================================
# GET ALL DOCUMENTS
# =========================================================

@app.get("/api/documents")
def get_documents(

    authorization: Optional[str] = Header(
        default=None
    )

):

    try:

        # -------------------------------------------------
        # AUTHENTICATION
        # -------------------------------------------------

        authenticated_user_id = (
            get_authenticated_user_id(
                authorization
            )
        )

        # -------------------------------------------------
        # ONLY THIS USER'S DOCUMENTS
        # -------------------------------------------------

        response = (
            supabase
            .table("documents")
            .select("*")
            .eq(
                "user_id",
                authenticated_user_id
            )
            .order(
                "created_at",
                desc=True
            )
            .execute()
        )

        documents = []

        for document in (
            response.data or []
        ):

            pages = get_document_pages(
                document["id"]
            )

            documents.append(
                document_to_dict(
                    document,
                    pages
                )
            )

        return {

            "success":
                True,

            "documents":
                documents

        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            "GET DOCUMENTS ERROR:",
            str(error)
        )

        return {

            "success":
                False,

            "error":
                str(error)

        }


# =========================================================
# GET SINGLE DOCUMENT
# =========================================================

@app.get(
    "/api/documents/{document_id}"
)
def get_document(

    document_id: str,

    authorization: Optional[str] = Header(
        default=None
    )

):

    try:

        # -------------------------------------------------
        # AUTHENTICATION
        # -------------------------------------------------

        authenticated_user_id = (
            get_authenticated_user_id(
                authorization
            )
        )

        # -------------------------------------------------
        # Validate UUID
        # -------------------------------------------------

        try:

            UUID(document_id)

        except ValueError:

            raise HTTPException(
                status_code=400,
                detail="Invalid document_id."
            )

        # -------------------------------------------------
        # IMPORTANT:
        # Match BOTH document ID and user ID
        # -------------------------------------------------

        response = (
            supabase
            .table("documents")
            .select("*")
            .eq(
                "id",
                document_id
            )
            .eq(
                "user_id",
                authenticated_user_id
            )
            .maybe_single()
            .execute()
        )

        document = response.data

        if not document:

            raise HTTPException(
                status_code=404,
                detail="Document not found."
            )

        pages = get_document_pages(
            document_id
        )

        return {

            "success":
                True,

            "document":
                document_to_dict(
                    document,
                    pages
                )

        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            "GET DOCUMENT ERROR:",
            str(error)
        )

        return {

            "success":
                False,

            "error":
                str(error)

        }


# =========================================================
# UPDATE DOCUMENT
# =========================================================

@app.put(
    "/api/documents/{document_id}"
)
def update_document(

    document_id: str,

    document_data: DocumentUpdate,

    authorization: Optional[str] = Header(
        default=None
    )

):

    try:

        # -------------------------------------------------
        # AUTHENTICATION
        # -------------------------------------------------

        authenticated_user_id = (
            get_authenticated_user_id(
                authorization
            )
        )

        # -------------------------------------------------
        # Find user's document
        # -------------------------------------------------

        response = (
            supabase
            .table("documents")
            .select("*")
            .eq(
                "id",
                document_id
            )
            .eq(
                "user_id",
                authenticated_user_id
            )
            .maybe_single()
            .execute()
        )

        existing = response.data

        if not existing:

            raise HTTPException(
                status_code=404,
                detail="Document not found."
            )

        # -------------------------------------------------
        # Metadata update
        # -------------------------------------------------

        update_data = {}

        if document_data.title is not None:

            update_data["title"] = (
                document_data.title
            )

        if document_data.language is not None:

            update_data["language"] = (
                document_data.language
            )

        update_data["updated_at"] = (
            now_iso()
        )

        updated_response = (
            supabase
            .table("documents")
            .update(update_data)
            .eq(
                "id",
                document_id
            )
            .eq(
                "user_id",
                authenticated_user_id
            )
            .select("*")
            .execute()
        )

        if not updated_response.data:

            raise Exception(
                "Document update failed."
            )

        updated_document = (
            updated_response.data[0]
        )

        # -------------------------------------------------
        # Update text
        # -------------------------------------------------

        if document_data.text is not None:

            pages = get_document_pages(
                document_id
            )

            if pages:

                first_page = pages[0]

                (
                    supabase
                    .table("document_pages")
                    .update({

                        "extracted_text":
                            document_data.text,

                        "text_hash":
                            calculate_hash(
                                document_data.text.encode(
                                    "utf-8"
                                )
                            )

                    })
                    .eq(
                        "id",
                        first_page["id"]
                    )
                    .eq(
                        "document_id",
                        document_id
                    )
                    .execute()
                )

        # -------------------------------------------------
        # Final document
        # -------------------------------------------------

        pages = get_document_pages(
            document_id
        )

        return {

            "success":
                True,

            "document":
                document_to_dict(
                    updated_document,
                    pages
                )

        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            "UPDATE DOCUMENT ERROR:",
            str(error)
        )

        return {

            "success":
                False,

            "error":
                str(error)

        }


# =========================================================
# DELETE DOCUMENT
# =========================================================

@app.delete(
    "/api/documents/{document_id}"
)
def delete_document(

    document_id: str,

    authorization: Optional[str] = Header(
        default=None
    )

):

    try:

        # -------------------------------------------------
        # AUTHENTICATION
        # -------------------------------------------------

        authenticated_user_id = (
            get_authenticated_user_id(
                authorization
            )
        )

        # -------------------------------------------------
        # Find user's document
        # -------------------------------------------------

        response = (
            supabase
            .table("documents")
            .select("*")
            .eq(
                "id",
                document_id
            )
            .eq(
                "user_id",
                authenticated_user_id
            )
            .maybe_single()
            .execute()
        )

        document = response.data

        if not document:

            raise HTTPException(
                status_code=404,
                detail="Document not found."
            )

        # -------------------------------------------------
        # Get pages
        # -------------------------------------------------

        pages = get_document_pages(
            document_id
        )

        # -------------------------------------------------
        # Storage paths
        # -------------------------------------------------

        storage_paths = [

            page["image_path"]

            for page in pages

            if page.get(
                "image_path"
            )

        ]

        # -------------------------------------------------
        # Remove storage files
        # -------------------------------------------------

        if storage_paths:

            try:

                (
                    supabase.storage
                    .from_(STORAGE_BUCKET)
                    .remove(
                        storage_paths
                    )
                )

            except Exception as storage_error:

                print(
                    "STORAGE DELETE ERROR:",
                    str(storage_error)
                )

        # -------------------------------------------------
        # Delete pages
        # -------------------------------------------------

        (
            supabase
            .table("document_pages")
            .delete()
            .eq(
                "document_id",
                document_id
            )
            .execute()
        )

        # -------------------------------------------------
        # Delete document
        #
        # IMPORTANT:
        # Match user_id as well.
        # -------------------------------------------------

        (
            supabase
            .table("documents")
            .delete()
            .eq(
                "id",
                document_id
            )
            .eq(
                "user_id",
                authenticated_user_id
            )
            .execute()
        )

        return {

            "success":
                True,

            "message":
                "Document deleted successfully."

        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            "DELETE DOCUMENT ERROR:",
            str(error)
        )

        return {

            "success":
                False,

            "error":
                str(error)

        }