import os
import io
import re
import json
import time
import tempfile
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import boto3
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# -------------------------
# Env helpers
# -------------------------

def env_nonempty(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v or not v.strip():
        return default
    try:
        return int(v.strip())
    except Exception:
        return default


# -------------------------
# Config
# -------------------------

# R2
BUCKET_NAME = env_nonempty("BUCKET_NAME")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")

# Drive
GDRIVE_SERVICE_ACCOUNT_JSON = env_nonempty("GDRIVE_SERVICE_ACCOUNT_JSON")
GDRIVE_FOLDER_ID = env_nonempty("GDRIVE_FOLDER_ID")                 # "UGC_UPLOADS"
GDRIVE_DONE_FOLDER_ID = env_nonempty("GDRIVE_DONE_FOLDER_ID")       # "UGC_DONE"

# Behavior
UGC_INBOX_PREFIX = (env_nonempty("UGC_INBOX_PREFIX", "ugc/inbox/") or "ugc/inbox/").lstrip("/")
if UGC_INBOX_PREFIX and not UGC_INBOX_PREFIX.endswith("/"):
    UGC_INBOX_PREFIX += "/"

MAX_FILES_PER_RUN = env_int("MAX_FILES_PER_RUN", 20)
ONLY_VIDEO = env_bool("ONLY_VIDEO", True)
DRY_RUN = env_bool("DRY_RUN", False)


# -------------------------
# Helpers
# -------------------------

def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d__%H%M%S")

def sanitize_filename(name: str) -> str:
    name = (name or "file").strip()
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"[^a-zA-Z0-9\.\-_ ()]+", "", name)
    # evita nombres demasiado largos
    if len(name) > 140:
        root, ext = os.path.splitext(name)
        name = root[:120] + ext[:20]
    return name or "file.mp4"

def guess_ext_from_name(name: str, mime_type: str) -> str:
    _, ext = os.path.splitext(name or "")
    ext = (ext or "").lower().strip()
    if ext:
        return ext
    # fallback básico
    if "mp4" in (mime_type or "").lower():
        return ".mp4"
    if "quicktime" in (mime_type or "").lower():
        return ".mov"
    return ".mp4"


# -------------------------
# R2 client
# -------------------------

def r2_client():
    if not (R2_ENDPOINT_URL and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and BUCKET_NAME):
        raise RuntimeError("Faltan credenciales R2 (BUCKET_NAME, R2_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )

def r2_upload_file(local_path: str, key: str, content_type: str) -> None:
    s3 = r2_client()
    with open(local_path, "rb") as f:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=f.read(),
            ContentType=content_type or "application/octet-stream",
        )


# -------------------------
# Drive client
# -------------------------

def drive_client():
    if not GDRIVE_SERVICE_ACCOUNT_JSON:
        raise RuntimeError("Falta GDRIVE_SERVICE_ACCOUNT_JSON en secrets/env.")
    info = json.loads(GDRIVE_SERVICE_ACCOUNT_JSON)

    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def drive_list_files(drive, folder_id: str, max_files: int) -> List[Dict[str, Any]]:
    q = f"'{folder_id}' in parents and trashed = false"
    page_token = None
    out: List[Dict[str, Any]] = []

    while True:
        resp = drive.files().list(
            q=q,
            pageSize=min(100, max_files),
            fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
            pageToken=page_token,
            orderBy="modifiedTime asc",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        files = resp.get("files", []) or []
        out.extend(files)

        if len(out) >= max_files:
            return out[:max_files]

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return out

def drive_download_file(drive, file_id: str, dst_path: str) -> None:
    request = drive.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.FileIO(dst_path, "wb")
    downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024 * 8)  # 8MB
    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"  download {pct}%")
    fh.close()

def drive_move_to_done(drive, file_id: str, src_folder_id: str, done_folder_id: str) -> None:
    # remove src parent, add done parent
    drive.files().update(
        fileId=file_id,
        addParents=done_folder_id,
        removeParents=src_folder_id,
        fields="id, parents",
        supportsAllDrives=True,
    ).execute()


# -------------------------
# Main
# -------------------------

def main():
    if not GDRIVE_FOLDER_ID:
        raise RuntimeError("Falta GDRIVE_FOLDER_ID (carpeta origen en Drive).")
    if not GDRIVE_DONE_FOLDER_ID:
        raise RuntimeError("Falta GDRIVE_DONE_FOLDER_ID (carpeta DONE en Drive).")
    if not BUCKET_NAME:
        raise RuntimeError("Falta BUCKET_NAME.")
    if not UGC_INBOX_PREFIX:
        raise RuntimeError("UGC_INBOX_PREFIX inválido.")

    print("===== DRIVE SYNC -> R2 =====")
    print("DRY_RUN:", DRY_RUN)
    print("ONLY_VIDEO:", ONLY_VIDEO)
    print("MAX_FILES_PER_RUN:", MAX_FILES_PER_RUN)
    print("UGC_INBOX_PREFIX:", UGC_INBOX_PREFIX)
    print("GDRIVE_FOLDER_ID:", GDRIVE_FOLDER_ID)
    print("GDRIVE_DONE_FOLDER_ID:", GDRIVE_DONE_FOLDER_ID)

    drive = drive_client()

    files = drive_list_files(drive, GDRIVE_FOLDER_ID, MAX_FILES_PER_RUN)

    if ONLY_VIDEO:
        files = [f for f in files if (f.get("mimeType") or "").startswith("video/")]

    if not files:
        print("No hay archivos para mover (Drive vacío o no hay videos). ✅")
        return

    print(f"Encontrados {len(files)} archivo(s) para procesar.")

    processed = 0

    for f in files:
        file_id = f.get("id")
        name = f.get("name") or "video.mp4"
        mime = f.get("mimeType") or "application/octet-stream"

        if not file_id:
            continue

        safe_name = sanitize_filename(name)
        ext = guess_ext_from_name(safe_name, mime)
        # key único SIEMPRE (así si resubes el mismo video, entra otra vez)
        key = f"{UGC_INBOX_PREFIX}{now_utc_str()}__drive__{file_id}__{safe_name}"
        if not key.lower().endswith(ext):
            key += ext

        print("\n---")
        print("Drive file:", safe_name)
        print("mimeType:", mime)
        print("R2 key:", key)

        if DRY_RUN:
            print("[DRY_RUN] Saltando download/upload/move.")
            processed += 1
            continue

        with tempfile.TemporaryDirectory() as td:
            local_path = os.path.join(td, "in" + ext)

            print("Downloading from Drive...")
            drive_download_file(drive, file_id, local_path)

            print("Uploading to R2...")
            r2_upload_file(local_path, key, content_type=mime)

        print("Moving file to DONE folder in Drive...")
        drive_move_to_done(drive, file_id, GDRIVE_FOLDER_ID, GDRIVE_DONE_FOLDER_ID)

        print("OK ✅ moved to DONE and queued in R2 inbox.")
        processed += 1

    print("\n===== DONE =====")
    print("Processed:", processed)


if __name__ == "__main__":
    main()
