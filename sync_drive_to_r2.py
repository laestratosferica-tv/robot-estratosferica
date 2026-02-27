import os
import json
import io
from datetime import datetime
from zoneinfo import ZoneInfo

import boto3
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# =============================
# ENV
# =============================

BUCKET_NAME = os.getenv("BUCKET_NAME")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

GDRIVE_SERVICE_ACCOUNT_JSON = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON")
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")
GDRIVE_DONE_FOLDER_ID = os.getenv("GDRIVE_DONE_FOLDER_ID")

LOCAL_TZ = ZoneInfo("America/Bogota")


# =============================
# R2 CLIENT
# =============================

def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def upload_to_r2(key: str, data: bytes):
    s3 = r2_client()
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType="video/mp4",
    )


# =============================
# GOOGLE DRIVE CLIENT
# =============================

def get_drive_service():
    creds_info = json.loads(GDRIVE_SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds)


# =============================
# MAIN SYNC
# =============================

def run_drive_sync():
    print("===== DRIVE → R2 SYNC =====")

    service = get_drive_service()

    query = f"'{GDRIVE_FOLDER_ID}' in parents and mimeType='video/mp4' and trashed=false"

    results = service.files().list(
        q=query,
        fields="files(id, name)",
    ).execute()

    files = results.get("files", [])

    if not files:
        print("No hay videos nuevos en Drive.")
        return

    print(f"Encontrados {len(files)} videos.")

    for file in files:
        file_id = file["id"]
        filename = file["name"]

        print(f"Descargando: {filename}")

        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        fh.seek(0)
        video_bytes = fh.read()

        today = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
        r2_key = f"ugc/inbox/{today}__{filename}"

        print(f"Subiendo a R2: {r2_key}")
        upload_to_r2(r2_key, video_bytes)

        # mover a DONE
        service.files().update(
            fileId=file_id,
            addParents=GDRIVE_DONE_FOLDER_ID,
            removeParents=GDRIVE_FOLDER_ID,
        ).execute()

        print(f"Movido a UGC_DONE: {filename}")

    print("SYNC terminado.")


if __name__ == "__main__":
    run_drive_sync()
