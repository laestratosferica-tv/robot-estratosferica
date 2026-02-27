import os
import io
import json
import boto3
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# =============================
# ENV
# =============================
GDRIVE_FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"]
GDRIVE_DONE_FOLDER_ID = os.environ.get("GDRIVE_DONE_FOLDER_ID")

BUCKET_NAME = os.environ["BUCKET_NAME"]
R2_ENDPOINT_URL = os.environ["R2_ENDPOINT_URL"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

SERVICE_ACCOUNT_INFO = json.loads(os.environ["GDRIVE_SERVICE_ACCOUNT_JSON"])

# =============================
# Drive auth
# =============================
SCOPES = ["https://www.googleapis.com/auth/drive"]

credentials = service_account.Credentials.from_service_account_info(
    SERVICE_ACCOUNT_INFO, scopes=SCOPES
)

drive_service = build("drive", "v3", credentials=credentials)

# =============================
# R2 client
# =============================
s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# =============================
# List files
# =============================
results = drive_service.files().list(
    q=f"'{GDRIVE_FOLDER_ID}' in parents and trashed = false",
    fields="files(id, name)",
).execute()

files = results.get("files", [])

if not files:
    print("No hay archivos nuevos en Drive.")
    exit(0)

for file in files:
    file_id = file["id"]
    file_name = file["name"]

    print(f"Descargando: {file_name}")

    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    fh.seek(0)

    r2_key = f"ugc/inbox/{file_name}"

    print(f"Subiendo a R2: {r2_key}")

    s3.upload_fileobj(fh, BUCKET_NAME, r2_key)

    if GDRIVE_DONE_FOLDER_ID:
        print("Moviendo archivo a carpeta DONE")

        drive_service.files().update(
            fileId=file_id,
            addParents=GDRIVE_DONE_FOLDER_ID,
            removeParents=GDRIVE_FOLDER_ID,
            fields="id, parents",
        ).execute()

print("Sync completado.")
