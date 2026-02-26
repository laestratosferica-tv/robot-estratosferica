import os
import re
import json
import time
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import boto3
import requests


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


@dataclass
class UGCConfig:
    r2_endpoint_url: str
    aws_access_key_id: str
    aws_secret_access_key: str
    bucket_name: str
    r2_public_base_url: str

    openai_api_key: str
    openai_model: str

    enable_ig_publish: bool
    ig_user_id: str
    ig_access_token: str
    graph_version: str

    inbox_prefix: str
    processed_prefix: str
    failed_prefix: str
    output_reels_prefix: str
    state_key: str

    reel_seconds_max: int
    max_per_run: int


def load_cfg() -> UGCConfig:
    r2_public = (env_nonempty("R2_PUBLIC_BASE_URL") or "").rstrip("/")
    if not r2_public:
        raise RuntimeError("Falta R2_PUBLIC_BASE_URL")

    graph_version = (env_nonempty("GRAPH_VERSION", "v25.0") or "v25.0").lstrip("v")
    model = env_nonempty("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"

    return UGCConfig(
        r2_endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        bucket_name=os.environ["BUCKET_NAME"],
        r2_public_base_url=r2_public,

        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_model=model,

        enable_ig_publish=env_bool("ENABLE_IG_PUBLISH", False),
        ig_user_id=env_nonempty("IG_USER_ID", "") or "",
        ig_access_token=env_nonempty("IG_ACCESS_TOKEN", "") or "",
        graph_version=graph_version,

        inbox_prefix=(env_nonempty("UGC_INBOX_PREFIX", "ugc/inbox/") or "ugc/inbox/"),
        processed_prefix=(env_nonempty("UGC_PROCESSED_PREFIX", "ugc/processed/") or "ugc/processed/"),
        failed_prefix=(env_nonempty("UGC_FAILED_PREFIX", "ugc/failed/") or "ugc/failed/"),
        output_reels_prefix=(env_nonempty("UGC_OUTPUT_REELS_PREFIX", "ugc/outputs/reels/") or "ugc/outputs/reels/"),
        state_key=(env_nonempty("UGC_STATE_KEY", "ugc/state/state.json") or "ugc/state/state.json"),

        reel_seconds_max=env_int("UGC_REEL_SECONDS", 20),
        max_per_run=env_int("UGC_MAX_PER_RUN", 3),
    )


def r2_client(cfg: UGCConfig):
    return boto3.client(
        "s3",
        endpoint_url=cfg.r2_endpoint_url,
        aws_access_key_id=cfg.aws_access_key_id,
        aws_secret_access_key=cfg.aws_secret_access_key,
        region_name="auto",
    )

def public_url(cfg: UGCConfig, key: str) -> str:
    return f"{cfg.r2_public_base_url}/{key}"

def list_mp4(s3, bucket: str, prefix: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            k = obj["Key"]
            if k.lower().endswith(".mp4") and not k.endswith("/"):
                out.append(obj)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return out

def get_json_or_default(s3, bucket: str, key: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read().decode("utf-8")
        return json.loads(data)
    except Exception:
        return default

def put_json(s3, bucket: str, key: str, payload: Dict[str, Any]) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

def copy_then_delete(s3, bucket: str, src_key: str, dst_key: str) -> None:
    s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": src_key}, Key=dst_key)
    s3.delete_object(Bucket=bucket, Key=src_key)


def generate_caption(cfg: UGCConfig, filename: str) -> str:
    prompt = f"""
Eres un creador de contenido de esports LATAM (energético, divertido, sin groserías fuertes).
Escribe un caption para un Reel de gameplay.

Archivo: {filename}

Reglas:
- 1-2 líneas máximo
- termina con 5-10 hashtags (gaming/esports/latam)
- NO inventes equipos/torneos específicos
- cierra con una pregunta corta
"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.openai_api_key}", "Content-Type": "application/json"}
    body = {
        "model": cfg.openai_model,
        "messages": [
            {"role": "system", "content": "Eres un experto en captions virales de esports LATAM."},
            {"role": "user", "content": prompt.strip()},
        ],
        "temperature": 0.9,
    }
    r = requests.post(url, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    j = r.json()
    return (j["choices"][0]["message"]["content"] or "").strip()


DURATION_TAG_RE = re.compile(r"__s(\d{1,3})", re.IGNORECASE)

def seconds_from_filename(filename: str) -> Optional[int]:
    m = DURATION_TAG_RE.search(filename or "")
    if not m:
        return None
    try:
        s = int(m.group(1))
        if 1 <= s <= 120:
            return s
    except Exception:
        return None
    return None

def probe_video_duration_seconds(path: str) -> int:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe falló: {(p.stderr or '')[:1000]}")
    raw = (p.stdout or "").strip()
    if not raw:
        raise RuntimeError("ffprobe no devolvió duración")
    dur = int(float(raw))
    return max(1, dur)


def build_vertical_reel(input_mp4: str, output_mp4: str, seconds: int) -> None:
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_mp4,
        "-t", str(seconds),
        "-vf", vf,
        "-r", "30",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_mp4,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg falló:\nSTDERR:\n{(p.stderr or '')[:4000]}")


def graph_base(cfg: UGCConfig) -> str:
    return f"https://graph.facebook.com/v{cfg.graph_version}"

def ig_create_container(cfg: UGCConfig, video_url: str, caption: str) -> str:
    url = f"{graph_base(cfg)}/{cfg.ig_user_id}/media"
    data = {
        "media_type": "REELS",
        "video_url": video_url,
        "caption": caption,
        "access_token": cfg.ig_access_token,
    }
    r = requests.post(url, data=data, timeout=60)
    r.raise_for_status()
    return r.json()["id"]

def ig_wait_ready(cfg: UGCConfig, creation_id: str, timeout_sec: int = 420) -> None:
    url = f"{graph_base(cfg)}/{creation_id}"
    start = time.time()
    while time.time() - start < timeout_sec:
        r = requests.get(url, params={"fields": "status_code", "access_token": cfg.ig_access_token}, timeout=30)
        r.raise_for_status()
        status = (r.json().get("status_code") or "").upper()
        if status in ("FINISHED", "PUBLISHED"):
            return
        if status in ("ERROR", "FAILED", "EXPIRED"):
            raise RuntimeError(f"IG container status={status}")
        time.sleep(3)
    raise TimeoutError("Timeout esperando IG container")

def ig_publish(cfg: UGCConfig, creation_id: str) -> str:
    url = f"{graph_base(cfg)}/{cfg.ig_user_id}/media_publish"
    data = {"creation_id": creation_id, "access_token": cfg.ig_access_token}
    r = requests.post(url, data=data, timeout=60)
    r.raise_for_status()
    return r.json()["id"]


def default_state() -> Dict[str, Any]:
    return {
        "version": 1,
        "processed": {},
        "failed": {},
        "last_run_at": None,
    }

def already_done(state: Dict[str, Any], inbox_key: str, etag: str) -> bool:
    info = (state.get("processed") or {}).get(inbox_key)
    return bool(info and info.get("etag") == etag)


def run_mode_b() -> None:
    cfg = load_cfg()
    s3 = r2_client(cfg)

    print("===== MODO B (UGC) =====")
    print("Inbox:", cfg.inbox_prefix)
    print("Output reels:", cfg.output_reels_prefix)
    print("State:", cfg.state_key)

    state = get_json_or_default(s3, cfg.bucket_name, cfg.state_key, default_state())

    items = list_mp4(s3, cfg.bucket_name, cfg.inbox_prefix)
    if not items:
        print("[UGC] No hay mp4 en inbox.")
        state["last_run_at"] = int(time.time())
        put_json(s3, cfg.bucket_name, cfg.state_key, state)
        return

    items.sort(key=lambda o: o.get("LastModified"))
    done_this_run = 0

    for obj in items:
        if done_this_run >= cfg.max_per_run:
            break

        inbox_key = obj["Key"]
        etag = (obj.get("ETag") or "").strip('"')
        filename = os.path.basename(inbox_key)

        if already_done(state, inbox_key, etag):
            print("[UGC] Saltando (ya procesado):", inbox_key)
            continue

        print("\n[UGC] Procesando:", inbox_key)

        tmpdir = tempfile.mkdtemp(prefix="ugc_b_")
        try:
            local_in = os.path.join(tmpdir, filename)
            local_out = os.path.join(tmpdir, f"REEL_{filename}")

            s3.download_file(cfg.bucket_name, inbox_key, local_in)

            forced = seconds_from_filename(filename)
            real_dur = probe_video_duration_seconds(local_in)
            target = forced if forced is not None else min(real_dur, cfg.reel_seconds_max)
            target = max(1, min(int(target), 120))

            print(f"[UGC] Duración real={real_dur}s | forced={forced} | target={target}s | cap={cfg.reel_seconds_max}s")

            caption = generate_caption(cfg, filename)
            print("[UGC] Caption:\n", caption)

            build_vertical_reel(local_in, local_out, seconds=target)

            safe_name = filename.replace(" ", "_")
            reel_key = f"{cfg.output_reels_prefix}{int(time.time())}_{safe_name}"
            s3.upload_file(local_out, cfg.bucket_name, reel_key, ExtraArgs={"ContentType": "video/mp4"})
            reel_url = public_url(cfg, reel_key)
            print("[UGC] REEL subido a R2:", reel_url)

            ig_media_id = None
            if cfg.enable_ig_publish:
                if not cfg.ig_user_id or not cfg.ig_access_token:
                    raise RuntimeError("ENABLE_IG_PUBLISH=true pero faltan IG_USER_ID / IG_ACCESS_TOKEN")

                creation_id = ig_create_container(cfg, reel_url, caption)
                print("[UGC] IG container:", creation_id)

                ig_wait_ready(cfg, creation_id)
                ig_media_id = ig_publish(cfg, creation_id)
                print("[UGC] IG PUBLISH OK:", {"id": ig_media_id})
            else:
                print("[UGC] ENABLE_IG_PUBLISH=false → no publico en IG.")

            processed_key = f"{cfg.processed_prefix}{filename}"
            copy_then_delete(s3, cfg.bucket_name, inbox_key, processed_key)
            print("[UGC] Original movido a processed:", processed_key)

            state.setdefault("processed", {})[inbox_key] = {
                "etag": etag,
                "processed_at": int(time.time()),
                "reel_key": reel_key,
                "reel_url": reel_url,
                "ig_media_id": ig_media_id,
                "caption": caption,
                "target_seconds": target,
                "real_seconds": real_dur,
                "forced_seconds": forced,
            }
            state.get("failed", {}).pop(inbox_key, None)

            put_json(s3, cfg.bucket_name, cfg.state_key, state)
            done_this_run += 1

        except Exception as e:
            err = str(e)
            print("[UGC] ERROR:", err)
            try:
                failed_key = f"{cfg.failed_prefix}{filename}"
                copy_then_delete(s3, cfg.bucket_name, inbox_key, failed_key)
                print("[UGC] Original movido a failed:", failed_key)
            except Exception as move_err:
                print("[UGC] No pude mover a failed:", str(move_err))

            state.setdefault("failed", {})[inbox_key] = {
                "etag": etag,
                "failed_at": int(time.time()),
                "error": err[:2000],
            }
            put_json(s3, cfg.bucket_name, cfg.state_key, state)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    state["last_run_at"] = int(time.time())
    put_json(s3, cfg.bucket_name, cfg.state_key, state)
    print("\n[UGC] Listo. Procesados en este run:", done_this_run)


if __name__ == "__main__":
    run_mode_b()
