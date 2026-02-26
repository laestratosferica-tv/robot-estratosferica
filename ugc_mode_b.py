# ugc_mode_b.py
import os
import re
import json
import time
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import boto3
import requests

# OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -------------------------
# Env helpers
# -------------------------

def env_nonempty(name: str, default: str = "") -> str:
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

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


# -------------------------
# Config
# -------------------------

@dataclass
class UGCConfig:
    # R2
    r2_endpoint_url: str
    aws_access_key_id: str
    aws_secret_access_key: str
    bucket_name: str
    r2_public_base_url: str

    # OpenAI
    openai_api_key: str
    openai_model: str

    # Instagram
    enable_ig_publish: bool
    ig_user_id: str
    ig_access_token: str
    graph_version: str  # "25.0"

    # Prefixes
    inbox_prefix: str
    processed_prefix: str
    failed_prefix: str
    output_reels_prefix: str
    state_key: str

    # Video behavior
    cap_seconds: int          # max duración si el video es muy largo
    default_min_seconds: int  # si video dura 2s, igual lo dejamos 2s (no lo alargamos)
    ffmpeg_timeout_sec: int


def load_cfg() -> UGCConfig:
    graph_version = env_nonempty("GRAPH_VERSION", "v25.0").lstrip("v")

    return UGCConfig(
        r2_endpoint_url=env_nonempty("R2_ENDPOINT_URL"),
        aws_access_key_id=env_nonempty("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=env_nonempty("AWS_SECRET_ACCESS_KEY"),
        bucket_name=env_nonempty("BUCKET_NAME"),
        r2_public_base_url=env_nonempty("R2_PUBLIC_BASE_URL").rstrip("/"),

        openai_api_key=env_nonempty("OPENAI_API_KEY"),
        openai_model=env_nonempty("OPENAI_MODEL", "gpt-4.1-mini"),

        enable_ig_publish=env_bool("ENABLE_IG_PUBLISH", False),
        ig_user_id=env_nonempty("IG_USER_ID"),
        ig_access_token=env_nonempty("IG_ACCESS_TOKEN"),
        graph_version=graph_version,

        inbox_prefix=env_nonempty("UGC_INBOX_PREFIX", "ugc/inbox/").strip().lstrip("/"),
        processed_prefix=env_nonempty("UGC_PROCESSED_PREFIX", "ugc/processed/").strip().lstrip("/"),
        failed_prefix=env_nonempty("UGC_FAILED_PREFIX", "ugc/failed/").strip().lstrip("/"),
        output_reels_prefix=env_nonempty("UGC_OUTPUT_REELS_PREFIX", "ugc/outputs/reels/").strip().lstrip("/"),
        state_key=env_nonempty("UGC_STATE_KEY", "ugc/state/state.json").strip().lstrip("/"),

        cap_seconds=env_int("UGC_CAP_SECONDS", 20),
        default_min_seconds=env_int("UGC_MIN_SECONDS", 5),
        ffmpeg_timeout_sec=env_int("UGC_FFMPEG_TIMEOUT", 600),
    )


# -------------------------
# R2 helpers
# -------------------------

def r2_client(cfg: UGCConfig):
    if not (cfg.r2_endpoint_url and cfg.aws_access_key_id and cfg.aws_secret_access_key and cfg.bucket_name):
        raise RuntimeError("Faltan credenciales R2/S3 (R2_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME)")
    return boto3.client(
        "s3",
        endpoint_url=cfg.r2_endpoint_url,
        aws_access_key_id=cfg.aws_access_key_id,
        aws_secret_access_key=cfg.aws_secret_access_key,
        region_name="auto",
    )

def r2_list_mp4(cfg: UGCConfig, prefix: str) -> List[str]:
    s3 = r2_client(cfg)
    out: List[str] = []
    token = None
    while True:
        kwargs = {"Bucket": cfg.bucket_name, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []) or []:
            key = it.get("Key") or ""
            if key.lower().endswith(".mp4"):
                out.append(key)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return out

def r2_get_bytes(cfg: UGCConfig, key: str) -> bytes:
    s3 = r2_client(cfg)
    obj = s3.get_object(Bucket=cfg.bucket_name, Key=key)
    return obj["Body"].read()

def r2_put_bytes(cfg: UGCConfig, key: str, data: bytes, content_type: str) -> str:
    s3 = r2_client(cfg)
    s3.put_object(Bucket=cfg.bucket_name, Key=key, Body=data, ContentType=content_type)
    return f"{cfg.r2_public_base_url}/{key}"

def r2_copy_delete(cfg: UGCConfig, src_key: str, dst_key: str) -> None:
    s3 = r2_client(cfg)
    s3.copy_object(
        Bucket=cfg.bucket_name,
        CopySource={"Bucket": cfg.bucket_name, "Key": src_key},
        Key=dst_key,
    )
    s3.delete_object(Bucket=cfg.bucket_name, Key=src_key)

def load_state(cfg: UGCConfig) -> Dict[str, Any]:
    s3 = r2_client(cfg)
    try:
        obj = s3.get_object(Bucket=cfg.bucket_name, Key=cfg.state_key)
        data = obj["Body"].read().decode("utf-8")
        j = json.loads(data)
        if not isinstance(j, dict):
            raise ValueError("state no es dict")
    except Exception:
        j = {}
    j.setdefault("processed", {})
    return j

def save_state(cfg: UGCConfig, state: Dict[str, Any]) -> None:
    s3 = r2_client(cfg)
    body = json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8")
    s3.put_object(Bucket=cfg.bucket_name, Key=cfg.state_key, Body=body, ContentType="application/json")


# -------------------------
# OpenAI caption
# -------------------------

def openai_client(cfg: UGCConfig):
    if not OpenAI:
        raise RuntimeError("No se pudo importar OpenAI. Revisa requirements.txt (openai).")
    if not cfg.openai_api_key:
        raise RuntimeError("Falta OPENAI_API_KEY.")
    return OpenAI(api_key=cfg.openai_api_key)

def openai_text(cfg: UGCConfig, prompt: str) -> str:
    model = cfg.openai_model or "gpt-4.1-mini"
    client = openai_client(cfg)
    try:
        resp = client.responses.create(model=model, input=prompt)
        out = getattr(resp, "output_text", None)
        if out:
            return out.strip()
    except Exception:
        pass
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return (resp.choices[0].message.content or "").strip()

CAPTION_PROMPT = """Eres comentarista esports LATAM.
Te doy el nombre del clip y su duración.
Escribe un caption para Instagram REEL:
- 2-3 líneas cortas
- 5-10 hashtags
- termina con una pregunta
Nombre del clip: {name}
Duración (segundos): {sec}
"""


# -------------------------
# Video tools (ffprobe / ffmpeg)
# -------------------------

FORCE_SEC_RE = re.compile(r"__s(\d{1,3})", re.IGNORECASE)

def parse_forced_seconds(filename: str) -> Optional[int]:
    m = FORCE_SEC_RE.search(filename or "")
    if not m:
        return None
    try:
        v = int(m.group(1))
        if 1 <= v <= 180:
            return v
    except Exception:
        return None
    return None

def ffprobe_duration_seconds(path: str) -> Optional[float]:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            path
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        if p.returncode != 0:
            return None
        s = (p.stdout or "").strip()
        return float(s)
    except Exception:
        return None

def ffmpeg_make_vertical_reel(
    in_path: str,
    out_path: str,
    seconds: int,
    timeout_sec: int
) -> None:
    # filtro: rellena a 1080x1920, recorta centrado (sirve para landscape o cualquier aspect)
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", "0",
        "-t", str(int(seconds)),
        "-i", in_path,
        "-vf", vf,
        "-r", "30",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        out_path
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg falló:\nSTDERR:\n{(p.stderr or '')[:4000]}")


# -------------------------
# Instagram Graph API (REELS) - FIX
# -------------------------

def graph_base(cfg: UGCConfig) -> str:
    return f"https://graph.facebook.com/v{cfg.graph_version}"

def ig_post(cfg: UGCConfig, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{graph_base(cfg)}/{path.lstrip('/')}"
    r = requests.post(url, data=data, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"IG POST error {r.status_code} url={url} resp={r.text}")
    return r.json()

def ig_get(cfg: UGCConfig, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{graph_base(cfg)}/{path.lstrip('/')}"
    r = requests.get(url, params=params, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"IG GET error {r.status_code} url={url} resp={r.text}")
    return r.json()

def ig_wait_container(cfg: UGCConfig, creation_id: str, timeout_sec: int = 900) -> None:
    start = time.time()
    last = None
    while time.time() - start < timeout_sec:
        j = ig_get(cfg, creation_id, {"fields": "status_code", "access_token": cfg.ig_access_token})
        last = j
        status = (j.get("status_code") or "").upper()
        if status in ("FINISHED", "PUBLISHED"):
            return
        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"IG container failed: {j}")
        time.sleep(5)
    raise RuntimeError(f"IG container not ready after {timeout_sec}s. last={last}")

def ig_publish_reel(cfg: UGCConfig, video_url: str, caption: str) -> Dict[str, Any]:
    # 1) crear container REELS
    create = ig_post(cfg, f"{cfg.ig_user_id}/media", {
        "media_type": "REELS",
        "video_url": video_url,
        "caption": caption,
        "share_to_feed": "true",
        "access_token": cfg.ig_access_token,
    })
    creation_id = create.get("id")
    if not creation_id:
        raise RuntimeError(f"IG reels create no devolvió id: {create}")

    # 2) esperar a FINISHED
    ig_wait_container(cfg, creation_id, timeout_sec=900)

    # 3) publicar
    pub = ig_post(cfg, f"{cfg.ig_user_id}/media_publish", {
        "creation_id": creation_id,
        "access_token": cfg.ig_access_token,
    })
    return {"create": create, "publish": pub, "creation_id": creation_id}


# -------------------------
# Main mode B
# -------------------------

def run_mode_b() -> None:
    cfg = load_cfg()

    print("===== MODO B (UGC) =====")
    print("Inbox:", cfg.inbox_prefix)
    print("Output reels:", cfg.output_reels_prefix)
    print("State:", cfg.state_key)

    state = load_state(cfg)
    processed_map: Dict[str, Any] = state.get("processed", {}) or {}

    keys = r2_list_mp4(cfg, cfg.inbox_prefix)
    if not keys:
        print("[UGC] No hay mp4 en inbox.")
        return

    did = 0
    for key in sorted(keys):
        if key in processed_map:
            continue

        fname = key.split("/")[-1]
        print(f"\n[UGC] Procesando: {key}")

        try:
            # descargar
            mp4_bytes = r2_get_bytes(cfg, key)

            with tempfile.TemporaryDirectory() as td:
                in_path = os.path.join(td, "in.mp4")
                out_path = os.path.join(td, "out.mp4")
                with open(in_path, "wb") as f:
                    f.write(mp4_bytes)

                # duración real
                real = ffprobe_duration_seconds(in_path) or 0.0
                forced = parse_forced_seconds(fname)

                # target
                if forced is not None:
                    target = forced
                else:
                    if real <= 0.01:
                        target = cfg.default_min_seconds
                    else:
                        target = int(max(1, min(real, cfg.cap_seconds)))

                print(f"[UGC] Duración real={int(real)}s | forced={forced} | target={target}s | cap={cfg.cap_seconds}s")

                # caption IA
                cap_prompt = CAPTION_PROMPT.format(name=fname, sec=target)
                caption = openai_text(cfg, cap_prompt).strip()
                if not caption:
                    caption = "🔥 Clip esports LATAM. ¿Qué opinas?"
                print("[UGC] Caption:\n", caption)

                # convertir a vertical reel mp4
                ffmpeg_make_vertical_reel(in_path, out_path, target, timeout_sec=cfg.ffmpeg_timeout_sec)
                with open(out_path, "rb") as f:
                    out_bytes = f.read()

            # subir reel a R2 outputs
            out_key = f"{cfg.output_reels_prefix}{now_ts()}_{fname}".replace("//", "/")
            reel_url = r2_put_bytes(cfg, out_key, out_bytes, "video/mp4")
            print("[UGC] REEL subido a R2:", reel_url)

            # publicar IG
            ig_res = None
            if cfg.enable_ig_publish:
                if not (cfg.ig_user_id and cfg.ig_access_token):
                    raise RuntimeError("ENABLE_IG_PUBLISH=true pero faltan IG_USER_ID o IG_ACCESS_TOKEN.")
                ig_res = ig_publish_reel(cfg, reel_url, caption)
                print("[UGC] IG PUBLISH OK:", ig_res.get("publish"))

            # mover original a processed
            dst = f"{cfg.processed_prefix}{fname}".replace("//", "/")
            r2_copy_delete(cfg, key, dst)
            print("[UGC] Original movido a processed:", dst)

            # state
            processed_map[key] = {
                "processed_at": now_ts(),
                "original_key": key,
                "processed_key": dst,
                "reel_key": out_key,
                "reel_url": reel_url,
                "duration_target": target,
                "forced": forced,
                "ig": ig_res,
            }
            state["processed"] = processed_map
            save_state(cfg, state)
            did += 1

        except Exception as e:
            print("[UGC] ERROR:", str(e))
            try:
                # mover a failed
                dst = f"{cfg.failed_prefix}{fname}".replace("//", "/")
                r2_copy_delete(cfg, key, dst)
                print("[UGC] Original movido a failed:", dst)
            except Exception as e2:
                print("[UGC] ERROR moviendo a failed:", str(e2))

    print(f"\n[UGC] Listo. Procesados en este run: {did}")
