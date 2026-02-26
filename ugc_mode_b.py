# ugc_mode_b.py
import os
import re
import json
import time
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import boto3
import requests

# OpenAI (opcional: si falla, usa fallback)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# Env helpers
# -----------------------------
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


# -----------------------------
# Config
# -----------------------------
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
    graph_version: str

    # Prefixes / state
    inbox_prefix: str
    processed_prefix: str
    failed_prefix: str
    output_reels_prefix: str
    state_key: str

    # Processing
    cap_seconds: int          # max duration if video is long
    min_seconds: int          # min duration
    default_target_seconds: int  # fallback if cannot detect duration


def load_config() -> UGCConfig:
    r2_public = (env_nonempty("R2_PUBLIC_BASE_URL") or "").rstrip("/")
    if not r2_public:
        raise RuntimeError("Falta R2_PUBLIC_BASE_URL")

    graph_version = (env_nonempty("GRAPH_VERSION", "v25.0") or "v25.0").lstrip("v")

    return UGCConfig(
        r2_endpoint_url=env_nonempty("R2_ENDPOINT_URL") or "",
        aws_access_key_id=env_nonempty("AWS_ACCESS_KEY_ID") or "",
        aws_secret_access_key=env_nonempty("AWS_SECRET_ACCESS_KEY") or "",
        bucket_name=env_nonempty("BUCKET_NAME") or "",
        r2_public_base_url=r2_public,

        openai_api_key=env_nonempty("OPENAI_API_KEY") or "",
        openai_model=env_nonempty("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini",

        enable_ig_publish=env_bool("ENABLE_IG_PUBLISH", True),
        ig_user_id=env_nonempty("IG_USER_ID") or "",
        ig_access_token=env_nonempty("IG_ACCESS_TOKEN") or "",
        graph_version=graph_version,

        inbox_prefix=(env_nonempty("UGC_INBOX_PREFIX", "ugc/inbox/") or "ugc/inbox/").strip(),
        processed_prefix=(env_nonempty("UGC_PROCESSED_PREFIX", "ugc/processed/") or "ugc/processed/").strip(),
        failed_prefix=(env_nonempty("UGC_FAILED_PREFIX", "ugc/failed/") or "ugc/failed/").strip(),
        output_reels_prefix=(env_nonempty("UGC_OUTPUT_REELS_PREFIX", "ugc/outputs/reels/") or "ugc/outputs/reels/").strip(),
        state_key=(env_nonempty("UGC_STATE_KEY", "ugc/state/state.json") or "ugc/state/state.json").strip(),

        cap_seconds=env_int("UGC_CAP_SECONDS", 20),
        min_seconds=env_int("UGC_MIN_SECONDS", 5),
        default_target_seconds=env_int("UGC_DEFAULT_SECONDS", 10),
    )


# -----------------------------
# R2 (S3)
# -----------------------------
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

def r2_list_mp4_keys(cfg: UGCConfig, prefix: str, limit: int = 25) -> List[str]:
    s3 = r2_client(cfg)
    out: List[str] = []
    resp = s3.list_objects_v2(Bucket=cfg.bucket_name, Prefix=prefix)
    for obj in resp.get("Contents", [])[:limit]:
        key = obj.get("Key") or ""
        if key.lower().endswith(".mp4"):
            out.append(key)
    return out

def r2_get_bytes(cfg: UGCConfig, key: str) -> bytes:
    s3 = r2_client(cfg)
    obj = s3.get_object(Bucket=cfg.bucket_name, Key=key)
    return obj["Body"].read()

def r2_put_bytes(cfg: UGCConfig, key: str, data: bytes, content_type: str) -> None:
    s3 = r2_client(cfg)
    s3.put_object(Bucket=cfg.bucket_name, Key=key, Body=data, ContentType=content_type)

def r2_copy(cfg: UGCConfig, src_key: str, dst_key: str) -> None:
    s3 = r2_client(cfg)
    s3.copy_object(
        Bucket=cfg.bucket_name,
        CopySource={"Bucket": cfg.bucket_name, "Key": src_key},
        Key=dst_key,
    )

def r2_delete(cfg: UGCConfig, key: str) -> None:
    s3 = r2_client(cfg)
    s3.delete_object(Bucket=cfg.bucket_name, Key=key)

def r2_public_url(cfg: UGCConfig, key: str) -> str:
    base = cfg.r2_public_base_url.rstrip("/")
    return f"{base}/{key}"

def r2_load_json(cfg: UGCConfig, key: str) -> Optional[Dict[str, Any]]:
    try:
        b = r2_get_bytes(cfg, key)
        return json.loads(b.decode("utf-8"))
    except Exception:
        return None

def r2_save_json(cfg: UGCConfig, key: str, payload: Dict[str, Any]) -> None:
    b = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    r2_put_bytes(cfg, key, b, "application/json")


# -----------------------------
# Caption (OpenAI, fallback)
# -----------------------------
CAPTION_PROMPT = """Eres caster/editor de esports en español LATAM.
Escribe un caption corto para Instagram Reels sobre un clip de gameplay.

Reglas:
- 1 línea fuerte (hook) con vibe esports.
- 1 línea extra de opinión o hype.
- Cierra con pregunta.
- 6-10 hashtags (esports/gaming latam).
No menciones "IA" ni "OpenAI". No pongas links.
"""

def openai_client(cfg: UGCConfig):
    if not OpenAI:
        raise RuntimeError("No se pudo importar OpenAI. Revisa requirements.txt (openai).")
    if not cfg.openai_api_key:
        raise RuntimeError("Falta OPENAI_API_KEY.")
    return OpenAI(api_key=cfg.openai_api_key)

def openai_text(cfg: UGCConfig, prompt: str) -> str:
    client = openai_client(cfg)
    model = cfg.openai_model or "gpt-4.1-mini"
    try:
        resp = client.responses.create(model=model, input=prompt)
        out = getattr(resp, "output_text", None)
        if out:
            return out.strip()
    except Exception:
        pass
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()

def build_caption(cfg: UGCConfig) -> str:
    try:
        if not cfg.openai_api_key:
            raise RuntimeError("no key")
        return openai_text(cfg, CAPTION_PROMPT).strip()
    except Exception:
        return "¡Qué jugada! 🔥 ¿Tú también la intentas o te tilteas?\n#EsportsLATAM #GamingLatam #Gameplay #Gamers #LatamEsports #LevelUp #JuegaFuerte"


# -----------------------------
# Video utils
# -----------------------------
S_FORCED_RE = re.compile(r"__s(\d{1,3})", re.IGNORECASE)

def parse_forced_seconds_from_name(key: str) -> Optional[int]:
    m = S_FORCED_RE.search(key)
    if not m:
        return None
    try:
        v = int(m.group(1))
        return v if v > 0 else None
    except Exception:
        return None

def ffprobe_duration_seconds(path: str) -> Optional[float]:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        if p.returncode != 0:
            return None
        s = (p.stdout or "").strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None

def make_vertical_reel_FAST(input_mp4: str, output_mp4: str, seconds: int) -> None:
    """
    SUPER estable en GitHub Actions:
    - corta a N segundos
    - fuerza vertical 1080x1920
    - crop centrado si el video es horizontal
    - SIN overlays / SIN audio (evita cuelgues)
    """
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", input_mp4,
        "-t", str(int(seconds)),
        "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
        "-r", "30",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        output_mp4
    ]
    # más tiempo para evitar falsos timeouts
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=480, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg falló:\nSTDERR:\n{(p.stderr or '')[:4000]}")


# -----------------------------
# Instagram publish (wait + retry)
# -----------------------------
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

def ig_publish_reel_wait_retry(cfg: UGCConfig, video_url: str, caption: str) -> Dict[str, Any]:
    j = ig_post(cfg, f"{cfg.ig_user_id}/media", {
    "media_type": "REELS",
    "video_url": video_url,
    "caption": caption,
    "share_to_feed": "true",
    "access_token": cfg.ig_access_token
})
    creation_id = j.get("id")
    if not creation_id:
        raise RuntimeError(f"IG create reels container no devolvió id: {j}")

    ig_wait_container(cfg, creation_id, timeout_sec=900)

    last_err = None
    for attempt in range(1, 6):
        try:
            pub = ig_post(cfg, f"{cfg.ig_user_id}/media_publish", {
                "creation_id": creation_id,
                "access_token": cfg.ig_access_token
            })
            return pub
        except Exception as e:
            last_err = e
            time.sleep(8 * attempt)

    raise RuntimeError(f"IG publish falló tras reintentos. Last error: {last_err}")


# -----------------------------
# State
# -----------------------------
def load_state(cfg: UGCConfig) -> Dict[str, Any]:
    st = r2_load_json(cfg, cfg.state_key) or {}
    st.setdefault("processed", {})  # key -> iso time
    return st

def save_state(cfg: UGCConfig, st: Dict[str, Any]) -> None:
    r2_save_json(cfg, cfg.state_key, st)


# -----------------------------
# Mode B runner
# -----------------------------
def run_mode_b() -> None:
    cfg = load_config()

    print("===== MODO B (UGC) =====")
    print(f"Inbox: {cfg.inbox_prefix}")
    print(f"Output reels: {cfg.output_reels_prefix}")
    print(f"State: {cfg.state_key}")

    st = load_state(cfg)

    keys = r2_list_mp4_keys(cfg, cfg.inbox_prefix, limit=25)
    if not keys:
        print("[UGC] No hay mp4 en inbox.")
        return

    processed_this_run = 0

    for key in keys:
        if key in st["processed"]:
            continue

        print(f"\n[UGC] Procesando: {key}")

        forced = parse_forced_seconds_from_name(key)

        try:
            b = r2_get_bytes(cfg, key)

            with tempfile.TemporaryDirectory() as td:
                in_path = os.path.join(td, "in.mp4")
                out_path = os.path.join(td, "out.mp4")

                with open(in_path, "wb") as f:
                    f.write(b)

                dur = ffprobe_duration_seconds(in_path)
                real_sec = int(round(dur)) if dur and dur > 0 else None

                # Decide target duration
                if forced:
                    target = forced
                elif real_sec:
                    target = real_sec
                else:
                    target = cfg.default_target_seconds

                # Apply min/max
                target = max(cfg.min_seconds, min(cfg.cap_seconds, int(target)))

                print(f"[UGC] Duración real={real_sec}s | forced={forced} | target={target}s | cap={cfg.cap_seconds}s")

                caption = build_caption(cfg)
                print("[UGC] Caption:\n", caption)

                # SUPER estable
                make_vertical_reel_FAST(
                    input_mp4=in_path,
                    output_mp4=out_path,
                    seconds=target
                )

                with open(out_path, "rb") as f:
                    out_bytes = f.read()

            # Upload reel output
            out_key = f"{cfg.output_reels_prefix.rstrip('/')}/{os.path.basename(key)}"
            r2_put_bytes(cfg, out_key, out_bytes, "video/mp4")
            video_url = r2_public_url(cfg, out_key)
            print(f"[UGC] REEL subido a R2: {video_url}")

            if cfg.enable_ig_publish:
                if not (cfg.ig_user_id and cfg.ig_access_token):
                    raise RuntimeError("ENABLE_IG_PUBLISH=true pero faltan IG_USER_ID o IG_ACCESS_TOKEN.")
                pub = ig_publish_reel_wait_retry(cfg, video_url, caption)
                print("[UGC] IG PUBLISH OK:", pub)

            # Move original to processed
            dst = f"{cfg.processed_prefix.rstrip('/')}/{os.path.basename(key)}"
            r2_copy(cfg, key, dst)
            r2_delete(cfg, key)

            st["processed"][key] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            save_state(cfg, st)

            processed_this_run += 1

        except Exception as e:
            print("[UGC] ERROR:", str(e))
            # Move original to failed (best-effort)
            try:
                dst = f"{cfg.failed_prefix.rstrip('/')}/{os.path.basename(key)}"
                r2_copy(cfg, key, dst)
                r2_delete(cfg, key)
                print(f"[UGC] Original movido a failed: {dst}")
            except Exception as e2:
                print("[UGC] No se pudo mover a failed:", str(e2))

    print(f"\n[UGC] Listo. Procesados en este run: {processed_this_run}")
