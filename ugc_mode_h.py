import os
import re
import json
import random
import hashlib
import subprocess
import tempfile
from datetime import datetime, timezone

import boto3


def env_nonempty(name, default=None):
    v = os.getenv(name)
    if not v:
        return default
    v = v.strip()
    return v if v else default


def env_int(name, default):
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def env_float(name, default):
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def env_bool(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")
R2_PUBLIC_BASE_URL = (env_nonempty("R2_PUBLIC_BASE_URL", "https://example.r2.dev") or "").rstrip("/")

MODE_H_INPUT_PREFIX = (env_nonempty("MODE_H_INPUT_PREFIX", "ugc/library/clips") or "ugc/library/clips").strip().strip("/")
MODE_H_OUTPUT_PREFIX = (env_nonempty("MODE_H_OUTPUT_PREFIX", "ugc/final_clean") or "ugc/final_clean").strip().strip("/")
MODE_H_META_PREFIX = (env_nonempty("MODE_H_META_PREFIX", "ugc/meta/final_clean") or "ugc/meta/final_clean").strip().strip("/")
MODE_H_STATE_KEY = env_nonempty("MODE_H_STATE_KEY", "ugc/state/mode_h_state.json")

MODE_H_MAX_ITEMS = env_int("MODE_H_MAX_ITEMS", 6)
MODE_H_ONLY_KEYS_CONTAIN = env_nonempty("MODE_H_ONLY_KEYS_CONTAIN", "")
MODE_H_NEWEST_FIRST = env_bool("MODE_H_NEWEST_FIRST", True)

REEL_W = env_int("REEL_W", 1080)
REEL_H = env_int("REEL_H", 1920)
REEL_FPS = env_int("REEL_FPS", 30)

FONT_BOLD = env_nonempty("FONT_BOLD", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

MUSIC_SEARCH_DIR = env_nonempty("MUSIC_SEARCH_DIR", "assets") or "assets"
MUSIC_PROBABILITY = env_float("MUSIC_PROBABILITY", 0.0)
MUSIC_VOLUME = env_float("MUSIC_VOLUME", 0.18)

KEEP_ORIGINAL_AUDIO = env_bool("KEEP_ORIGINAL_AUDIO", True)

ENABLE_HUD = env_bool("ENABLE_HUD", False)
HUD_DIR = env_nonempty("HUD_DIR", "assets") or "assets"
HUD_PREFIX = env_nonempty("HUD_PREFIX", "hud_") or "hud_"
HUD_OPACITY = env_float("HUD_OPACITY", 0.10)

ENABLE_BADGE_TEXT = env_bool("ENABLE_BADGE_TEXT", False)
ENABLE_HOOK_TEXT = env_bool("ENABLE_HOOK_TEXT", False)
ENABLE_CTA_TEXT = env_bool("ENABLE_CTA_TEXT", False)

SAVE_DEBUG_FINAL = env_bool("SAVE_DEBUG_FINAL", True)
DEBUG_FINAL_NAME = env_nonempty("DEBUG_FINAL_NAME", "debug_final_reel.mp4") or "debug_final_reel.mp4"


GAME_BADGES = {
    "valorant": "VALORANT",
    "cs2": "CS2",
    "counter-strike": "CS2",
    "counter strike": "CS2",
    "league of legends": "LOL",
    "fortnite": "FORTNITE",
    "warzone": "WARZONE",
    "apex": "APEX",
    "apex legends": "APEX",
    "minecraft": "MINECRAFT",
    "ea sports fc": "FC",
    "f1": "F1",
    "gran turismo": "GT",
}

GENERIC_HOOKS = [
    "ESTO ES ABSURDO",
    "NAH, MIRA ESTO",
    "NO ES NORMAL",
    "ESTÁ ROTÍSIMO",
    "ESTO NO TIENE SENTIDO",
    "OJO CON ESTA PLAY",
    "ESTO FUE CINE",
    "QUÉ ACABO DE VER",
]

GAME_HOOKS = {
    "valorant": [
        "ESTE CLUTCH ES ILEGAL",
        "ESO FUE UN ACE GRATIS",
        "VALORANT EN SU PEAK",
    ],
    "cs2": [
        "CS2 ENFERMÍSIMO",
        "ESTO ES PURO AIM",
        "CLUTCH DE OTRO PLANETA",
    ],
    "league of legends": [
        "ESTA TEAMFIGHT FUE CINE",
        "LOL EN SU MEJOR MOMENTO",
        "ESTO CAMBIA TODA LA PARTIDA",
    ],
    "fortnite": [
        "EDITS DE OTRO MUNDO",
        "FORTNITE ESTÁ ROTO",
        "BUGHA MODE ACTIVADO",
    ],
    "warzone": [
        "WARZONE ESTÁ LOQUÍSIMO",
        "ESTO NO ERA GANABLE",
        "QUÉ CIERRE TAN SUCIO",
    ],
    "apex legends": [
        "APEX EN MODO BESTIA",
        "FINAL CIRCLE DE LOCOS",
        "ESTO ES PURO MOVEMENT",
    ],
    "minecraft": [
        "MINECRAFT PERO ES CINE",
        "ESTO ES MUY MALA SUERTE",
        "NAH, QUÉ MOMENTO",
    ],
    "ea sports fc": [
        "ESO ES GOLAZO",
        "FC ESTÁ DEMENCIAL",
        "ESTE GOL ES RIDÍCULO",
    ],
    "f1": [
        "ADELANTAMIENTO DE CINE",
        "ESO FUE PURO SKILL",
        "F1 EN MODO BESTIA",
    ],
    "gran turismo": [
        "ESTA ÚLTIMA VUELTA FUE CINE",
        "GT ESTÁ HERMOSO",
        "QUÉ FINAL TAN LIMPIO",
    ],
}

CTAS = [
    "¿TOP O HUMO?",
    "¿SKILL O SUERTE?",
    "¿TÚ LO SACABAS?",
    "¿W O BASURA?",
    "¿ESTO ES CINE O NO?",
    "¿MEJOR JUGADA O NO?",
    "¿SE LA COMEN O NO?",
    "¿ESTO ES LEGAL?",
]


def now_utc():
    return datetime.now(timezone.utc)


def iso_now_full():
    return now_utc().isoformat()


def safe_slug(s):
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9\-_]+", "_", s)[:140]
    return s.strip("_") or "clip"


def short_hash_bytes(data):
    return hashlib.sha1(data).hexdigest()[:10]


def safe_json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def list_mp3_files(search_dir):
    files = []
    if search_dir and os.path.isdir(search_dir):
        for root, _, fnames in os.walk(search_dir):
            for f in fnames:
                if f.lower().endswith(".mp3"):
                    files.append(os.path.join(root, f))
    return files


def pick_music():
    if random.random() > max(0.0, min(1.0, MUSIC_PROBABILITY)):
        return None

    candidates = list_mp3_files(MUSIC_SEARCH_DIR)
    good = []

    for c in candidates:
        try:
            if os.path.getsize(c) > 50_000:
                good.append(c)
        except Exception:
            pass

    if not good:
        return None

    return random.choice(good)


def pick_hud_overlay():
    if not ENABLE_HUD:
        return None

    if not os.path.isdir(HUD_DIR):
        return None

    files = []
    for f in os.listdir(HUD_DIR):
        name = f.lower()

        if not name.startswith(HUD_PREFIX.lower()):
            continue
        if not name.endswith(".png"):
            continue
        if (
            "safearea" in name
            or "guide" in name
            or "guides" in name
            or "template" in name
            or "layout" in name
            or "grid" in name
        ):
            continue

        files.append(os.path.join(HUD_DIR, f))

    if not files:
        return None

    return random.choice(files)


def wrap_text(text, max_chars_per_line=18, max_lines=2):
    t = (text or "").strip().replace("\n", " ")
    words = t.split()
    if not words:
        return ""

    lines = []
    cur = ""

    for w in words:
        if not cur:
            cur = w
            continue

        if len(cur) + 1 + len(w) <= max_chars_per_line:
            cur = cur + " " + w
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break

    if len(lines) < max_lines and cur:
        lines.append(cur)

    if lines:
        lines[-1] = lines[-1][:max_chars_per_line].rstrip()

    return "\n".join(lines).strip()


def ffprobe_json(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {}
    try:
        return json.loads(p.stdout or "{}")
    except Exception:
        return {}


def get_video_duration(path):
    info = ffprobe_json(path)
    try:
        return float(info.get("format", {}).get("duration", 0.0) or 0.0)
    except Exception:
        return 0.0


def game_from_key(key):
    k = (key or "").lower()
    normalized = k.replace("_", " ").replace("-", " ")

    checks = [
        "ea sports fc",
        "gran turismo",
        "league of legends",
        "counter strike",
        "counter-strike",
        "valorant",
        "fortnite",
        "warzone",
        "apex legends",
        "minecraft",
        "f1",
        "cs2",
        "apex",
    ]

    for c in checks:
        if c in normalized:
            return c

    return "generic"


def pick_hook(game_name):
    g = (game_name or "").lower()
    options = GAME_HOOKS.get(g) or GENERIC_HOOKS
    return random.choice(options)


def pick_cta():
    return random.choice(CTAS)


def pick_badge(game_name):
    g = (game_name or "").lower()

    mapping = {
        "counter strike": "CS2",
        "counter-strike": "CS2",
        "ea sports fc": "FC",
        "gran turismo": "GT",
        "league of legends": "LOL",
        "valorant": "VALORANT",
        "fortnite": "FORTNITE",
        "warzone": "WARZONE",
        "apex legends": "APEX",
        "apex": "APEX",
        "minecraft": "MINECRAFT",
        "f1": "F1",
        "cs2": "CS2",
    }

    return mapping.get(g, "GAMER")


def r2_client():
    if not (R2_ENDPOINT_URL and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and BUCKET_NAME):
        raise RuntimeError("Faltan credenciales R2/S3")

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def s3_put_bytes(key, data, content_type="application/octet-stream"):
    s3 = r2_client()
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=data, ContentType=content_type)

    if R2_PUBLIC_BASE_URL.startswith("http"):
        return f"{R2_PUBLIC_BASE_URL}/{key}"

    return key


def s3_put_json(key, payload):
    s3_put_bytes(key, safe_json_dumps(payload), "application/json")


def s3_get_json(key):
    try:
        obj = r2_client().get_object(Bucket=BUCKET_NAME, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None


def r2_list_objects(prefix):
    s3 = r2_client()
    items = []
    continuation_token = None

    while True:
        kwargs = {
            "Bucket": BUCKET_NAME,
            "Prefix": prefix,
            "MaxKeys": 200,
        }

        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        resp = s3.list_objects_v2(**kwargs)

        for obj in resp.get("Contents", []):
            k = obj["Key"]
            if not k.endswith("/"):
                items.append(
                    {
                        "key": k,
                        "last_modified": obj.get("LastModified"),
                    }
                )

        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break

    return items


def r2_download_to_file(key, dst_path):
    r2_client().download_file(BUCKET_NAME, key, dst_path)


def load_state():
    st = s3_get_json(MODE_H_STATE_KEY)
    if not st:
        st = {"processed_keys": [], "last_run_at": None}

    if "processed_keys" not in st or not isinstance(st["processed_keys"], list):
        st["processed_keys"] = []

    if "last_run_at" not in st:
        st["last_run_at"] = None

    return st


def save_state(st):
    st["last_run_at"] = iso_now_full()
    s3_put_json(MODE_H_STATE_KEY, st)


def build_hype_reel(
    input_video,
    output_video,
    hook_text,
    badge_text,
    cta_text,
    music_mp3,
    hud_png,
):
    if not FONT_BOLD or not os.path.exists(FONT_BOLD):
        raise RuntimeError(f"No existe FONT_BOLD: {FONT_BOLD}")

    hook_wrapped = wrap_text(hook_text.upper(), max_chars_per_line=14, max_lines=2)
    cta_wrapped = wrap_text(cta_text.upper(), max_chars_per_line=18, max_lines=1)
    badge_wrapped = wrap_text(badge_text.upper(), max_chars_per_line=12, max_lines=1)

    with tempfile.TemporaryDirectory() as td:
        hook_txt = os.path.join(td, "hook.txt")
        cta_txt = os.path.join(td, "cta.txt")
        badge_txt = os.path.join(td, "badge.txt")

        with open(hook_txt, "w", encoding="utf-8") as f:
            f.write(hook_wrapped)

        with open(cta_txt, "w", encoding="utf-8") as f:
            f.write(cta_wrapped)

        with open(badge_txt, "w", encoding="utf-8") as f:
            f.write(badge_wrapped)

        duration = max(3.0, get_video_duration(input_video))

        cmd = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_video,
        ]

        hud_input_idx = None
        if hud_png and os.path.exists(hud_png) and ENABLE_HUD:
            cmd += ["-i", hud_png]
            hud_input_idx = 1

        music_input_idx = None
        if music_mp3 and os.path.exists(music_mp3) and MUSIC_PROBABILITY > 0:
            cmd += ["-i", music_mp3]
            music_input_idx = 2 if hud_input_idx is not None else 1

        vf_parts = []

        vf_parts.append(
            f"[0:v]"
            f"fps={REEL_FPS},"
            f"scale={REEL_W}:{REEL_H}:force_original_aspect_ratio=increase,"
            f"crop={REEL_W}:{REEL_H},"
            f"setsar=1,"
            f"format=rgba[v0];"
        )

        current = "[v0]"

        if hud_input_idx is not None:
            vf_parts.append(
                f"[{hud_input_idx}:v]"
                f"scale={REEL_W}:{REEL_H},"
                f"format=rgba,"
                f"colorchannelmixer=aa={max(0.0, min(1.0, HUD_OPACITY))}[hud];"
            )
            vf_parts.append(f"{current}[hud]overlay=0:0:format=auto[v1];")
            current = "[v1]"

        if ENABLE_BADGE_TEXT:
            vf_parts.append(
                f"{current}"
                f"drawbox=x=60:y=70:w=250:h=78:color=white@0.82:t=fill,"
                f"drawtext=fontfile={FONT_BOLD}:textfile={badge_txt}:"
                f"x=94:y=90:fontsize=38:fontcolor=black"
                f"[vbadge];"
            )
            current = "[vbadge]"

        if ENABLE_HOOK_TEXT:
            vf_parts.append(
                f"{current}"
                f"drawtext=fontfile={FONT_BOLD}:textfile={hook_txt}:"
                f"x=64:y=140:"
                f"fontsize=64:"
                f"line_spacing=8:"
                f"fontcolor=white:"
                f"borderw=2:bordercolor=black@0.55:"
                f"box=1:boxcolor=black@0.18:boxborderw=12"
                f"[vhook];"
            )
            current = "[vhook]"

        if ENABLE_CTA_TEXT:
            vf_parts.append(
                f"{current}"
                f"drawtext=fontfile={FONT_BOLD}:textfile={cta_txt}:"
                f"x=64:y=1660:"
                f"fontsize=42:"
                f"fontcolor=white:"
                f"borderw=2:bordercolor=black@0.45:"
                f"box=1:boxcolor=black@0.16:boxborderw=10"
                f"[vcta]"
            )
            current = "[vcta]"
        else:
            vf_parts.append(f"{current}format=yuv420p[vcta]")
            current = "[vcta]"

        cmd += [
            "-filter_complex",
            "".join(vf_parts),
            "-map",
            current,
        ]

        if KEEP_ORIGINAL_AUDIO:
            cmd += [
                "-map",
                "0:a?",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
            ]
        elif music_input_idx is not None:
            fade_out_start = max(0.0, duration - 0.9)
            cmd += [
                "-map",
                f"{music_input_idx}:a",
                "-filter:a",
                f"volume={MUSIC_VOLUME},afade=t=in:st=0:d=0.35,afade=t=out:st={fade_out_start}:d=0.8",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
            ]
        else:
            cmd += ["-an"]

        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_video,
        ]

        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg falló:\n{(p.stderr or '')[:4000]}")


def run_mode_h():
    print("===== MODE H (HYPE PACKER) START =====")
    print("MODE_H_INPUT_PREFIX:", MODE_H_INPUT_PREFIX)
    print("MODE_H_OUTPUT_PREFIX:", MODE_H_OUTPUT_PREFIX)
    print("MODE_H_META_PREFIX:", MODE_H_META_PREFIX)
    print("MODE_H_STATE_KEY:", MODE_H_STATE_KEY)
    print("MODE_H_MAX_ITEMS:", MODE_H_MAX_ITEMS)
    print("MODE_H_ONLY_KEYS_CONTAIN:", MODE_H_ONLY_KEYS_CONTAIN or "(vacío)")
    print("MODE_H_NEWEST_FIRST:", MODE_H_NEWEST_FIRST)

    print("KEEP_ORIGINAL_AUDIO:", KEEP_ORIGINAL_AUDIO)
    print("MUSIC_PROBABILITY:", MUSIC_PROBABILITY)
    print("ENABLE_HUD:", ENABLE_HUD)
    print("ENABLE_BADGE_TEXT:", ENABLE_BADGE_TEXT)
    print("ENABLE_HOOK_TEXT:", ENABLE_HOOK_TEXT)
    print("ENABLE_CTA_TEXT:", ENABLE_CTA_TEXT)

    state = load_state()
    processed = set(state.get("processed_keys", []))

    source_items = r2_list_objects(f"{MODE_H_INPUT_PREFIX}/")
    clip_items = [x for x in source_items if x["key"].lower().endswith(".mp4")]

    clip_items.sort(
        key=lambda x: x.get("last_modified") or 0,
        reverse=MODE_H_NEWEST_FIRST,
    )

    clip_keys = [x["key"] for x in clip_items]

    print("Clips encontrados:", len(clip_keys))
    print("Clips procesados en state:", len(processed))

    if MODE_H_ONLY_KEYS_CONTAIN:
        clip_keys = [k for k in clip_keys if MODE_H_ONLY_KEYS_CONTAIN in k]
        print("Clips tras filtro:", len(clip_keys))

    processed_count = 0

    for key in clip_keys:
        if processed_count >= MODE_H_MAX_ITEMS:
            break

        if key in processed:
            print("SKIP already processed:", key)
            continue

        print("Empacando:", key)

        game_name = game_from_key(key)
        hook = pick_hook(game_name)
        cta = pick_cta()
        badge = pick_badge(game_name)
        music = pick_music()
        hud = pick_hud_overlay()

        print("GAME:", game_name)
        print("HOOK:", hook)
        print("CTA:", cta)
        print("BADGE:", badge)
        print("MUSIC:", music if music else "NONE")
        print("HUD:", hud if hud else "NONE")

        try:
            with tempfile.TemporaryDirectory() as td:
                in_path = os.path.join(td, "in.mp4")
                out_path = os.path.join(td, "out.mp4")

                r2_download_to_file(key, in_path)

                build_hype_reel(
                    input_video=in_path,
                    output_video=out_path,
                    hook_text=hook,
                    badge_text=badge,
                    cta_text=cta,
                    music_mp3=music,
                    hud_png=hud,
                )

                with open(out_path, "rb") as f:
                    data = f.read()

                h = short_hash_bytes(data)
                base = os.path.basename(key).rsplit(".", 1)[0]
                out_key = f"{MODE_H_OUTPUT_PREFIX}/{base}__hype__{h}.mp4"

                print("Uploading final reel:", out_key)
                s3_put_bytes(out_key, data, "video/mp4")

                meta = {
                    "source_clip_key": key,
                    "final_key": out_key,
                    "game_name": game_name,
                    "hook": hook,
                    "cta": cta,
                    "badge": badge,
                    "music": music,
                    "hud": hud,
                    "generated_at": iso_now_full(),
                }

                meta_key = f"{MODE_H_META_PREFIX}/{os.path.basename(out_key).rsplit('.', 1)[0]}.json"
                s3_put_json(meta_key, meta)
                print("Saved final meta:", meta_key)

                if SAVE_DEBUG_FINAL:
                    with open(DEBUG_FINAL_NAME, "wb") as f:
                        f.write(data)
                    print("Saved debug final:", DEBUG_FINAL_NAME)

            processed.add(key)
            processed_count += 1

        except Exception as e:
            print("ERROR empacando:", key, repr(e))

    state["processed_keys"] = list(processed)[-5000:]
    save_state(state)

    print("===== MODE H DONE =====")
    print("Final reels creados:", processed_count)


if __name__ == "__main__":
    run_mode_h()
