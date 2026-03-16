# ===== ugc_mode_b.py =====

import os
import re
import json
import math
import tempfile
import random
from datetime import datetime, timezone

import requests
import boto3


def env_nonempty(name, default=None):
    v = os.getenv(name)
    if not v:
        return default
    v = v.strip()
    return v if v else default


def env_bool(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


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


AWS_ACCESS_KEY_ID = env_nonempty("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_nonempty("AWS_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = env_nonempty("R2_ENDPOINT_URL")
BUCKET_NAME = env_nonempty("BUCKET_NAME")
R2_PUBLIC_BASE_URL = (env_nonempty("R2_PUBLIC_BASE_URL", "") or "").rstrip("/")

PREFIX_PRIORITY = (
    env_nonempty("B_PREFIX_PRIORITY", "ugc/final_priority") or "ugc/final_priority"
).strip().strip("/")
PREFIX_MANUAL = (
    env_nonempty("B_PREFIX_MANUAL", "ugc/final_manual") or "ugc/final_manual"
).strip().strip("/")
PREFIX_AUTO = (
    env_nonempty("B_PREFIX_AUTO", "ugc/final_clean") or "ugc/final_clean"
).strip().strip("/")

META_FINAL_PREFIX = (
    env_nonempty("B_META_FINAL_PREFIX", "ugc/meta/final_clean") or "ugc/meta/final_clean"
).strip().strip("/")

STATE_KEY = env_nonempty("B_STATE_KEY", "ugc/state/mode_b_state.json")

B_MAX_PUBLISH_PER_RUN = env_int("B_MAX_PUBLISH_PER_RUN", 2)
B_MAX_PER_SOURCE_GROUP_PER_RUN = env_int("B_MAX_PER_SOURCE_GROUP_PER_RUN", 2)

B_ONLY_KEYS_CONTAIN = env_nonempty("B_ONLY_KEYS_CONTAIN", "")
B_AVOID_SAME_SOURCE_PER_RUN = env_bool("B_AVOID_SAME_SOURCE_PER_RUN", True)
B_BLOCK_IF_SOURCE_ALREADY_PUBLISHED = env_bool("B_BLOCK_IF_SOURCE_ALREADY_PUBLISHED", False)

B_MIN_CANDIDATE_SCORE = env_float("B_MIN_CANDIDATE_SCORE", 0.55)
B_ALLOW_LOW_SCORE_FOR_PRIORITY = env_bool("B_ALLOW_LOW_SCORE_FOR_PRIORITY", True)
B_ALLOW_LOW_SCORE_FOR_MANUAL = env_bool("B_ALLOW_LOW_SCORE_FOR_MANUAL", True)

B_BLOCK_WEAK_GENERIC_AUTO = env_bool("B_BLOCK_WEAK_GENERIC_AUTO", True)
B_REQUIRE_EDITORIAL_SIGNALS = env_bool("B_REQUIRE_EDITORIAL_SIGNALS", True)

B_EDITORIAL_SCORE_MIN = env_float("B_EDITORIAL_SCORE_MIN", 1.25)
B_ALLOW_STRONG_HOOK_OVERRIDE = env_bool("B_ALLOW_STRONG_HOOK_OVERRIDE", True)

B_CAPTION_MAX_WORDS = env_int("B_CAPTION_MAX_WORDS", 42)
B_CAPTION_MIN_SCORE = env_float("B_CAPTION_MIN_SCORE", 3.6)
B_USE_OPENAI_POLISH = env_bool("B_USE_OPENAI_POLISH", True)
B_OPENAI_POLISH_MIN_EDITORIAL = env_float("B_OPENAI_POLISH_MIN_EDITORIAL", 3.0)

EDITORIAL_MEMORY_SUMMARY_KEY = env_nonempty(
    "B_EDITORIAL_MEMORY_SUMMARY_KEY",
    "ugc/state/editorial_memory_summary.json",
)

ENABLE_INSTAGRAM = env_bool("ENABLE_INSTAGRAM", True)
ENABLE_FACEBOOK = env_bool("ENABLE_FACEBOOK", True)
ENABLE_TIKTOK = env_bool("ENABLE_TIKTOK", False)
ENABLE_SHORTS = env_bool("ENABLE_SHORTS", False)

DRY_RUN = env_bool("DRY_RUN", False)

GRAPH_VERSION = (env_nonempty("GRAPH_VERSION", "v25.0") or "v25.0").lstrip("v")
GRAPH_BASE = f"https://graph.facebook.com/v{GRAPH_VERSION}"

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)

IG_USER_ID = env_nonempty("IG_USER_ID")
IG_ACCESS_TOKEN = env_nonempty("IG_ACCESS_TOKEN")

FB_PAGE_ID = env_nonempty("FB_PAGE_ID")
FB_PAGE_ACCESS_TOKEN = env_nonempty("FB_PAGE_ACCESS_TOKEN")

YOUTUBE_CLIENT_ID = env_nonempty("YOUTUBE_CLIENT_ID")
YOUTUBE_CLIENT_SECRET = env_nonempty("YOUTUBE_CLIENT_SECRET")
YOUTUBE_REFRESH_TOKEN = env_nonempty("YOUTUBE_REFRESH_TOKEN")
YOUTUBE_PRIVACY_STATUS = env_nonempty("YOUTUBE_PRIVACY_STATUS", "public")

OPENAI_API_KEY = env_nonempty("OPENAI_API_KEY")
OPENAI_MODEL = env_nonempty("OPENAI_MODEL", "gpt-4.1-mini")


def now_utc():
    return datetime.now(timezone.utc)


def iso_now_full():
    return now_utc().isoformat()


def r2():
    if not AWS_ACCESS_KEY_ID:
        raise RuntimeError("Falta AWS_ACCESS_KEY_ID")
    if not AWS_SECRET_ACCESS_KEY:
        raise RuntimeError("Falta AWS_SECRET_ACCESS_KEY")
    if not R2_ENDPOINT_URL:
        raise RuntimeError("Falta R2_ENDPOINT_URL")
    if not BUCKET_NAME:
        raise RuntimeError("Falta BUCKET_NAME")

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def list_keys(prefix):
    s3 = r2()
    items = []
    token = None

    while True:
        params = {
            "Bucket": BUCKET_NAME,
            "Prefix": f"{prefix}/",
            "MaxKeys": 200,
        }

        if token:
            params["ContinuationToken"] = token

        resp = s3.list_objects_v2(**params)

        for obj in resp.get("Contents", []):
            k = obj["Key"]
            if not k.endswith("/") and k.lower().endswith(".mp4"):
                items.append(
                    {
                        "key": k,
                        "last_modified": obj.get("LastModified"),
                    }
                )

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    items.sort(key=lambda x: x["last_modified"] or 0, reverse=True)
    return [x["key"] for x in items]


def load_json(key):
    try:
        obj = r2().get_object(Bucket=BUCKET_NAME, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None


def load_text(key):
    try:
        obj = r2().get_object(Bucket=BUCKET_NAME, Key=key)
        return obj["Body"].read().decode("utf-8", errors="replace")
    except Exception:
        return None


def load_editorial_memory_summary():
    data = load_json(EDITORIAL_MEMORY_SUMMARY_KEY)
    return data if isinstance(data, dict) else {}


def get_editorial_game_memory(summary, game_name):
    if not isinstance(summary, dict):
        return {}

    games = summary.get("games") or {}
    if not isinstance(games, dict):
        return {}

    key = str(game_name or "").strip().lower()
    value = games.get(key)
    return value if isinstance(value, dict) else {}


def load_state():
    try:
        obj = r2().get_object(Bucket=BUCKET_NAME, Key=STATE_KEY)
        st = json.loads(obj["Body"].read())
    except Exception:
        st = {}

    if not isinstance(st, dict):
        st = {}

    st.setdefault("published", {})
    st["published"].setdefault("instagram", [])
    st["published"].setdefault("facebook", [])
    st["published"].setdefault("youtube_shorts", [])

    st.setdefault("published_source_groups", {})
    st["published_source_groups"].setdefault("instagram", [])
    st["published_source_groups"].setdefault("facebook", [])
    st["published_source_groups"].setdefault("youtube_shorts", [])

    st.setdefault("history", [])
    st.setdefault("skips", [])

    if not isinstance(st["history"], list):
        st["history"] = []
    if not isinstance(st["skips"], list):
        st["skips"] = []

    return st


def save_state(st):
    if not isinstance(st, dict):
        st = {}

    st.setdefault("published", {})
    st["published"].setdefault("instagram", [])
    st["published"].setdefault("facebook", [])
    st["published"].setdefault("youtube_shorts", [])

    st.setdefault("published_source_groups", {})
    st["published_source_groups"].setdefault("instagram", [])
    st["published_source_groups"].setdefault("facebook", [])
    st["published_source_groups"].setdefault("youtube_shorts", [])

    st.setdefault("history", [])
    st.setdefault("skips", [])

    r2().put_object(
        Bucket=BUCKET_NAME,
        Key=STATE_KEY,
        Body=json.dumps(st, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def is_published_on(st, platform, key):
    return key in st.get("published", {}).get(platform, [])


def mark_published_on(st, platform, key):
    st.setdefault("published", {})
    st["published"].setdefault(platform, [])
    if key not in st["published"][platform]:
        st["published"][platform].append(key)
        st["published"][platform] = st["published"][platform][-5000:]


def is_source_group_published_on(st, platform, source_group):
    if not source_group:
        return False
    return source_group in st.get("published_source_groups", {}).get(platform, [])


def mark_source_group_published_on(st, platform, source_group):
    if not source_group:
        return
    st.setdefault("published_source_groups", {})
    st["published_source_groups"].setdefault(platform, [])
    if source_group not in st["published_source_groups"][platform]:
        st["published_source_groups"][platform].append(source_group)
        st["published_source_groups"][platform] = st["published_source_groups"][platform][-5000:]


def append_history(st, payload):
    st.setdefault("history", [])
    st["history"].append(payload)
    st["history"] = st["history"][-10000:]


def append_skip(st, payload):
    st.setdefault("skips", [])
    st["skips"].append(payload)
    st["skips"] = st["skips"][-10000:]


def extract_source_group_from_key(key):
    base = os.path.basename(key).rsplit(".", 1)[0]

    if "__hype__" in base:
        base = base.split("__hype__", 1)[0]

    m = re.match(r"^(\d{8}_\d{6})_\d+$", base)
    if m:
        return m.group(1)

    parts = base.split("__")
    if len(parts) >= 3:
        return "__".join(parts[:-2])

    return base


def resolve_meta_key_for_video_key(key):
    base = os.path.basename(key).rsplit(".", 1)[0]

    if key.startswith(f"{PREFIX_PRIORITY}/"):
        return f"ugc/meta/final_priority/{base}.json"

    if key.startswith(f"{PREFIX_MANUAL}/"):
        return f"ugc/meta/final_manual/{base}.json"

    return f"{META_FINAL_PREFIX}/{base}.json"


def resolve_sidecar_txt_for_video_key(key):
    base = os.path.basename(key).rsplit(".", 1)[0]
    folder = os.path.dirname(key)
    return f"{folder}/{base}.txt"


def load_meta_for_video_key(key):
    meta_key = resolve_meta_key_for_video_key(key)
    meta = load_json(meta_key)
    return meta_key, meta


def load_clip_meta_from_source_clip_key(source_clip_key):
    base = os.path.basename(source_clip_key).rsplit(".", 1)[0]
    clip_meta_key = f"ugc/meta/clips/{base}.json"
    return clip_meta_key, load_json(clip_meta_key)


def resolve_source_group(key, meta):
    if isinstance(meta, dict):
        sg = meta.get("source_group")
        if sg:
            return str(sg)

        source_video_id = meta.get("source_video_id")
        if source_video_id:
            return str(source_video_id)

        source_clip_key = meta.get("source_clip_key")
        if source_clip_key:
            _, clip_meta = load_clip_meta_from_source_clip_key(source_clip_key)
            if isinstance(clip_meta, dict):
                sg2 = clip_meta.get("source_group") or clip_meta.get("source_video_id")
                if sg2:
                    return str(sg2)

    return extract_source_group_from_key(key)


def normalize_game_name(name):
    t = str(name or "").strip().lower()

    mapping = {
        "granturismo": "Gran Turismo",
        "gran turismo": "Gran Turismo",
        "gt": "Gran Turismo",
        "easportsfc": "EA Sports FC",
        "ea sports fc": "EA Sports FC",
        "fc": "EA Sports FC",
        "leagueoflegends": "League of Legends",
        "league of legends": "League of Legends",
        "lol": "League of Legends",
        "apex": "Apex Legends",
        "apex legends": "Apex Legends",
        "fortnite": "Fortnite",
        "minecraft": "Minecraft",
        "cs2": "CS2",
        "counter strike": "CS2",
        "counter-strike": "CS2",
        "valorant": "Valorant",
        "warzone": "Warzone",
        "f1": "F1",
        "simracing": "Gran Turismo",
        "sim racing": "Gran Turismo",
        "esports": "Esports",
        "generic": "Generic",
    }

    return mapping.get(t, str(name).strip() or "Generic")


def detect_game_from_key(key):
    text = (key or "").lower().replace("_", " ").replace("-", " ")

    checks = [
        ("valorant", "Valorant"),
        ("vct", "Valorant"),
        ("cs2", "CS2"),
        ("counter strike", "CS2"),
        ("counter-strike", "CS2"),
        ("league of legends", "League of Legends"),
        ("lol", "League of Legends"),
        ("fortnite", "Fortnite"),
        ("warzone", "Warzone"),
        ("apex legends", "Apex Legends"),
        ("apex", "Apex Legends"),
        ("minecraft", "Minecraft"),
        ("ea sports fc", "EA Sports FC"),
        ("easportsfc", "EA Sports FC"),
        ("fc 26", "EA Sports FC"),
        ("fc 25", "EA Sports FC"),
        ("fc pro", "EA Sports FC"),
        ("f1", "F1"),
        ("gran turismo", "Gran Turismo"),
        ("granturismo", "Gran Turismo"),
        ("sim racing", "Gran Turismo"),
        ("simracing", "Gran Turismo"),
        ("gt world series", "Gran Turismo"),
    ]

    for needle, label in checks:
        if needle in text:
            return label

    return "Generic"


def resolve_game_name(key, meta):
    if isinstance(meta, dict):
        for field in ["game", "game_name"]:
            value = meta.get(field)
            if value:
                return normalize_game_name(value)

        source_clip_key = meta.get("source_clip_key")
        if source_clip_key:
            _, clip_meta = load_clip_meta_from_source_clip_key(source_clip_key)
            if isinstance(clip_meta, dict):
                for field in ["game", "game_name"]:
                    value = clip_meta.get(field)
                    if value:
                        return normalize_game_name(value)

    return detect_game_from_key(key)


def diversify_queue(items):
    if not B_AVOID_SAME_SOURCE_PER_RUN:
        return items

    groups = {}
    order = []

    for item in items:
        group = item.get("source_group") or "unknown"
        if group not in groups:
            groups[group] = []
            order.append(group)
        groups[group].append(item)

    diversified = []

    while True:
        added = 0
        for group in order:
            if groups[group]:
                diversified.append(groups[group].pop(0))
                added += 1
        if added == 0:
            break

    return diversified


GAME_CONTEXTS = {
    "Valorant": [
        "Aquí se gana antes del duelo.",
        "Esto separa al que pega tiros del que entiende el momento.",
        "No es reflejo. Es lectura total.",
        "Un error mínimo y ya te sentenció.",
    ],
    "CS2": [
        "Aim, timing y cabeza en el segundo exacto.",
        "Esto no es highlight vacío. Es castigo real.",
        "Un hueco mínimo y ya te borró la ronda.",
        "La ventana es mínima y aun así la cobra.",
    ],
    "Fortnite": [
        "En zona final el que duda se muere.",
        "Esto no es solo mecánica. Es sangre fría.",
        "Hay edits buenos y luego está esta barbaridad.",
        "Aquí el error se paga en un frame.",
    ],
    "Warzone": [
        "Aquí no ganó el más loco. Ganó el que entendió el caos.",
        "Esto no es solo aim. Es control del desastre.",
        "Hay cierres buenos y luego está esto.",
        "Todos corren, uno solo entiende el cierre.",
    ],
    "EA Sports FC": [
        "Esto entra para partir la comunidad en dos.",
        "Gol que vale uno y comentarios toda la semana.",
        "Aquí el rival ayuda, pero el castigo también tiene clase.",
        "Ve un hueco mínimo y lo manda adentro.",
    ],
    "Gran Turismo": [
        "Aquí no hubo humo: hubo control fino y sangre fría.",
        "Se ve limpio porque está hecho con cabeza.",
        "Esto es precisión real, no replay bonito.",
        "El margen es mínimo y aun así no falla.",
    ],
    "F1": [
        "Esto no sale limpio sin precisión y cero miedo al error.",
        "Hay espacio mínimo y aun así decide atacar ahí.",
        "No fue suerte. Fue cálculo.",
        "No todos meten el coche ahí.",
    ],
    "Minecraft": [
        "Parece absurdo, sí, y justo por eso internet se divide en dos.",
        "Esto parece meme, pero también hay mérito real.",
        "Hay momentos raros y luego está esta locura.",
        "Parece casualidad hasta que lo miras dos veces.",
    ],
    "Esports": [
        "Esto no fue una jugada cualquiera.",
        "Hay clips que entretienen y otros que te obligan a comentar.",
        "Aquí pasó algo que no se puede dejar sin debate.",
    ],
}

GAME_CTAS = {
    "Valorant": [
        "¿TÚ LO RESUELVES ASÍ?",
        "¿Skill total o el rival colaboró demasiado?",
        "¿Esto es IQ puro o la están inflando?",
    ],
    "CS2": [
        "¿ESO ES PURO AIM O IQ DE OTRO PLANETA?",
        "¿Play top o demasiado regalo rival?",
        "¿La están inflando o sí estuvo criminal?",
    ],
    "Fortnite": [
        "¿EDIT LIMPIO O PURO CAOS FAVORABLE?",
        "¿Esto es clutch real o milagro en zona?",
        "¿Top play o replay que la infla?",
    ],
    "Warzone": [
        "¿Skill brutal o suerte con esteroides?",
        "¿Clutch o puro caos favorable?",
        "¿Eso es control total o milagro armado?",
    ],
    "EA Sports FC": [
        "¿GOLAZO O DEFENSA DE PLASTILINA?",
        "¿Clase real o el rival ayudó demasiado?",
        "¿Esto es top o puro abuso del juego?",
    ],
    "Gran Turismo": [
        "¿CLASE REAL O SE VE MÁS BONITO DE LO QUE FUE?",
        "¿Precisión total o hype de replay?",
        "¿Manejo limpio o error rival?",
    ],
    "F1": [
        "¿MANIOBRA LEGENDARIA O RIESGO INNECESARIO?",
        "¿Talento puro o puerta abierta del rival?",
        "¿Mucho hype o adelantamiento serio?",
    ],
    "Minecraft": [
        "¿GENIALIDAD O CLIP MALDITO?",
        "¿Skill o puro momento de internet?",
        "¿Esto fue cerebro o caos bendecido?",
    ],
    "Esports": [
        "¿ESTO ES CINE O NO?",
        "¿Skill puro o bastante fortuna?",
        "¿Merece tanto hype como le están dando?",
    ],
}

GAME_HASHTAGS = {
    "Valorant": ["#Valorant", "#VCT", "#GamingLATAM", "#ValorantLATAM", "#EsportsLATAM"],
    "CS2": ["#CS2", "#CounterStrike", "#GamingLATAM", "#CS2LATAM", "#EsportsLATAM"],
    "League of Legends": ["#LeagueOfLegends", "#LoL", "#Gaming
