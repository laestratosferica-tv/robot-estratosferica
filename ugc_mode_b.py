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
EDITORIAL_MEMORY_SUMMARY_KEY = env_nonempty(
    "B_EDITORIAL_MEMORY_SUMMARY_KEY",
    "ugc/state/editorial_memory_summary.json"
)

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

# MODE B v6.1
B_CAPTION_MAX_WORDS = env_int("B_CAPTION_MAX_WORDS", 42)
B_CAPTION_MIN_SCORE = env_float("B_CAPTION_MIN_SCORE", 3.6)
B_USE_OPENAI_POLISH = env_bool("B_USE_OPENAI_POLISH", True)
B_OPENAI_POLISH_MIN_EDITORIAL = env_float("B_OPENAI_POLISH_MIN_EDITORIAL", 3.0)

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

    if not isinstance(st["published"]["instagram"], list):
        st["published"]["instagram"] = []
    if not isinstance(st["published"]["facebook"], list):
        st["published"]["facebook"] = []
    if not isinstance(st["published"]["youtube_shorts"], list):
        st["published"]["youtube_shorts"] = []

    if not isinstance(st["published_source_groups"]["instagram"], list):
        st["published_source_groups"]["instagram"] = []
    if not isinstance(st["published_source_groups"]["facebook"], list):
        st["published_source_groups"]["facebook"] = []
    if not isinstance(st["published_source_groups"]["youtube_shorts"], list):
        st["published_source_groups"]["youtube_shorts"] = []

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


def load_editorial_memory_summary():
    try:
        data = load_json(EDITORIAL_MEMORY_SUMMARY_KEY)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def get_editorial_game_memory(summary, game_name):
    if not isinstance(summary, dict):
        return {}

    games = summary.get("games") or {}
    if not isinstance(games, dict):
        return {}

    key = str(game_name or "").strip().lower()
    data = games.get(key)

    if isinstance(data, dict):
        return data

    generic = games.get("generic")
    return generic if isinstance(generic, dict) else {}


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
        "La mayoría aquí la vende.",
    ],
    "CS2": [
        "Aim, timing y cabeza en el segundo exacto.",
        "Esto no es highlight vacío. Es castigo real.",
        "La mayoría asoma mal aquí.",
        "Esto huele a castigo puro.",
    ],
    "Fortnite": [
        "En zona final el que duda se muere.",
        "Esto no es solo mecánica. Es sangre fría.",
        "Hay edits buenos y luego está esta barbaridad.",
        "La mayoría aquí se apaga.",
    ],
    "Warzone": [
        "Aquí no ganó el más loco. Ganó el que entendió el caos.",
        "Esto no es solo aim. Es control del desastre.",
        "Hay cierres buenos y luego está esto.",
        "La mayoría aquí entra en pánico.",
    ],
    "EA Sports FC": [
        "Esto entra para partir la comunidad en dos.",
        "Gol que vale uno y comentarios toda la semana.",
        "Aquí el rival ayuda, pero el castigo también tiene clase.",
        "La defensa queda mal parada y él no perdona.",
    ],
    "Gran Turismo": [
        "Aquí no hubo humo: hubo control fino y sangre fría.",
        "Se ve limpio porque está hecho con cabeza.",
        "Esto es precisión real, no replay bonito.",
        "Aquí el mínimo error te manda afuera.",
    ],
    "F1": [
        "Esto no sale limpio sin precisión y cero miedo al error.",
        "Hay espacio mínimo y aun así decide atacar ahí.",
        "No fue suerte. Fue cálculo.",
        "La mayoría no se tira ahí.",
    ],
    "Minecraft": [
        "Parece absurdo, sí, y justo por eso internet se divide en dos.",
        "Esto parece meme, pero también hay mérito real.",
        "Hay momentos raros y luego está esta locura.",
        "Se ve maldito y por eso funciona.",
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
    "League of Legends": ["#LeagueOfLegends", "#LoL", "#GamingLATAM", "#LoLLATAM", "#EsportsLATAM"],
    "Fortnite": ["#Fortnite", "#FortniteLATAM", "#GamingLATAM", "#FortniteClips", "#EsportsLATAM"],
    "Warzone": ["#Warzone", "#CallOfDuty", "#GamingLATAM", "#WarzoneLATAM", "#EsportsLATAM"],
    "Apex Legends": ["#ApexLegends", "#Apex", "#GamingLATAM", "#ApexLATAM", "#EsportsLATAM"],
    "Minecraft": ["#Minecraft", "#MinecraftLATAM", "#GamingLATAM", "#MinecraftClips", "#ReelsGaming"],
    "EA Sports FC": ["#EASportsFC", "#FCLATAM", "#GamingLATAM", "#EsportsLATAM", "#FCClips"],
    "F1": ["#F1", "#SimRacing", "#GamingLATAM", "#F1Esports", "#EsportsLATAM"],
    "Gran Turismo": ["#GranTurismo", "#SimRacing", "#GTLATAM", "#GamingLATAM", "#ReelsGaming"],
    "Esports": ["#EsportsLATAM", "#GamingLATAM", "#Esports", "#Gaming", "#ReelsGaming"],
    "Generic": ["#GamingLATAM", "#ReelsGaming", "#Gaming"],
}

VALORANT_HOOKS = [
    "ESTO NO ES AIM",
    "ESTO ES IQ DE VALORANT",
    "LO GANÓ ANTES DEL DUELO",
    "LECTURA QUE HUMILLA",
    "ESTO ES CLUTCH MENTAL",
    "NO DISPARA... DECIDE",
]

CS2_HOOKS = [
    "ESO ES PURO AIM",
    "BORRÓ AL SERVER",
    "PREAIM DE OTRO PLANETA",
    "ESTO ES HEADSHOT LAB",
    "LO DEJÓ SIN JUGAR",
]

FORTNITE_HOOKS = [
    "NO JUEGA... CONSTRUYE UNA PELÍCULA",
    "ESTO ES FINAL CIRCLE REAL",
    "EDIT QUE HUMILLA",
    "LO BORRÓ EN 2 SEGUNDOS",
    "ESO ES MECHANICS",
]

WARZONE_HOOKS = [
    "AQUÍ NO SOBREVIVE CUALQUIERA",
    "ESTO ES CAOS BIEN LEÍDO",
    "LO GANÓ CON CABEZA",
    "NO ES AIM... ES CONTROL",
]

FC_HOOKS = [
    "GOL QUE DUELE",
    "ESO ES TIMING PURO",
    "LO LEYÓ COMPLETO",
    "DEFENSA DESAPARECIDA",
    "ESTO NO ES SUERTE",
]

GT_HOOKS = [
    "ESTO ES PURA MANO",
    "PRECISIÓN QUE DUELE",
    "NO ES VELOCIDAD... ES CONTROL",
    "LIMPIO COMO CIRUGÍA",
]

F1_HOOKS = [
    "ESO ES PRECISIÓN PURA",
    "MANIOBRA QUE HUMILLA",
    "NO SE TIRA CUALQUIERA",
    "ADELANTAMIENTO DE HIELO",
]

MINECRAFT_HOOKS = [
    "ESTO PARECE EDITADO",
    "MINECRAFT ACABA DE REGALAR CINE",
    "¿QUÉ ACABO DE VER?",
    "ESTO NO TENÍA SENTIDO",
]

GENERIC_HOOKS = [
    "ESTO NO TENÍA SENTIDO",
    "¿QUÉ ACABO DE VER?",
    "ESTO ES CINE",
]


def pick_game_hook(game_name, emotion=None, moment_type=None):
    g = str(game_name or "").strip().lower()
    emotion = normalize_emotion(emotion)
    moment_type = normalize_moment_type(moment_type)

    if "valorant" in g:
        return random.choice(VALORANT_HOOKS)
    if "cs2" in g or "counter" in g:
        return random.choice(CS2_HOOKS)
    if "fortnite" in g:
        return random.choice(FORTNITE_HOOKS)
    if "warzone" in g:
        return random.choice(WARZONE_HOOKS)
    if "ea sports fc" in g or g == "fc" or "fifa" in g:
        return random.choice(FC_HOOKS)
    if "gran turismo" in g or g == "gt":
        return random.choice(GT_HOOKS)
    if g == "f1":
        return random.choice(F1_HOOKS)
    if "minecraft" in g:
        return random.choice(MINECRAFT_HOOKS)

    if moment_type == "clutch":
        return "ESTO ES CLUTCH MENTAL"
    if emotion == "skill":
        return "ESO ES PURA MANO"

    return random.choice(GENERIC_HOOKS)


def safe_float(v, default=0.0):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def is_weak_text(v):
    t = str(v or "").strip().lower()
    return t in ("", "generic", "unknown", "misc", "none", "null", "n/a", "na")


def clean_signal(v):
    return str(v or "").strip()


def is_invalid_numeric(v):
    try:
        f = float(v)
        return math.isnan(f) or math.isinf(f)
    except Exception:
        return True


def sanitize_candidate_score(raw_value):
    if raw_value is None:
        return {"score": 0.0, "raw": raw_value, "valid": False, "reason": "missing"}

    try:
        f = float(raw_value)
        if math.isnan(f):
            return {"score": 0.0, "raw": raw_value, "valid": False, "reason": "nan"}
        if math.isinf(f):
            return {"score": 0.0, "raw": raw_value, "valid": False, "reason": "inf"}
        return {"score": f, "raw": raw_value, "valid": True, "reason": "ok"}
    except Exception:
        return {"score": 0.0, "raw": raw_value, "valid": False, "reason": "non_numeric"}


def is_weak_game_name(game_name):
    t = str(game_name or "").strip().lower()
    return t in ("", "generic", "gaming", "unknown")


def normalize_emotion(emotion):
    t = str(emotion or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "clutch": "clutch",
        "panic": "panic",
        "hype": "hype",
        "chaos": "chaos",
        "insane": "insane",
        "tense": "tense",
        "calm": "calm",
        "clean": "clean",
        "rage": "rage",
        "skill": "skill",
        "mechanical": "skill",
        "precision": "skill",
    }
    return aliases.get(t, t)


def normalize_intensity(intensity):
    t = str(intensity or "").strip().lower()
    aliases = {
        "very_high": "high",
        "high": "high",
        "mid": "medium",
        "medium": "medium",
        "low": "low",
    }
    return aliases.get(t, t)


def normalize_moment_type(moment_type):
    t = str(moment_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "finalcircle": "final_circle",
        "final_circle": "final_circle",
        "zone": "final_circle",
        "endgame": "final_circle",
        "clutch": "clutch",
        "ace": "ace",
        "goal": "goal",
        "gol": "goal",
        "last_second": "last_second",
        "ultimo_segundo": "last_second",
        "overtime": "last_second",
        "1v2": "clutch",
        "1v3": "clutch",
        "1v4": "clutch",
        "1v5": "clutch",
        "wipe": "ace",
        "team_wipe": "ace",
        "moment": "highlight",
        "action": "highlight",
        "play": "highlight",
        "highlight": "highlight",
    }
    return aliases.get(t, t or "highlight")


def extract_copy_signals(meta):
    if not isinstance(meta, dict):
        return {}

    return {
        "hook": clean_signal(meta.get("hook") or meta.get("hook_text") or meta.get("title_hook")),
        "cta": clean_signal(meta.get("cta") or meta.get("cta_text")),
        "badge": clean_signal(meta.get("badge") or meta.get("badge_text")),
        "caption_base": clean_signal(meta.get("caption_base") or meta.get("base_caption")),
        "caption": clean_signal(meta.get("caption")),
        "shorts_title": clean_signal(meta.get("shorts_title")),
        "shorts_description": clean_signal(meta.get("shorts_description")),
        "emotion": clean_signal(meta.get("emotion")),
        "intensity": clean_signal(meta.get("intensity")),
        "moment_type": clean_signal(meta.get("moment_type")),
        "game": clean_signal(meta.get("game") or meta.get("game_name")),
    }


def should_fallback_field(value, kind="text"):
    if kind == "score":
        return value is None or is_invalid_numeric(value)
    if kind == "game":
        return is_weak_game_name(value)
    return is_weak_text(value)


def merge_meta_with_clip_fallback(final_meta, clip_meta):
    final_meta = final_meta or {}
    clip_meta = clip_meta or {}

    merged = dict(final_meta)

    for field in ["candidate_score", "emotion", "intensity", "moment_type"]:
        if should_fallback_field(merged.get(field), "score" if field == "candidate_score" else "text"):
            if clip_meta.get(field) is not None:
                merged[field] = clip_meta.get(field)

    for field in ["game", "game_name", "source_group", "source_video_id", "source_video_key"]:
        if should_fallback_field(merged.get(field), "game" if field in ("game", "game_name") else "text"):
            if clip_meta.get(field) is not None:
                merged[field] = clip_meta.get(field)

    for field in ["hook", "cta", "badge", "caption_base", "caption", "shorts_title", "shorts_description"]:
        if should_fallback_field(merged.get(field), "text"):
            if clip_meta.get(field) is not None:
                merged[field] = clip_meta.get(field)

    if not merged.get("source_clip_key") and clip_meta.get("source_clip_key"):
        merged["source_clip_key"] = clip_meta.get("source_clip_key")

    return merged


def compute_editorial_score(item):
    score = 0.0
    game_name = item.get("game_name")
    emotion = normalize_emotion(item.get("emotion"))
    intensity = normalize_intensity(item.get("intensity"))
    moment_type = normalize_moment_type(item.get("moment_type"))
    candidate_score = safe_float(item.get("candidate_score"), 0.0)
    copy_signals = item.get("copy_signals") or {}

    if not is_weak_game_name(game_name):
        score += 1.5

    if candidate_score >= 0.80:
        score += 1.75
    elif candidate_score >= 0.65:
        score += 1.25
    elif candidate_score >= 0.55:
        score += 0.75
    elif candidate_score >= 0.50:
        score += 0.45
    elif candidate_score > 0:
        score += 0.15

    if emotion in ("skill", "clutch", "chaos", "insane", "tense", "hype", "clean"):
        score += 0.75

    if intensity == "high":
        score += 0.9
    elif intensity == "medium":
        score += 0.55
    elif intensity == "low":
        score += 0.15

    if moment_type in ("clutch", "ace", "final_circle", "goal", "last_second"):
        score += 1.4
    elif moment_type == "highlight":
        score += 0.55

    if clean_signal(copy_signals.get("hook")):
        score += 1.0
    if clean_signal(copy_signals.get("cta")):
        score += 0.85
    if clean_signal(copy_signals.get("badge")):
        score += 0.2
    if clean_signal(copy_signals.get("caption_base")):
        score += 0.35

    if game_name in ("Valorant", "Fortnite", "EA Sports FC", "Warzone", "CS2"):
        score += 0.35

    if is_weak_game_name(game_name):
        score -= 1.8

    if item.get("score_valid") is False:
        score -= 1.2

    return round(score, 4)


def openai_text(prompt):
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {"model": OPENAI_MODEL, "input": prompt}

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=payload,
        timeout=90,
    )
    r.raise_for_status()
    j = r.json()

    if j.get("output_text"):
        return j["output_text"].strip()

    texts = []
    for item in j.get("output", []) or []:
        for part in item.get("content", []) or []:
            if part.get("type") == "output_text" and part.get("text"):
                texts.append(part["text"])

    return "\n".join(texts).strip()


def parse_brief_text(raw_text):
    result = {
        "campaign": None,
        "game": None,
        "type": None,
        "priority": None,
        "emotion": None,
        "angle": None,
        "target": None,
        "hook": None,
        "cta": None,
        "style": None,
        "intensity": None,
        "notes": None,
        "raw_text": (raw_text or "").strip(),
    }

    if not raw_text or not raw_text.strip():
        return result

    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    field_map = {
        "CAMPAIGN": "campaign",
        "GAME": "game",
        "TYPE": "type",
        "PRIORITY": "priority",
        "EMOTION": "emotion",
        "ANGLE": "angle",
        "TARGET": "target",
        "HOOK": "hook",
        "CTA": "cta",
        "STYLE": "style",
        "INTENSITY": "intensity",
        "NOTES": "notes",
        "NOTA": "notes",
        "NOTA EDITORIAL": "notes",
        "OBJETIVO": "notes",
    }

    current_multiline_key = None
    notes_buffer = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current_multiline_key == "notes":
                notes_buffer.append("")
            continue

        matched = False
        for label, dest in field_map.items():
            prefix = f"{label}:"
            if line.upper().startswith(prefix):
                value = line[len(prefix):].strip()
                if dest == "notes":
                    current_multiline_key = "notes"
                    if value:
                        notes_buffer.append(value)
                else:
                    result[dest] = value or None
                    current_multiline_key = None
                matched = True
                break

        if matched:
            continue

        if current_multiline_key == "notes":
            notes_buffer.append(line)

    if notes_buffer:
        result["notes"] = "\n".join(notes_buffer).strip() or None

    return result


def load_and_parse_sidecar_brief(key):
    txt_key = resolve_sidecar_txt_for_video_key(key)
    raw_txt = load_text(txt_key)

    if not raw_txt:
        return txt_key, None, None

    parsed = parse_brief_text(raw_txt)
    return txt_key, raw_txt, parsed


def ensure_game_in_caption(caption, game_name):
    caption = (caption or "").strip()
    game_name = (game_name or "").strip()

    if not caption or not game_name:
        return caption
    if game_name.lower() in caption.lower():
        return caption

    return f"{game_name}: {caption}"


def build_campaign_caption_from_brief(key, meta, brief):
    game_name = brief.get("game") or resolve_game_name(key, meta) or "gaming"

    prompt = f"""
Eres editor viral premium de gaming y esports LATAM.

Escribe un caption para una pieza PRIORITY.
Debe sonar:
- potente
- emocional
- premium
- comentable
- nada genérico

Contexto:
Archivo: {os.path.basename(key)}
Juego: {game_name}
Campaign: {brief.get('campaign') or ''}
Type: {brief.get('type') or ''}
Priority: {brief.get('priority') or ''}
Emotion: {brief.get('emotion') or ''}
Angle: {brief.get('angle') or ''}
Target: {brief.get('target') or ''}
Hook idea: {brief.get('hook') or ''}
CTA idea: {brief.get('cta') or ''}
Style: {brief.get('style') or ''}
Intensity: {brief.get('intensity') or ''}
Notes: {brief.get('notes') or ''}

Reglas:
- hook fuerte al inicio
- 2 a 4 líneas
- pregunta/cta final
- incluir el juego sí o sí
- 5 a 8 hashtags
- máximo 120 palabras

Devuelve solo el caption final.
"""
    try:
        if OPENAI_API_KEY:
            text = openai_text(prompt).strip()
            if text:
                return ensure_game_in_caption(text, game_name)
    except Exception as e:
        print("OpenAI brief caption fallback:", repr(e))

    hashtags = []
    if brief.get("campaign"):
        hashtags.append(f"#{str(brief['campaign']).replace(' ', '')}")
    if game_name:
        hashtags.append(f"#{str(game_name).replace(' ', '')}")
    hashtags.extend(["#GamingLATAM", "#EsportsLATAM", "#ReelsGaming"])

    hook = brief.get("hook") or f"{game_name} no necesitaba hablar tan fuerte, pero aquí está."
    angle = brief.get("angle") or "Pieza premium con conflicto real."
    cta = brief.get("cta") or "¿Esto te gana o te parece demasiado vendible?"

    return f"""{hook}

{angle}

🔥 {cta}

{" ".join(hashtags[:7])}""".strip()


def build_campaign_shorts_title_from_brief(key, meta, brief):
    game_name = brief.get("game") or resolve_game_name(key, meta) or "Gaming"

    prompt = f"""
Eres editor de Shorts gaming LATAM.

Crea un título corto, fuerte, premium y comentable.

Archivo: {os.path.basename(key)}
Juego: {game_name}
Campaign: {brief.get('campaign') or ''}
Angle: {brief.get('angle') or ''}
Hook idea: {brief.get('hook') or ''}
Style: {brief.get('style') or ''}
Intensity: {brief.get('intensity') or ''}

Reglas:
- máximo 80 caracteres antes de #Shorts
- devolver solo el título final
"""
    try:
        if OPENAI_API_KEY:
            text = openai_text(prompt).strip().replace('"', "").strip()
            if text:
                if "#Shorts" not in text:
                    text = f"{text} #Shorts"
                return text[:100]
    except Exception as e:
        print("OpenAI brief shorts title fallback:", repr(e))

    base = brief.get("hook") or f"{game_name}: demasiado fuerte para ignorarlo"
    if "#Shorts" not in base:
        base = f"{base} #Shorts"
    return base[:100]


def build_campaign_shorts_description_from_brief(key, meta, brief):
    game_name = brief.get("game") or resolve_game_name(key, meta) or "Gaming"

    prompt = f"""
Escribe descripción para Shorts de gaming LATAM.

Juego: {game_name}
Campaign: {brief.get('campaign') or ''}
Emotion: {brief.get('emotion') or ''}
Angle: {brief.get('angle') or ''}
Target: {brief.get('target') or ''}
CTA: {brief.get('cta') or ''}
Notes: {brief.get('notes') or ''}

Reglas:
- 2 a 4 líneas
- tono hype/premium
- terminar con hashtags
- devuelve solo el texto final
"""
    try:
        if OPENAI_API_KEY:
            text = openai_text(prompt).strip()
            if text:
                return text[:5000]
    except Exception as e:
        print("OpenAI brief shorts description fallback:", repr(e))

    hashtags = []
    if brief.get("campaign"):
        hashtags.append(f"#{str(brief['campaign']).replace(' ', '')}")
    if game_name:
        hashtags.append(f"#{str(game_name).replace(' ', '')}")
    hashtags.extend(["#GamingLATAM", "#EsportsLATAM", "#ReelsGaming"])

    angle = brief.get("angle") or "Pieza premium gaming"
    cta = brief.get("cta") or "¿La comprarías o puro humo?"

    return f"""{angle}

{cta}

{" ".join(hashtags[:7])}""".strip()


def clean_line(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = text.strip(" -–—|•")
    return text


def sentence_case(text):
    text = clean_line(text)
    if not text:
        return text
    return text[0].upper() + text[1:]


def clip_text(text, max_len):
    text = clean_line(text)
    if len(text) <= max_len:
        return text
