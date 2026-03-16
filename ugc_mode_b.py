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
    return text[:max_len].rstrip(" ,.;:!?") + "…"


def word_count(text):
    return len(re.findall(r"\S+", str(text or "")))


def contains_question(text):
    t = str(text or "")
    return "¿" in t or "?" in t


def has_conflict_words(text):
    t = str(text or "").lower()
    needles = [
        "skill", "suerte", "regalado", "choke", "clutch", "humo", "cine",
        "rotísimo", "roto", "milagro", "regalo", "fraude", "castigo",
        "error rival", "lobby", "top", "inflando", "humilla", "sentencia",
        "vende", "vende humo", "de plastilina", "caos", "iq", "mano",
        "dormido", "regalada", "puerta abierta", "abuso", "hielo",
    ]
    return any(n in t for n in needles)


def looks_generic_caption(text):
    t = str(text or "").strip().lower()

    generic_patterns = [
        "por esto seguimos viendo",
        "gran jugada",
        "increíble jugada",
        "increible jugada",
        "qué jugada",
        "que jugada",
        "qué momento",
        "que momento",
        "esto demuestra",
        "nivel competitivo",
        "momento épico",
        "momento epico",
        "clip increíble",
        "clip increible",
        "clip brutal",
        "no tenía sentido",
        "no tenia sentido",
        "es exactamente por lo que seguimos viendo esports",
    ]

    if any(p in t for p in generic_patterns):
        return True

    if len(t) < 18:
        return True

    return False


def has_soft_editorial_phrases(text):
    t = str(text or "").lower()
    bad = [
        "esto no sale sin lectura real",
        "y sí, esto va a partir comentarios",
        "y si, esto va a partir comentarios",
        "esto obliga a tomar postura",
        "lectura criminal",
        "esto no cae bien a todo el mundo",
    ]
    return any(x in t for x in bad)


def dedupe_preserve_order(items):
    seen = set()
    result = []
    for x in items:
        k = clean_line(x).lower()
        if not k or k in seen:
            continue
        seen.add(k)
        result.append(x)
    return result


def infer_moment_label(ctx):
    game = ctx["game_name"]
    moment_type = ctx["moment_type"]
    emotion = ctx["emotion"]

    if moment_type == "final_circle":
        if game == "Fortnite":
            return "zona final"
        if game == "Warzone":
            return "cierre"
        return "momento final"

    if moment_type == "ace":
        if game == "Valorant":
            return "ace"
        if game == "CS2":
            return "wipe"
        return "borrada total"

    if moment_type == "goal":
        return "golazo"

    if moment_type == "last_second":
        return "último segundo"

    if moment_type == "clutch":
        return "clutch"

    if emotion == "chaos":
        return "caos total"

    if emotion == "skill":
        return "momento de manos"

    return "jugada"


def infer_verdict_label(ctx):
    game = ctx["game_name"]
    emotion = ctx["emotion"]
    intensity = ctx["intensity"]
    moment_type = ctx["moment_type"]

    if moment_type == "goal":
        return random.choice([
            "no define, sentencia",
            "no remata, firma la humillación",
            "la manda a guardar con una calma criminal",
            "ve medio hueco y castiga",
        ])

    if game == "Fortnite" and moment_type == "final_circle":
        return random.choice([
            "no juega, sentencia",
            "no sobrevive, castiga",
            "no improvisa, ejecuta",
            "ve el error y lo borra",
        ])

    if game == "Valorant" and moment_type == "clutch":
        return random.choice([
            "gana la ronda antes del duelo",
            "no entra por reflejo, entra con lectura",
            "no se acelera, los lee completos",
            "parece simple porque lo leyó mejor",
        ])

    if game == "CS2":
        return random.choice([
            "no perdona el timing",
            "ve la ventana mínima y la cobra",
            "borra la ronda sin regalar nada",
            "castiga en el segundo exacto",
        ])

    if game == "Warzone":
        return random.choice([
            "no compra el caos, lo ordena",
            "huele el error y lo liquida",
            "parece milagro hasta que ves la lectura",
            "no regala ni un frame",
        ])

    if game == "EA Sports FC":
        return random.choice([
            "ve el hueco y sentencia",
            "la defensa ayuda, sí, pero hay que cobrarla",
            "esto divide porque entra demasiado limpio",
            "lee la jugada antes del toque final",
        ])

    if game == "F1":
        return random.choice([
            "no todos meten el coche ahí",
            "parece limpio porque hay cálculo detrás",
            "no es humo, es decisión quirúrgica",
            "ve una puerta mínima y la usa",
        ])

    if game == "Minecraft":
        return random.choice([
            "parece casualidad hasta que lo ves dos veces",
            "suena meme, pero hay mérito real",
            "internet le dirá suerte, pero hay lectura",
            "se ve maldito y por eso divide",
        ])

    if emotion == "chaos":
        return random.choice([
            "el caos lo favorece, sí, pero también lo sabe leer",
            "en el desastre también hay mérito",
            "parece suerte hasta que lo ves dos veces",
            "todo se rompe y él queda mejor parado",
        ])

    if emotion == "skill":
        return random.choice([
            "parece fácil porque está bien leído",
            "ve medio error y lo sentencia",
            "no se acelera, castiga",
            "no lo resuelve por reflejo, lo cobra con lectura",
        ])

    if intensity == "high":
        return random.choice([
            "en alta presión casi nadie resuelve así",
            "con esta tensión muchos se rompen",
            "aquí el error vale carísimo",
            "con esta presión la mayoría se nubla",
        ])

    return random.choice([
        "se ve limpio porque está hecho con decisión",
        "aquí muchos la regalan",
        "un error mínimo y ya te castiga",
    ])


def infer_question_label(ctx):
    game = ctx["game_name"]
    moment_type = ctx["moment_type"]
    emotion = ctx["emotion"]
    cta = clean_line(ctx.get("cta"))

    if cta and contains_question(cta):
        return cta

    if game == "Fortnite":
        if moment_type == "final_circle":
            return random.choice([
                "¿Clutch real o zona demasiado regalada?",
                "¿Skill total o milagro con esteroides?",
                "¿Esto te parece cine o puro caos favorable?",
            ])
        return random.choice([
            "¿Mechanics reales o lobby dormido?",
            "¿Top play o la están inflando?",
        ])

    if game == "Valorant":
        return random.choice([
            "¿IQ puro o el rival colaboró demasiado?",
            "¿Clutch real o mucha ayuda enfrente?",
            "¿Esto es top tier o puro humo con replay?",
        ])

    if game == "CS2":
        return random.choice([
            "¿Puro aim o timing de otro planeta?",
            "¿Play criminal o demasiado regalo rival?",
            "¿La están inflando o sí estuvo asquerosa?",
        ])

    if game == "EA Sports FC":
        return random.choice([
            "¿Golazo o defensa de plastilina?",
            "¿Clase real o el rival hizo cosplay de cono?",
            "¿Esto fue talento o abuso del juego?",
        ])

    if game in ("Warzone", "Apex Legends"):
        return random.choice([
            "¿Skill brutal o suerte con esteroides?",
            "¿Control total o puro caos bendecido?",
            "¿Clutch real o milagro armado?",
        ])

    if moment_type == "goal":
        return random.choice([
            "¿Golazo real o defensa dormida?",
            "¿Clase pura o regalo total?",
        ])

    if emotion == "chaos":
        return random.choice([
            "¿Esto es cerebro o puro caos bendecido?",
            "¿Skill o suerte demasiado maquillada?",
        ])

    return random.choice([
        "¿Esto es cine o la están vendiendo de más?",
        "¿Skill puro o bastante fortuna?",
        "¿Top clip o hype inflado?",
    ])


def build_caption_context(key, meta, item):
    copy_signals = item.get("copy_signals") or {}

    game_name = normalize_game_name(item.get("game_name") or resolve_game_name(key, meta))
    emotion = normalize_emotion(item.get("emotion"))
    intensity = normalize_intensity(item.get("intensity"))
    moment_type = normalize_moment_type(item.get("moment_type"))
    candidate_score = safe_float(item.get("candidate_score"), 0.0)
    editorial_score = safe_float(item.get("editorial_score"), 0.0)

    hook = clean_line(copy_signals.get("hook"))
    cta = clean_signal(copy_signals.get("cta"))
    badge = clean_line(copy_signals.get("badge"))
    caption_base = clean_line(copy_signals.get("caption_base"))
    source_group = clean_line(item.get("source_group"))
    filename = os.path.basename(key)

    weak_game = is_weak_game_name(game_name)

    editorial_summary = load_editorial_memory_summary()
    game_memory = get_editorial_game_memory(editorial_summary, game_name)

    context = {
        "key": key,
        "filename": filename,
        "game_name": game_name,
        "emotion": emotion,
        "intensity": intensity,
        "moment_type": moment_type,
        "candidate_score": candidate_score,
        "editorial_score": editorial_score,
        "hook": hook,
        "cta": cta,
        "badge": badge,
        "caption_base": caption_base,
        "source_group": source_group,
        "weak_game": weak_game,
        "game_memory": game_memory,
    }

    context["moment_label"] = infer_moment_label(context)
    context["verdict_label"] = infer_verdict_label(context)
    context["question_label"] = infer_question_label(context)

    return context


def normalize_memory_word(word):
    return re.sub(r"^[^\wáéíóúñ]+|[^\wáéíóúñ]+$", "", str(word or "").lower()).strip()


def get_memory_top_words(ctx):
    gm = ctx.get("game_memory") or {}
    words = gm.get("top_words") or []
    if not isinstance(words, list):
        return []
    out = []
    for w in words:
        nw = normalize_memory_word(w)
        if nw:
            out.append(nw)
    return out


def get_recent_phrase_counts(ctx):
    gm = ctx.get("game_memory") or {}
    counts = gm.get("recent_phrase_counts") or {}
    if not isinstance(counts, dict):
        return {}
    out = {}
    for k, v in counts.items():
        try:
            out[str(k).strip().lower()] = int(v)
        except Exception:
            continue
    return out


def get_opening_counts(ctx):
    gm = ctx.get("game_memory") or {}
    counts = gm.get("opening_counts") or {}
    if not isinstance(counts, dict):
        return {}
    out = {}
    for k, v in counts.items():
        try:
            out[str(k).strip().lower()] = int(v)
        except Exception:
            continue
    return out


def has_memory_words(ctx, *needles):
    words = set(get_memory_top_words(ctx))
    return all(str(n).lower() in words for n in needles)


def first_meaningful_line(text):
    lines = [clean_line(x) for x in str(text or "").splitlines() if clean_line(x)]
    for line in lines:
        if line.startswith("#"):
            continue
        return line
    return ""


def normalize_opening(line):
    line = clean_line(line).lower()
    line = re.sub(r"\s+", " ", line)

    known_prefixes = [
        "valorant:",
        "cs2:",
        "warzone:",
        "f1:",
        "minecraft:",
        "apex legends:",
        "ea sports fc:",
        "league of legends:",
        "fortnite:",
        "gran turismo:",
        "momento de manos y cero miedo",
        "momento de manos y sangre fría",
        "esto no es highlight, es castigo",
        "esto es mano y cabeza",
    ]

    for p in known_prefixes:
        if line.startswith(p):
            return p

    words = line.split()
    if len(words) >= 5:
        return " ".join(words[:5])
    return line


def score_recent_phrase_penalty(text, ctx):
    tl = str(text or "").lower()
    counts = get_recent_phrase_counts(ctx)

    if not counts:
        return 0.0

    penalty = 0.0

    tracked_phrases = [
        "en momento de manos",
        "la mayoría la vende aquí",
        "esto no es highlight, es castigo",
        "momento de manos y sangre fría",
        "la mayoría aquí la vende",
        "la mayoría se apaga aquí",
        "momento de manos y cero miedo",
        "esto es mano y cabeza",
        "aquí muchos dudan y la regalan",
    ]

    for phrase in tracked_phrases:
        seen = counts.get(phrase, 0)
        if seen <= 0:
            continue

        if phrase in tl:
            if seen >= 5:
                penalty += 3.2
            elif seen >= 3:
                penalty += 2.2
            elif seen >= 2:
                penalty += 1.35
            else:
                penalty += 0.65

    return penalty


def score_opening_penalty(text, ctx):
    opening = normalize_opening(first_meaningful_line(text))
    counts = get_opening_counts(ctx)
    seen = counts.get(opening.lower(), 0)

    if seen >= 5:
        return 3.0
    if seen >= 3:
        return 2.0
    if seen >= 2:
        return 1.1
    return 0.0


def score_template_penalty(text, ctx):
    tl = str(text or "").lower()
    penalty = 0.0

    hard_templates = {
        "momento de manos y cero miedo": 3.2,
        "esto es mano y cabeza": 2.6,
        "aquí muchos dudan y la regalan": 2.2,
        "la mayoría aquí la vende": 2.6,
        "la mayoría la vende aquí": 2.8,
        "esto no es highlight, es castigo": 2.8,
        "momento de manos": 1.4,
    }

    for phrase, base_penalty in hard_templates.items():
        if phrase in tl:
            penalty += base_penalty

    if "momento de manos" in tl and has_memory_words(ctx, "manos", "momento"):
        penalty += 0.9

    if "cero miedo" in tl and has_memory_words(ctx, "cero", "miedo"):
        penalty += 1.0

    if "mano y cabeza" in tl and (has_memory_words(ctx, "mano", "cabeza") or has_memory_words(ctx, "manos", "cabeza")):
        penalty += 1.0

    if "la mayoría" in tl and "vende" in tl and has_memory_words(ctx, "mayoría", "vende"):
        penalty += 1.15

    return penalty


def generate_caption_candidates(ctx):
    game = ctx["game_name"]
    hook = clean_line(ctx.get("hook"))
    caption_base = clean_line(ctx.get("caption_base"))
    question = clean_line(ctx.get("question_label"))
    verdict = clean_line(ctx.get("verdict_label"))
    moment_label = clean_line(ctx.get("moment_label"))

    game_prefix = game if game and not is_weak_game_name(game) else "Este clip"

    hook_fallback = sentence_case(hook) if hook else None
    if not hook_fallback:
        hook_fallback = f"{game_prefix} no perdona aquí"

    line2_base = sentence_case(caption_base) if caption_base else ""
    if looks_generic_caption(line2_base):
        line2_base = ""

    candidates = []

    def add_candidate(line1, line2, line3):
        parts = [
            clean_line(line1),
            clean_line(line2),
            clean_line(line3),
        ]
        parts = [p for p in parts if p]
        text = "\n".join(parts)
        text = ensure_game_in_caption(text, game)
        candidates.append(text)

    add_candidate(
        hook_fallback,
        f"{game_prefix} {verdict}.",
        question,
    )

    add_candidate(
        f"{game_prefix}: {verdict}.",
        line2_base or "Aquí el error sale carísimo.",
        question,
    )

    add_candidate(
        hook_fallback,
        line2_base or "Todos dudan medio segundo; uno solo cobra.",
        question,
    )

    add_candidate(
        f"{game_prefix} no juega bonito, juega para castigar.",
        line2_base or verdict,
        question,
    )

    add_candidate(
        f"{game_prefix} ve medio error y lo cobra.",
        line2_base or verdict,
        question,
    )

    add_candidate(
        f"{game_prefix} aquí no improvisa.",
        line2_base or verdict,
        question,
    )

    add_candidate(
        f"{game_prefix} no lo hace por reflejo.",
        line2_base or verdict,
        question,
    )

    add_candidate(
        f"{game_prefix}: {moment_label} y sangre fría.",
        verdict,
        question,
    )

    add_candidate(
        f"{game_prefix} huele el error y lo castiga.",
        line2_base or verdict,
        question,
    )

    add_candidate(
        f"{game_prefix} ve una puerta mínima y la usa.",
        line2_base or verdict,
        question,
    )

    add_candidate(
        f"{game_prefix} parece simple porque está bien leído.",
        line2_base or verdict,
        question,
    )

    add_candidate(
        f"{game_prefix} no perdona ese hueco.",
        line2_base or verdict,
        question,
    )

    add_candidate(
        f"{game_prefix} lo cobra con una calma criminal.",
        line2_base or verdict,
        question,
    )

    add_candidate(
        f"{game_prefix} castiga donde casi nadie se anima.",
        line2_base or verdict,
        question,
    )

    add_candidate(
        f"{game_prefix} ve una décima de error y la convierte en clip.",
        line2_base or verdict,
        question,
    )

    if ctx["moment_type"] == "goal":
        add_candidate(
            f"{game_prefix}: {moment_label} que parte a la comunidad.",
            verdict,
            question,
        )

    if ctx["moment_type"] == "clutch":
        add_candidate(
            f"{game_prefix} aquí no tiembla.",
            verdict,
            question,
        )

    if ctx["moment_type"] == "final_circle":
        add_candidate(
            f"{game_prefix} en {moment_label} no perdona errores.",
            verdict,
            question,
        )

    if line2_base:
        add_candidate(
            hook_fallback,
            line2_base,
            question,
        )

    return dedupe_preserve_order(candidates)


def score_caption_candidate(text, ctx):
    score = 0.0
    t = str(text or "").strip()
    tl = t.lower()

    if not t:
        return -999.0

    game = (ctx.get("game_name") or "").strip().lower()
    hook = (ctx.get("hook") or "").strip().lower()
    cta = (ctx.get("cta") or "").strip().lower()
    caption_base = (ctx.get("caption_base") or "").strip().lower()
    moment_label = (ctx.get("moment_label") or "").strip().lower()

    if game and game in tl:
        score += 2.0
    else:
        score -= 3.0

    if contains_question(t):
        score += 1.4
    else:
        score -= 1.25

    if has_conflict_words(t):
        score += 1.6

    if moment_label and moment_label in tl:
        score += 0.5

    if hook and hook[:18] in tl:
        score += 0.55

    if cta and any(word in tl for word in re.findall(r"\w+", cta.lower())[:4]):
        score += 0.4

    if caption_base and any(word in tl for word in re.findall(r"\w+", caption_base.lower())[:5]):
        score += 0.2

    memory_words = get_memory_top_words(ctx)
    if memory_words:
        hits = 0
        for w in memory_words[:6]:
            if w and w in tl:
                hits += 1
        score += min(hits * 0.18, 0.55)

    wc = word_count(t)
    if 10 <= wc <= B_CAPTION_MAX_WORDS:
        score += 1.0
    elif wc < 8:
        score -= 1.4
    elif wc > B_CAPTION_MAX_WORDS:
        score -= 1.0

    lines = [x.strip() for x in t.splitlines() if x.strip()]
    if 2 <= len(lines) <= 3:
        score += 0.8
    elif len(lines) >= 4:
        score -= 0.4

    if looks_generic_caption(t):
        score -= 2.4

    if has_soft_editorial_phrases(t):
        score -= 1.6

    corporate_words = [
        "demuestra", "nivel competitivo", "seguimos viendo", "esports",
        "momento increíble", "momento increible", "gran jugada", "impresionante"
    ]
    if any(w in tl for w in corporate_words):
        score -= 2.5

    if "gaming latam" in tl or "reelsgaming" in tl:
        score -= 1.5

    if t.count("🔥") > 1:
        score -= 0.5

    score -= score_recent_phrase_penalty(t, ctx)
    score -= score_template_penalty(t, ctx)
    score -= score_opening_penalty(t, ctx)

    return round(score, 4)


def build_hashtags_for_game(game_name):
    tags = GAME_HASHTAGS.get(game_name, GAME_HASHTAGS["Esports"])[:4]
    return " ".join(tags)


def select_best_caption(ctx):
    candidates = generate_caption_candidates(ctx)

    ranked = []
    for c in candidates:
        s = score_caption_candidate(c, ctx)
        ranked.append({"text": c, "score": s})

    ranked.sort(key=lambda x: x["score"], reverse=True)

    best = ranked[0] if ranked else {"text": "", "score": -999.0}

    return {
        "best_text": best["text"],
        "best_score": best["score"],
        "ranked": ranked[:10],
    }


def polish_caption_with_openai(base_caption, ctx):
    if not OPENAI_API_KEY or not B_USE_OPENAI_POLISH:
        return base_caption

    if safe_float(ctx.get("editorial_score"), 0.0) < B_OPENAI_POLISH_MIN_EDITORIAL:
        return base_caption

    game_memory = ctx.get("game_memory") or {}
    memory_top_words = ", ".join((game_memory.get("top_words") or [])[:8])
    memory_patterns = "; ".join((game_memory.get("patterns_hint") or [])[:5])
    opening_counts = json.dumps(game_memory.get("opening_counts") or {}, ensure_ascii=False)
    recent_posts = game_memory.get("recent_posts", 0)

    prompt = f"""
Reescribe este caption para que suene más gamer LATAM, más natural y más comentable.

NO cambies el sentido.
NO lo vuelvas corporativo.
NO lo hagas genérico.
NO uses frases tipo "esto obliga a comentar" o "lectura criminal".
NO uses estas muletillas si puedes evitarlas:
- momento de manos y cero miedo
- esto es mano y cabeza
- la mayoría aquí la vende
- esto no es highlight, es castigo
- aquí muchos dudan y la regalan

Mantén el nombre del juego.
Mantén una pregunta final polarizante.
Máximo 42 palabras.
2 o 3 líneas.
Sin hashtags.

Juego: {ctx.get('game_name')}
Emotion: {ctx.get('emotion')}
Intensity: {ctx.get('intensity')}
Moment type: {ctx.get('moment_type')}
Hook H: {ctx.get('hook')}
CTA H: {ctx.get('cta')}
Caption base: {ctx.get('caption_base')}

Aprendizaje reciente del juego:
Recent posts: {recent_posts}
Top words: {memory_top_words}
Patterns hint: {memory_patterns}
Opening counts: {opening_counts}

Caption base:
{base_caption}

Devuelve solo el caption final.
""".strip()

    try:
        out = openai_text(prompt).strip()
        if not out:
            return base_caption

        out = ensure_game_in_caption(out, ctx.get("game_name"))
        if score_caption_candidate(out, ctx) >= score_caption_candidate(base_caption, ctx):
            return out
        return base_caption
    except Exception as e:
        print("OpenAI polish fallback:", repr(e))
        return base_caption


def build_caption_from_meta_v6(key, meta, item):
    ctx = build_caption_context(key, meta, item)

    selected = select_best_caption(ctx)
    caption = selected["best_text"]

    if not caption or selected["best_score"] < B_CAPTION_MIN_SCORE:
        caption = f"{ctx['game_name']}: clip que divide.\n{ctx['question_label']}"

    caption = polish_caption_with_openai(caption, ctx)
    caption = ensure_game_in_caption(caption, ctx["game_name"])

    hashtags = build_hashtags_for_game(ctx["game_name"])
    if hashtags:
        caption = f"{caption}\n{hashtags}"

    print("[B_V6] caption_context =", json.dumps({
        "game_name": ctx["game_name"],
        "emotion": ctx["emotion"],
        "intensity": ctx["intensity"],
        "moment_type": ctx["moment_type"],
        "hook": ctx["hook"],
        "cta": ctx["cta"],
        "caption_base": ctx["caption_base"],
        "editorial_score": ctx["editorial_score"],
        "candidate_score": ctx["candidate_score"],
    }, ensure_ascii=False))

    print("[B_V6] game_memory =", json.dumps(ctx.get("game_memory") or {}, ensure_ascii=False))
    print("[B_V6] top_caption_score =", selected["best_score"])
    print("[B_V6] top_candidates =", json.dumps(selected["ranked"][:3], ensure_ascii=False))

    item["caption_engine_debug"] = {
        "version": "v6.3_conflict_engine_opening_lock",
        "context": {
            "game_name": ctx["game_name"],
            "emotion": ctx["emotion"],
            "intensity": ctx["intensity"],
            "moment_type": ctx["moment_type"],
            "hook": ctx["hook"],
            "cta": ctx["cta"],
            "caption_base": ctx["caption_base"],
            "editorial_score": ctx["editorial_score"],
            "candidate_score": ctx["candidate_score"],
        },
        "game_memory": ctx.get("game_memory") or {},
        "top_caption_score": selected["best_score"],
        "top_candidates": selected["ranked"][:5],
    }

    return caption


def build_shorts_title_from_meta_v62(key, meta, item):
    game_name = item.get("game_name") or resolve_game_name(key, meta)
    moment_type = normalize_moment_type(item.get("moment_type"))
    emotion = normalize_emotion(item.get("emotion"))
    copy_signals = item.get("copy_signals") or {}
    hook_hint = clean_signal(copy_signals.get("hook"))

    if OPENAI_API_KEY:
        prompt = f"""
Crea título para YouTube Shorts gamer LATAM.

Juego: {game_name}
Emotion: {emotion}
Moment type: {moment_type}
Editorial score: {item.get("editorial_score")}
Hook sugerido: {hook_hint}
Archivo: {os.path.basename(key)}

Reglas:
- corto
- potente
- comentable
- máximo 75 caracteres antes de #Shorts
- evitar plantillas quemadas
- devolver solo el título
"""
        try:
            text = openai_text(prompt).strip().replace('"', "").strip()
            if text:
                if "#Shorts" not in text:
                    text = f"{text} #Shorts"
                return text[:100]
        except Exception as e:
            print("OpenAI shorts title v6.2 fallback:", repr(e))

    if hook_hint and len(hook_hint) < 70:
        title = hook_hint
    else:
        title = pick_game_hook(game_name, emotion, moment_type)

    if game_name and game_name.lower() not in title.lower():
        title = f"{game_name}: {title}"

    if "#Shorts" not in title:
        title = f"{title} #Shorts"
    return title[:100]


def build_shorts_description_from_meta_v62(key, meta, item):
    game_name = item.get("game_name") or resolve_game_name(key, meta)
    copy_signals = item.get("copy_signals") or {}
    cta_hint = clean_signal(copy_signals.get("cta"))
    caption_base = clean_signal(copy_signals.get("caption_base"))

    if OPENAI_API_KEY:
        prompt = f"""
Escribe descripción para YouTube Shorts gamer LATAM.

Juego: {game_name}
Editorial score: {item.get("editorial_score")}
CTA sugerido: {cta_hint}
Caption base: {caption_base}
Archivo: {os.path.basename(key)}

Reglas:
- 2 líneas + hashtags
- no sonar genérico
- máximo 350 caracteres
- evitar templates quemados
- devolver solo el texto
"""
        try:
            text = openai_text(prompt).strip()
            if text:
                return text[:5000]
        except Exception as e:
            print("OpenAI shorts description v6.2 fallback:", repr(e))

    line1 = caption_base or random.choice(GAME_CONTEXTS.get(game_name, GAME_CONTEXTS["Esports"]))
    line2 = cta_hint or random.choice(GAME_CTAS.get(game_name, GAME_CTAS["Esports"]))
    hashtags = " ".join(GAME_HASHTAGS.get(game_name, GAME_HASHTAGS["Esports"])[:5])

    return f"""{line1}

{line2}

{hashtags}""".strip()


def ig_publish(video_url, caption):
    print("IG publish: creando container...")

    resp = requests.post(
        f"{GRAPH_BASE}/{IG_USER_ID}/media",
        data={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "access_token": IG_ACCESS_TOKEN,
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()

    payload = resp.json()
    if "id" not in payload:
        raise RuntimeError(f"IG media container error: {payload}")

    container = payload["id"]
    print("IG publish: esperando container...", container)

    while True:
        status_resp = requests.get(
            f"{GRAPH_BASE}/{container}",
            params={"fields": "status_code", "access_token": IG_ACCESS_TOKEN},
            timeout=HTTP_TIMEOUT,
        )
        status_resp.raise_for_status()

        status_payload = status_resp.json()
        status = status_payload.get("status_code")
        print("IG status:", status)

        if status == "FINISHED":
            break

        if status in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"IG container failed: {status_payload}")

        __import__("time").sleep(3)

    print("IG publish: publicando...")

    publish_resp = requests.post(
        f"{GRAPH_BASE}/{IG_USER_ID}/media_publish",
        data={"creation_id": container, "access_token": IG_ACCESS_TOKEN},
        timeout=HTTP_TIMEOUT,
    )
    publish_resp.raise_for_status()

    result = publish_resp.json()
    if "id" not in result:
        raise RuntimeError(f"IG publish error: {result}")

    return result


def fb_publish(video_url, caption):
    print("FB reel upload START")

    start_resp = requests.post(
        f"{GRAPH_BASE}/{FB_PAGE_ID}/video_reels",
        data={"upload_phase": "START", "access_token": FB_PAGE_ACCESS_TOKEN},
        timeout=HTTP_TIMEOUT,
    )
    start_resp.raise_for_status()

    start = start_resp.json()
    if "upload_url" not in start or "video_id" not in start:
        raise RuntimeError(f"FB START error: {start}")

    upload_url = start["upload_url"]
    video_id = start["video_id"]

    print("FB reel upload TRANSFER")

    transfer_resp = requests.post(
        upload_url,
        headers={"Authorization": f"OAuth {FB_PAGE_ACCESS_TOKEN}", "file_url": video_url},
        timeout=HTTP_TIMEOUT,
    )
    transfer_resp.raise_for_status()

    try:
        transfer = transfer_resp.json()
    except Exception:
        transfer = {"raw": transfer_resp.text}

    print("FB transfer:", transfer)
    print("FB reel upload FINISH")

    finish_resp = requests.post(
        f"{GRAPH_BASE}/{FB_PAGE_ID}/video_reels",
        data={
            "upload_phase": "FINISH",
            "video_id": video_id,
            "video_state": "PUBLISHED",
            "description": caption,
            "access_token": FB_PAGE_ACCESS_TOKEN,
        },
        timeout=HTTP_TIMEOUT,
    )
    finish_resp.raise_for_status()

    finish = finish_resp.json()
    if not finish.get("success") and "post_id" not in finish:
        raise RuntimeError(f"FB FINISH error: {finish}")

    return finish


def youtube_publish(video_url, title, description):
    if not YOUTUBE_CLIENT_ID:
        raise RuntimeError("Falta YOUTUBE_CLIENT_ID")
    if not YOUTUBE_CLIENT_SECRET:
        raise RuntimeError("Falta YOUTUBE_CLIENT_SECRET")
    if not YOUTUBE_REFRESH_TOKEN:
        raise RuntimeError("Falta YOUTUBE_REFRESH_TOKEN")

    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    with tempfile.TemporaryDirectory() as td:
        local_path = os.path.join(td, "short.mp4")

        print("YT download temp desde R2:", video_url)
        resp = requests.get(video_url, timeout=120)
        resp.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(resp.content)

        creds = Credentials(
            None,
            refresh_token=YOUTUBE_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=YOUTUBE_CLIENT_ID,
            client_secret=YOUTUBE_CLIENT_SECRET,
            scopes=["https://www.googleapis.com/auth/youtube.upload"],
        )

        youtube = build("youtube", "v3", credentials=creds)

        body = {
            "snippet": {
                "title": title[:100],
                "description": description[:5000],
                "categoryId": "20",
                "tags": ["shorts", "gaming", "esports", "latam"],
            },
            "status": {
                "privacyStatus": YOUTUBE_PRIVACY_STATUS,
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(
            local_path,
            mimetype="video/mp4",
            resumable=True,
            chunksize=5 * 1024 * 1024,
        )

        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media,
        )

        response = None
        try:
            while response is None:
                status, response = request.next_chunk()
                if status:
                    print(f"YT upload progress: {int(status.progress() * 100)}%")
        except HttpError as e:
            msg = str(e)
            if "uploadLimitExceeded" in msg or "exceeded the number of videos" in msg:
                raise RuntimeError("YOUTUBE_UPLOAD_LIMIT_EXCEEDED")
            raise

        if "id" not in response:
            raise RuntimeError(f"YouTube upload error: {response}")

        return {
            "id": response["id"],
            "url": f"https://www.youtube.com/watch?v={response['id']}",
            "raw": response,
        }


def build_public_url(key):
    return f"{R2_PUBLIC_BASE_URL}/{key}"


def is_priority_key(key):
    return key.startswith(f"{PREFIX_PRIORITY}/")


def is_manual_key(key):
    return key.startswith(f"{PREFIX_MANUAL}/")


def allow_low_score(item):
    key = item["key"]
    if is_priority_key(key) and B_ALLOW_LOW_SCORE_FOR_PRIORITY:
        return True
    if is_manual_key(key) and B_ALLOW_LOW_SCORE_FOR_MANUAL:
        return True
    return False


def strong_copy_present(item):
    copy_signals = item.get("copy_signals") or {}
    return bool(clean_signal(copy_signals.get("hook")) and clean_signal(copy_signals.get("cta")))


def can_override_invalid_score(item):
    if is_priority_key(item["key"]) or is_manual_key(item["key"]):
        return True

    if is_weak_game_name(item.get("game_name")):
        return False

    score_reason = str(item.get("score_reason") or "").strip().lower()
    if score_reason not in ("nan", "missing", "non_numeric", "inf"):
        return False

    editorial_score = safe_float(item.get("editorial_score"), 0.0)
    if editorial_score < 3.5:
        return False

    copy_signals = item.get("copy_signals") or {}
    if not clean_signal(copy_signals.get("hook")):
        return False
    if not clean_signal(copy_signals.get("cta")):
        return False

    moment_type = normalize_moment_type(item.get("moment_type"))
    emotion = normalize_emotion(item.get("emotion"))

    has_editorial_signal = (
        moment_type in ("clutch", "ace", "final_circle", "goal", "last_second", "highlight")
        or emotion in ("skill", "clutch", "chaos", "insane", "tense", "hype", "clean")
    )

    if not has_editorial_signal:
        return False

    return True


def should_skip_for_score(item):
    score = safe_float(item.get("candidate_score"), 0.0)

    if allow_low_score(item):
        return False

    if can_override_invalid_score(item):
        print(
            "ALLOW invalid score override:",
            item["key"],
            "| score_reason:", item.get("score_reason"),
            "| editorial_score:", item.get("editorial_score"),
        )
        return False

    if B_ALLOW_STRONG_HOOK_OVERRIDE and strong_copy_present(item):
        if (
            item.get("editorial_score", 0.0) >= 4.5
            and score >= 0.50
            and not is_weak_game_name(item.get("game_name"))
        ):
            print(
                "ALLOW strong hook override:",
                item["key"],
                "| score:", score,
                "| editorial_score:", item.get("editorial_score"),
            )
            return False

    return score < B_MIN_CANDIDATE_SCORE


def should_skip_for_editorial_quality(item):
    key = item["key"]

    if is_priority_key(key) or is_manual_key(key):
        return False, None

    if B_BLOCK_WEAK_GENERIC_AUTO and is_weak_game_name(item.get("game_name")):
        if not (is_priority_key(key) or is_manual_key(key)):
            return True, "weak_game"

    if B_REQUIRE_EDITORIAL_SIGNALS:
        signals = 0

        if normalize_emotion(item.get("emotion")) in ("skill", "clutch", "chaos", "insane", "tense", "hype", "clean"):
            signals += 1

        if normalize_intensity(item.get("intensity")) in ("low", "medium", "high"):
            signals += 1

        if normalize_moment_type(item.get("moment_type")) in (
            "clutch", "ace", "final_circle", "goal", "last_second", "highlight"
        ):
            signals += 1

        copy_signals = item.get("copy_signals") or {}
        if clean_signal(copy_signals.get("hook")):
            signals += 1
        if clean_signal(copy_signals.get("cta")):
            signals += 1
        if clean_signal(copy_signals.get("caption_base")):
            signals += 1

        if signals == 0:
            return True, "weak_metadata"

    if item.get("editorial_score", 0.0) < B_EDITORIAL_SCORE_MIN:
        return True, f"low_editorial_score:{item.get('editorial_score')}"

    return False, None


def publish(item, target_platforms=None):
    if target_platforms is None:
        target_platforms = ["instagram", "facebook", "youtube_shorts"]

    key = item["key"]
    meta = item.get("meta") or {}
    brief = item.get("brief")
    public_url = build_public_url(key)

    caption = None
    shorts_title = None
    shorts_description = None

    copy_signals = item.get("copy_signals") or {}

    if isinstance(meta, dict):
        caption = clean_signal(meta.get("caption")) or clean_signal(copy_signals.get("caption"))
        shorts_title = clean_signal(meta.get("shorts_title")) or clean_signal(copy_signals.get("shorts_title"))
        shorts_description = clean_signal(meta.get("shorts_description")) or clean_signal(copy_signals.get("shorts_description"))

    if brief:
        print("BRIEF TXT DETECTADO -> modo campaign/priority")
        if not caption:
            caption = build_campaign_caption_from_brief(key, meta, brief)
        if not shorts_title:
            shorts_title = build_campaign_shorts_title_from_brief(key, meta, brief)
        if not shorts_description:
            shorts_description = build_campaign_shorts_description_from_brief(key, meta, brief)

    if not caption:
        caption = build_caption_from_meta_v6(key, meta, item)
    if not shorts_title:
        shorts_title = build_shorts_title_from_meta_v62(key, meta, item)
    if not shorts_description:
        shorts_description = build_shorts_description_from_meta_v62(key, meta, item)

    caption = ensure_game_in_caption(caption, item.get("game_name"))

    item["caption_final"] = caption
    item["shorts_title_final"] = shorts_title
    item["shorts_description_final"] = shorts_description

    print("PUBLICANDO VIDEO:")
    print("KEY:", key)
    print("URL:", public_url)
    print("SOURCE_GROUP:", item.get("source_group"))
    print("TARGET_PLATFORMS:", target_platforms)
    print("TXT KEY:", item.get("brief_txt_key"))
    print("BRIEF PARSED:", item.get("brief"))
    print("COPY SIGNALS:", item.get("copy_signals"))
    print("EDITORIAL SCORE:", item.get("editorial_score"))
    print("CAPTION:\n", caption)
    print("SHORTS TITLE:", shorts_title)

    results = {"instagram": None, "facebook": None, "youtube_shorts": None}

    if DRY_RUN:
        print("DRY_RUN activo: no se publica realmente")
        if "instagram" in target_platforms:
            results["instagram"] = {"ok": ENABLE_INSTAGRAM, "dry_run": True}
        if "facebook" in target_platforms:
            results["facebook"] = {"ok": ENABLE_FACEBOOK, "dry_run": True}
        if "youtube_shorts" in target_platforms:
            results["youtube_shorts"] = {"ok": ENABLE_SHORTS, "dry_run": True}
        print("Publicado OK\n")
        return results

    if ENABLE_INSTAGRAM and "instagram" in target_platforms:
        try:
            print("→ Publicando en Instagram...")
            ig = ig_publish(public_url, caption)
            print("IG OK:", ig)
            results["instagram"] = {"ok": True, "response": ig}
        except Exception as e:
            print("IG ERROR:", repr(e))
            results["instagram"] = {"ok": False, "error": str(e)}

    if ENABLE_FACEBOOK and "facebook" in target_platforms:
        try:
            print("→ Publicando en Facebook...")
            fb = fb_publish(public_url, caption)
            print("FB OK:", fb)
            results["facebook"] = {"ok": True, "response": fb}
        except Exception as e:
            print("FB ERROR:", repr(e))
            results["facebook"] = {"ok": False, "error": str(e)}

    if ENABLE_SHORTS and "youtube_shorts" in target_platforms:
        try:
            print("→ Publicando en YouTube Shorts...")
            yt = youtube_publish(public_url, shorts_title, shorts_description)
            print("YT OK:", yt)
            results["youtube_shorts"] = {"ok": True, "response": yt}
        except Exception as e:
            msg = str(e)
            if "YOUTUBE_UPLOAD_LIMIT_EXCEEDED" in msg:
                print("YT SKIP CONTROLADO: upload limit exceeded")
                results["youtube_shorts"] = {"ok": False, "error": "upload_limit_exceeded", "controlled": True}
            else:
                print("YT ERROR:", repr(e))
                results["youtube_shorts"] = {"ok": False, "error": str(e)}

    print("Publicado OK\n")
    return results


def should_process_item(st, item):
    key = item["key"]
    source_group = item.get("source_group")
    wanted = []

    if ENABLE_INSTAGRAM and not is_published_on(st, "instagram", key):
        if not (B_BLOCK_IF_SOURCE_ALREADY_PUBLISHED and is_source_group_published_on(st, "instagram", source_group)):
            wanted.append("instagram")

    if ENABLE_FACEBOOK and not is_published_on(st, "facebook", key):
        if not (B_BLOCK_IF_SOURCE_ALREADY_PUBLISHED and is_source_group_published_on(st, "facebook", source_group)):
            wanted.append("facebook")

    if ENABLE_SHORTS and not is_published_on(st, "youtube_shorts", key):
        if not (B_BLOCK_IF_SOURCE_ALREADY_PUBLISHED and is_source_group_published_on(st, "youtube_shorts", source_group)):
            wanted.append("youtube_shorts")

    return len(wanted) > 0, wanted


def build_item_from_key(key):
    meta_key, final_meta = load_meta_for_video_key(key)
    txt_key, brief_raw_text, brief = load_and_parse_sidecar_brief(key)

    source_clip_key = None
    if isinstance(final_meta, dict):
        source_clip_key = final_meta.get("source_clip_key")

    clip_meta_key = None
    clip_meta = None
    if source_clip_key:
        clip_meta_key, clip_meta = load_clip_meta_from_source_clip_key(source_clip_key)

    meta = merge_meta_with_clip_fallback(final_meta or {}, clip_meta or {})

    source_group = resolve_source_group(key, meta)
    game_name = resolve_game_name(key, meta)

    candidate_score_raw = meta.get("candidate_score")
    emotion = meta.get("emotion")
    intensity = meta.get("intensity")
    moment_type = meta.get("moment_type")

    score_info = sanitize_candidate_score(candidate_score_raw)
    candidate_score = score_info["score"]
    score_valid = score_info["valid"]
    score_reason = score_info["reason"]

    if is_priority_key(key) and brief:
        candidate_score = max(candidate_score, 999.0)
        score_valid = True
        score_reason = "priority_override"

    copy_signals = extract_copy_signals(meta)

    if is_weak_text(emotion):
        emotion = copy_signals.get("emotion") or emotion
    if is_weak_text(intensity):
        intensity = copy_signals.get("intensity") or intensity
    if is_weak_text(moment_type):
        moment_type = copy_signals.get("moment_type") or moment_type
    if is_weak_game_name(game_name) and copy_signals.get("game"):
        game_name = normalize_game_name(copy_signals.get("game"))

    emotion = normalize_emotion(emotion)
    intensity = normalize_intensity(intensity)
    moment_type = normalize_moment_type(moment_type)

    item = {
        "key": key,
        "meta_key": meta_key,
        "clip_meta_key": clip_meta_key,
        "meta": meta or {},
        "final_meta_raw": final_meta or {},
        "clip_meta_raw": clip_meta or {},
        "brief_txt_key": txt_key if brief_raw_text else None,
        "brief_text_raw": brief_raw_text,
        "brief": brief,
        "source_group": source_group,
        "game_name": game_name,
        "candidate_score": candidate_score,
        "candidate_score_raw": score_info["raw"],
        "score_valid": score_valid,
        "score_reason": score_reason,
        "emotion": emotion,
        "intensity": intensity,
        "moment_type": moment_type,
        "copy_signals": copy_signals,
        "fallback_used": bool(clip_meta),
    }

    item["editorial_score"] = compute_editorial_score(item)
    return item


def sort_queue_items(items):
    def priority_rank(item):
        key = item["key"]
        if is_priority_key(key):
            return 3
        if is_manual_key(key):
            return 2
        return 1

    return sorted(
        items,
        key=lambda x: (
            priority_rank(x),
            safe_float(x.get("editorial_score"), 0.0),
            safe_float(x.get("candidate_score"), 0.0),
            x["key"],
        ),
        reverse=True,
    )


def run_mode_b():
    print("===== MODE B (PUBLISHER) START =====")
    print("MODE B VERSION: V6_3_CONFLICT_ENGINE_OPENING_LOCK")
    print("B_MAX_PUBLISH_PER_RUN:", B_MAX_PUBLISH_PER_RUN)
    print("B_MAX_PER_SOURCE_GROUP_PER_RUN:", B_MAX_PER_SOURCE_GROUP_PER_RUN)
    print("B_AVOID_SAME_SOURCE_PER_RUN:", B_AVOID_SAME_SOURCE_PER_RUN)
    print("B_BLOCK_IF_SOURCE_ALREADY_PUBLISHED:", B_BLOCK_IF_SOURCE_ALREADY_PUBLISHED)
    print("B_MIN_CANDIDATE_SCORE:", B_MIN_CANDIDATE_SCORE)
    print("B_BLOCK_WEAK_GENERIC_AUTO:", B_BLOCK_WEAK_GENERIC_AUTO)
    print("B_REQUIRE_EDITORIAL_SIGNALS:", B_REQUIRE_EDITORIAL_SIGNALS)
    print("B_EDITORIAL_SCORE_MIN:", B_EDITORIAL_SCORE_MIN)
    print("B_ALLOW_STRONG_HOOK_OVERRIDE:", B_ALLOW_STRONG_HOOK_OVERRIDE)
    print("B_CAPTION_MAX_WORDS:", B_CAPTION_MAX_WORDS)
    print("B_CAPTION_MIN_SCORE:", B_CAPTION_MIN_SCORE)
    print("B_USE_OPENAI_POLISH:", B_USE_OPENAI_POLISH)
    print("B_OPENAI_POLISH_MIN_EDITORIAL:", B_OPENAI_POLISH_MIN_EDITORIAL)
    print("EDITORIAL_MEMORY_SUMMARY_KEY:", EDITORIAL_MEMORY_SUMMARY_KEY)
    print("B_ONLY_KEYS_CONTAIN:", B_ONLY_KEYS_CONTAIN or "(vacío)")
    print("PREFIX_PRIORITY:", PREFIX_PRIORITY)
    print("PREFIX_MANUAL:", PREFIX_MANUAL)
    print("PREFIX_AUTO:", PREFIX_AUTO)
    print("META_FINAL_PREFIX:", META_FINAL_PREFIX)
    print("STATE_KEY:", STATE_KEY)
    print("DRY_RUN:", DRY_RUN)
    print("ENABLE_INSTAGRAM:", ENABLE_INSTAGRAM)
    print("ENABLE_FACEBOOK:", ENABLE_FACEBOOK)
    print("ENABLE_SHORTS:", ENABLE_SHORTS)
    print("YOUTUBE_PRIVACY_STATUS:", YOUTUBE_PRIVACY_STATUS)

    state = load_state()

    priority = list_keys(PREFIX_PRIORITY)
    manual = list_keys(PREFIX_MANUAL)
    auto = list_keys(PREFIX_AUTO)

    print("Priority:", len(priority))
    print("Manual:", len(manual))
    print("Auto:", len(auto))
    print("Published IG:", len(state["published"]["instagram"]))
    print("Published FB:", len(state["published"]["facebook"]))
    print("Published YT:", len(state["published"]["youtube_shorts"]))

    raw_queue = []
    raw_queue.extend(priority)
    raw_queue.extend(manual)
    raw_queue.extend(auto)

    if B_ONLY_KEYS_CONTAIN:
        raw_queue = [k for k in raw_queue if B_ONLY_KEYS_CONTAIN in k]
        print("Queue tras filtro:", len(raw_queue))

    queue_items = []
    for key in raw_queue:
        try:
            item = build_item_from_key(key)
            queue_items.append(item)
        except Exception as e:
            print("ERROR armando item de queue:", key, repr(e))

    queue_items = sort_queue_items(queue_items)
    queue_items = diversify_queue(queue_items)

    print("Queue final diversificada:", len(queue_items))

    count = 0
    used_source_groups_this_run = {}

    for item in queue_items:
        if count >= B_MAX_PUBLISH_PER_RUN:
            break

        key = item["key"]
        source_group = item.get("source_group") or "unknown"

        if B_AVOID_SAME_SOURCE_PER_RUN:
            current_count = used_source_groups_this_run.get(source_group, 0)
            if current_count >= B_MAX_PER_SOURCE_GROUP_PER_RUN:
                print("SKIP max source group per run:", source_group, "|", key)
                append_skip(
                    state,
                    {
                        "skipped_at": iso_now_full(),
                        "key": key,
                        "source_group": source_group,
                        "reason": "max_source_group_per_run",
                    },
                )
                continue

        if should_skip_for_score(item):
            print(
                "SKIP low candidate_score:",
                key,
                "| score:", item.get("candidate_score"),
                "| min:", B_MIN_CANDIDATE_SCORE,
                "| editorial_score:", item.get("editorial_score"),
            )
            append_skip(
                state,
                {
                    "skipped_at": iso_now_full(),
                    "key": key,
                    "source_group": source_group,
                    "reason": "low_candidate_score",
                    "candidate_score": item.get("candidate_score"),
                    "candidate_score_raw": item.get("candidate_score_raw"),
                    "score_valid": item.get("score_valid"),
                    "score_reason": item.get("score_reason"),
                    "editorial_score": item.get("editorial_score"),
                    "game_name": item.get("game_name"),
                },
            )
            continue

        skip_editorial, editorial_reason = should_skip_for_editorial_quality(item)
        if skip_editorial:
            print("SKIP editorial quality:", key, "|", editorial_reason)
            append_skip(
                state,
                {
                    "skipped_at": iso_now_full(),
                    "key": key,
                    "source_group": source_group,
                    "reason": editorial_reason,
                    "candidate_score": item.get("candidate_score"),
                    "candidate_score_raw": item.get("candidate_score_raw"),
                    "score_valid": item.get("score_valid"),
                    "score_reason": item.get("score_reason"),
                    "editorial_score": item.get("editorial_score"),
                    "game_name": item.get("game_name"),
                    "emotion": item.get("emotion"),
                    "intensity": item.get("intensity"),
                    "moment_type": item.get("moment_type"),
                    "copy_signals": item.get("copy_signals"),
                    "fallback_used": item.get("fallback_used"),
                },
            )
            continue

        process, missing_platforms = should_process_item(state, item)
        if not process:
            print("SKIP already published / blocked by source group:", key)
            append_skip(
                state,
                {
                    "skipped_at": iso_now_full(),
                    "key": key,
                    "source_group": source_group,
                    "reason": "already_published_or_source_group_blocked",
                },
            )
            continue

        print("Procesando:", key)
        print("SOURCE GROUP:", source_group)
        print("GAME DETECTED:", item.get("game_name"))
        print("CANDIDATE SCORE:", item.get("candidate_score"))
        print("CANDIDATE SCORE RAW:", item.get("candidate_score_raw"))
        print("SCORE VALID:", item.get("score_valid"))
        print("SCORE REASON:", item.get("score_reason"))
        print("EMOTION:", item.get("emotion"))
        print("INTENSITY:", item.get("intensity"))
        print("MOMENT TYPE:", item.get("moment_type"))
        print("EDITORIAL SCORE:", item.get("editorial_score"))
        print("FALLBACK USED:", item.get("fallback_used"))
        print("COPY SIGNALS:", item.get("copy_signals"))
        print("META KEY:", item.get("meta_key"))
        print("CLIP META KEY:", item.get("clip_meta_key"))
        print("TXT KEY:", item.get("brief_txt_key"))
        print("MISSING PLATFORMS:", missing_platforms)

        try:
            result = publish(item, target_platforms=missing_platforms)

            success_any = False

            if "instagram" in missing_platforms and result.get("instagram", {}).get("ok"):
                mark_published_on(state, "instagram", key)
                mark_source_group_published_on(state, "instagram", source_group)
                success_any = True

            if "facebook" in missing_platforms and result.get("facebook", {}).get("ok"):
                mark_published_on(state, "facebook", key)
                mark_source_group_published_on(state, "facebook", source_group)
                success_any = True

            if "youtube_shorts" in missing_platforms and result.get("youtube_shorts", {}).get("ok"):
                mark_published_on(state, "youtube_shorts", key)
                mark_source_group_published_on(state, "youtube_shorts", source_group)
                success_any = True

            append_history(
                state,
                {
                    "published_at": iso_now_full(),
                    "key": key,
                    "source_group": source_group,
                    "candidate_score": item.get("candidate_score"),
                    "candidate_score_raw": item.get("candidate_score_raw"),
                    "score_valid": item.get("score_valid"),
                    "score_reason": item.get("score_reason"),
                    "editorial_score": item.get("editorial_score"),
                    "game_name": item.get("game_name"),
                    "emotion": item.get("emotion"),
                    "intensity": item.get("intensity"),
                    "moment_type": item.get("moment_type"),
                    "copy_signals": item.get("copy_signals"),
                    "fallback_used": item.get("fallback_used"),
                    "editorial_mode": "v6.3_conflict_engine_opening_lock",
                    "brief_txt_key": item.get("brief_txt_key"),
                    "brief": item.get("brief"),
                    "caption_final": item.get("caption_final"),
                    "shorts_title_final": item.get("shorts_title_final"),
                    "shorts_description_final": item.get("shorts_description_final"),
                    "caption_engine_debug": item.get("caption_engine_debug"),
                    "platform_results": result,
                },
            )

            save_state(state)

            if success_any:
                used_source_groups_this_run[source_group] = used_source_groups_this_run.get(source_group, 0) + 1
                count += 1

        except Exception as e:
            print("ERROR publicando:", repr(e))
            append_skip(
                state,
                {
                    "skipped_at": iso_now_full(),
                    "key": key,
                    "source_group": source_group,
                    "reason": "publish_exception",
                    "error": str(e),
                },
            )

    save_state(state)

    print("Publicados en esta corrida:", count)
    print("===== MODE B DONE =====")


if __name__ == "__main__":
    run_mode_b()
