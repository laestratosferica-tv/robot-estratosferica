# ===== INICIO: ugc_mode_b.py =====

import os
import re
import json
import math
import tempfile
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
B_ONLY_KEYS_CONTAIN = env_nonempty("B_ONLY_KEYS_CONTAIN", "")
B_AVOID_SAME_SOURCE_PER_RUN = env_bool("B_AVOID_SAME_SOURCE_PER_RUN", True)
B_BLOCK_IF_SOURCE_ALREADY_PUBLISHED = env_bool("B_BLOCK_IF_SOURCE_ALREADY_PUBLISHED", True)

# nuevo: filtro mínimo de score
B_MIN_CANDIDATE_SCORE = env_float("B_MIN_CANDIDATE_SCORE", 0.55)
B_ALLOW_LOW_SCORE_FOR_PRIORITY = env_bool("B_ALLOW_LOW_SCORE_FOR_PRIORITY", True)
B_ALLOW_LOW_SCORE_FOR_MANUAL = env_bool("B_ALLOW_LOW_SCORE_FOR_MANUAL", True)

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
    return load_json(clip_meta_key)


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
            clip_meta = load_clip_meta_from_source_clip_key(source_clip_key)
            if isinstance(clip_meta, dict):
                sg2 = clip_meta.get("source_group") or clip_meta.get("source_video_id")
                if sg2:
                    return str(sg2)

    return extract_source_group_from_key(key)


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
        ("fc 26", "EA Sports FC"),
        ("fc 25", "EA Sports FC"),
        ("fc pro", "EA Sports FC"),
        ("vejrgang", "EA Sports FC"),
        ("tekkz", "EA Sports FC"),
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

    return "Esports"


def resolve_game_name(key, meta):
    if isinstance(meta, dict):
        for field in ["game", "game_name"]:
            value = meta.get(field)
            if value:
                return normalize_game_name(value)

        source_clip_key = meta.get("source_clip_key")
        if source_clip_key:
            clip_meta = load_clip_meta_from_source_clip_key(source_clip_key)
            if isinstance(clip_meta, dict):
                for field in ["game", "game_name"]:
                    value = clip_meta.get(field)
                    if value:
                        return normalize_game_name(value)

    return detect_game_from_key(key)


def normalize_game_name(name):
    t = str(name or "").strip().lower()

    mapping = {
        "granturismo": "Gran Turismo",
        "gran turismo": "Gran Turismo",
        "easportsfc": "EA Sports FC",
        "ea sports fc": "EA Sports FC",
        "leagueoflegends": "League of Legends",
        "league of legends": "League of Legends",
        "apex": "Apex Legends",
        "apex legends": "Apex Legends",
        "fortnite": "Fortnite",
        "minecraft": "Minecraft",
        "cs2": "CS2",
        "valorant": "Valorant",
        "warzone": "Warzone",
        "f1": "F1",
        "esports": "Esports",
        "generic": "Esports",
    }

    return mapping.get(t, str(name))


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


# ========= COPY POR JUEGO =========

GAME_HOOKS = {
    "Valorant": [
        "Esto no fue un round normal.",
        "Hay rounds que te cambian el mapa entero.",
        "Esto en Valorant no se regala.",
    ],
    "CS2": [
        "Esto fue puro timing y sangre fría.",
        "Hay clips buenos y luego está esto.",
        "Si pestañeaste, te lo perdiste.",
    ],
    "League of Legends": [
        "Hay peleas que cambian una serie entera.",
        "Esto es LoL en modo cine competitivo.",
        "Aquí hubo lectura total del momento.",
    ],
    "Fortnite": [
        "Esto no se gana normalmente.",
        "Hay mecánicas buenas y luego está esto.",
        "Fortnite también sabe dar puro cine.",
    ],
    "Warzone": [
        "Esto no era un cierre normal.",
        "Warzone cuando se pone serio se vuelve caos puro.",
        "Aquí hubo lectura, control y cero pánico.",
    ],
    "Apex Legends": [
        "Esto no fue solo aim.",
        "Apex en su versión más salvaje.",
        "Aquí hubo lectura total del caos.",
    ],
    "Minecraft": [
        "Minecraft también puede ser puro cine.",
        "Esto no tenía ningún sentido y aun así pasó.",
        "Hay momentos que parecen editados… pero no.",
    ],
    "EA Sports FC": [
        "Hay goles buenos y luego está esta locura.",
        "Esto en FC te pone a pelear en comentarios.",
        "No me digas que esto es una jugada normal.",
    ],
    "F1": [
        "Esto fue una barbaridad de precisión.",
        "Hay maniobras limpias y luego está esta locura.",
        "F1 cuando se pone serio es arte puro.",
    ],
    "Gran Turismo": [
        "Esto no es solo velocidad: es precisión pura.",
        "Hay vueltas rápidas y luego están las que se ganan con cabeza fría.",
        "Gran Turismo cuando se pone serio parece carrera real.",
    ],
    "Esports": [
        "Esto no fue una jugada cualquiera.",
        "Hay clips que entretienen y otros que hacen comunidad.",
        "Esto es exactamente por lo que seguimos viendo esports.",
    ],
}

GAME_CONTEXTS = {
    "Valorant": [
        "En Valorant, una play así te cambia el mood de toda la partida.",
        "Esto en competitivo no perdona: o lo lees bien o te pasan por encima.",
        "Una secuencia así es la que separa el clip bueno del clip serio.",
    ],
    "CS2": [
        "En CS2, una decisión así vale más que mil highlights vacíos.",
        "Esto es lo que pasa cuando aim, timing y cabeza se alinean.",
        "Si entiendes CS2, sabes por qué esta jugada pesa tanto.",
    ],
    "League of Legends": [
        "En LoL, una sola secuencia puede romper todo el mapa.",
        "Este tipo de momento es el que termina marcando series enteras.",
        "Hay plays que no solo ganan la pelea: cambian la narrativa.",
    ],
    "Fortnite": [
        "En Fortnite, una mecánica así no solo se aplaude: se discute.",
        "Esto es exactamente el tipo de clip que prende a toda la escena.",
        "Hay finales buenos, pero esta clase de jugada se queda en la cabeza.",
    ],
    "Warzone": [
        "Warzone tiene caos, sí, pero esto ya fue lectura de élite.",
        "Aquí no fue solo suerte: hubo timing, decisión y sangre fría.",
        "Este tipo de clip es el que pone a debatir a toda la escena.",
    ],
    "Apex Legends": [
        "Apex recompensa a los que leen el caos mejor que nadie.",
        "Este tipo de momento define por qué Apex sigue siendo tan adictivo de ver.",
        "Aquí hubo más que aim: hubo lectura completa de la situación.",
    ],
    "Minecraft": [
        "Sí, Minecraft también puede dejar momentos absurdamente buenos.",
        "Esto parece meme, pero justamente por eso funciona tanto.",
        "Cuando Minecraft regala una escena así, internet hace su trabajo solo.",
    ],
    "EA Sports FC": [
        "En FC, una jugada así te pone a discutir con cualquiera.",
        "Esto es justo el tipo de acción que parte a la comunidad en dos.",
        "Hay goles que valen uno. Este vale conversación toda la semana.",
    ],
    "F1": [
        "En F1, una maniobra así no se regala: se gana.",
        "Esto es precisión pura con presión máxima.",
        "Este tipo de adelantamiento explica por qué la escena engancha tanto.",
    ],
    "Gran Turismo": [
        "En simracing, una diferencia mínima decide todo.",
        "Aquí no hubo caos: hubo control fino, lectura y paciencia.",
        "Esto es de esos momentos donde Gran Turismo se siente más carrera que juego.",
    ],
    "Esports": [
        "Este tipo de clip es el que hace que la gente vuelva a comentar.",
        "Aquí no hubo relleno: solo una jugada que merece conversación.",
        "Esto es justo el tipo de momento que construye comunidad.",
    ],
}

GAME_CTAS = {
    "Valorant": [
        "¿Esto fue puro skill o lectura del error rival?",
        "¿Play de élite o defensa desastrosa?",
        "¿Tú lo resuelves así o te comen vivo?",
    ],
    "CS2": [
        "¿Esto fue aim puro o IQ de otro planeta?",
        "¿Skill real o regalo del rival?",
        "¿Top play o la están inflando demasiado?",
    ],
    "League of Legends": [
        "¿Play histórica o la defensa regaló demasiado?",
        "¿Esto fue macro, manos o puro caos bien aprovechado?",
        "¿Tú dirías que aquí se ganó mentalmente?",
    ],
    "Fortnite": [
        "¿Esto fue locura mecánica o puro caos bien aprovechado?",
        "¿La mejor play o una de esas que se inflan por el replay?",
        "¿Esto es top o no tanto?",
    ],
    "Warzone": [
        "¿Esto fue puro control o caos favorable?",
        "¿Skill brutal o suerte con esteroides?",
        "¿Tú lo llamas clutch o milagro?",
    ],
    "Apex Legends": [
        "¿Esto fue lectura de élite o error del lobby?",
        "¿Top play o exageración de clip corto?",
        "¿Apex puro o fortuna demasiado conveniente?",
    ],
    "Minecraft": [
        "¿Esto fue genialidad o casualidad total?",
        "¿Play real o clip maldito de internet?",
        "¿Tú le llamas skill o meme perfecto?",
    ],
    "EA Sports FC": [
        "¿Golazo puro o defensa de plastilina?",
        "¿Esto es top mundial o el rival ayudó demasiado?",
        "¿Tú a esto le llamas clase o puro abuso del juego?",
    ],
    "F1": [
        "¿Maniobra legendaria o riesgo innecesario?",
        "¿Esto es talento puro o el rival dejó la puerta abierta?",
        "¿Top adelantamiento o mucho hype para tan poco?",
    ],
    "Gran Turismo": [
        "¿Manejo limpio o error del rival?",
        "¿Precisión total o la están inflando de más?",
        "¿Esto es clase real o solo se ve bonito?",
    ],
    "Esports": [
        "¿Skill puro o también hubo bastante fortuna?",
        "¿Top play o clip inflado por el contexto?",
        "¿Esto merece hype real o la están vendiendo demasiado?",
    ],
}

GAME_HASHTAGS = {
    "Valorant": ["#Valorant", "#VCT", "#EsportsLATAM", "#GamingLATAM", "#ValorantLATAM"],
    "CS2": ["#CS2", "#CounterStrike", "#EsportsLATAM", "#GamingLATAM", "#CS2LATAM"],
    "League of Legends": ["#LeagueOfLegends", "#LoL", "#EsportsLATAM", "#GamingLATAM", "#LoLLATAM"],
    "Fortnite": ["#Fortnite", "#FortniteLATAM", "#EsportsLATAM", "#GamingLATAM", "#FortniteClips"],
    "Warzone": ["#Warzone", "#CallOfDuty", "#EsportsLATAM", "#GamingLATAM", "#WarzoneLATAM"],
    "Apex Legends": ["#ApexLegends", "#Apex", "#EsportsLATAM", "#GamingLATAM", "#ApexLATAM"],
    "Minecraft": ["#Minecraft", "#MinecraftLATAM", "#GamingLATAM", "#MinecraftClips", "#ReelsGaming"],
    "EA Sports FC": ["#EASportsFC", "#FC", "#FCLATAM", "#GamingLATAM", "#EsportsLATAM"],
    "F1": ["#F1", "#SimRacing", "#EsportsLATAM", "#GamingLATAM", "#F1Esports"],
    "Gran Turismo": ["#GranTurismo", "#SimRacing", "#GTLATAM", "#GamingLATAM", "#ReelsGaming"],
    "Esports": ["#EsportsLATAM", "#GamingLATAM", "#Esports", "#Gaming", "#ReelsGaming"],
}


def pick_from_map(mapping, key_name, fallback="Esports"):
    options = mapping.get(key_name) or (mapping.get(fallback) if fallback is not None else None) or []
    if not options:
        return ""
    import random
    return random.choice(options)


def safe_float(v, default=0.0):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def openai_text(prompt):
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,
        "input": prompt,
    }

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


def build_campaign_caption_from_brief(key, meta, brief):
    game_name = brief.get("game") or resolve_game_name(key, meta) or "gaming"

    prompt = f"""
Eres editor viral premium de gaming y esports LATAM.

Escribe un caption para una pieza PRIORITY.
Debe sonar:
- potente
- publicable
- emocional
- nada genérico
- no robótico
- si es sponsor, que se sienta premium

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

Devuelve SOLO el caption final.
Reglas:
- 1 hook fuerte al inicio
- 2 a 4 líneas de desarrollo
- 1 CTA o pregunta final
- 5 a 8 hashtags
- máximo 120 palabras
"""
    try:
        if OPENAI_API_KEY:
            text = openai_text(prompt).strip()
            if text:
                return text
    except Exception as e:
        print("OpenAI brief caption fallback:", repr(e))

    hashtags = []
    if brief.get("campaign"):
        hashtags.append(f"#{str(brief['campaign']).replace(' ', '')}")
    if game_name:
        hashtags.append(f"#{str(game_name).replace(' ', '')}")
    hashtags.extend(["#GamingLATAM", "#EsportsLATAM", "#ReelsGaming"])

    hook = brief.get("hook") or "Esto se ve demasiado brutal para ignorarlo."
    angle = brief.get("angle") or "pieza premium con potencial de conversación"
    cta = brief.get("cta") or "¿Tu setup aguanta esto o ya toca upgrade?"

    return f"""{hook}

{angle}

🔥 {cta}

{" ".join(hashtags[:7])}""".strip()


def build_campaign_shorts_title_from_brief(key, meta, brief):
    game_name = brief.get("game") or resolve_game_name(key, meta) or "Gaming"

    prompt = f"""
Eres editor de Shorts gaming LATAM.

Crea un título corto, fuerte y premium.

Archivo: {os.path.basename(key)}
Juego: {game_name}
Campaign: {brief.get('campaign') or ''}
Angle: {brief.get('angle') or ''}
Hook idea: {brief.get('hook') or ''}
Style: {brief.get('style') or ''}
Intensity: {brief.get('intensity') or ''}

Reglas:
- máximo 80 caracteres antes de #Shorts
- devuelve solo el título final
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

    base = brief.get("hook") or f"{game_name} se ve RIDÍCULO"
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

    angle = brief.get("angle") or "pieza premium gaming"
    cta = brief.get("cta") or "¿La montarías en tu setup?"

    return f"""{angle}

{cta}

{" ".join(hashtags[:7])}""".strip()


def build_caption_from_meta(key, meta, item):
    game_name = resolve_game_name(key, meta)
    score = safe_float(item.get("candidate_score"), 0.0)
    emotion = str(item.get("emotion") or "").strip().lower()
    intensity = str(item.get("intensity") or "").strip().lower()
    moment_type = str(item.get("moment_type") or "").strip().lower()

    # si hay OpenAI, mejoramos mucho el tono
    if OPENAI_API_KEY:
        prompt = f"""
Eres editor gamer LATAM de Estratosférica TV.

Escribe caption para reel corto.
No sonar genérico.
No sonar corporativo.
No sonar como bot.

Datos:
Juego: {game_name}
Score: {score}
Emotion: {emotion}
Intensity: {intensity}
Moment type: {moment_type}
Archivo: {os.path.basename(key)}

Reglas:
- 1 hook fuerte
- 1 o 2 líneas de contexto
- 1 pregunta final
- 4 a 6 hashtags
- máximo 85 palabras
- respetar el lenguaje del juego
- si es Gran Turismo o F1, sonar a precisión/simracing
- si es Minecraft, sonar más internet/caos/momento absurdo
- si es shooter, sonar más clutch/lectura/skill
Devuelve solo el caption final.
"""
        try:
            text = openai_text(prompt).strip()
            if text:
                return text
        except Exception as e:
            print("OpenAI caption fallback:", repr(e))

    hook = pick_from_map(GAME_HOOKS, game_name)
    context = pick_from_map(GAME_CONTEXTS, game_name)
    cta = pick_from_map(GAME_CTAS, game_name)
    hashtags = " ".join(GAME_HASHTAGS.get(game_name, GAME_HASHTAGS["Esports"]))

    if score >= 0.82:
        context = context.replace(".", " de verdad.", 1) if "." in context else context

    if emotion == "clutch" and game_name in ("CS2", "Valorant", "Apex Legends", "Fortnite", "Warzone"):
        cta = "¿Esto fue clutch puro o lectura total del rival?"
    elif intensity == "low" and game_name == "Gran Turismo":
        cta = "¿Limpieza total o lo estás viendo demasiado bonito?"
    elif intensity == "low" and game_name == "Minecraft":
        cta = "¿Esto fue skill real o puro momento maldito?"

    return f"""{hook}

{context}

🔥 {cta}

{hashtags}""".strip()


def build_shorts_title_from_meta(key, meta, item):
    game_name = resolve_game_name(key, meta)
    score = safe_float(item.get("candidate_score"), 0.0)
    moment_type = str(item.get("moment_type") or "").strip().lower()

    if OPENAI_API_KEY:
        prompt = f"""
Crea título para YouTube Shorts gamer LATAM.

Juego: {game_name}
Score: {score}
Moment type: {moment_type}
Archivo: {os.path.basename(key)}

Reglas:
- corto
- potente
- no genérico
- máximo 75 caracteres antes de #Shorts
- devolver solo el título
"""
        try:
            text = openai_text(prompt).strip().replace('"', "").strip()
            if text:
                if "#Shorts" not in text:
                    text = f"{text} #Shorts"
                return text[:100]
        except Exception as e:
            print("OpenAI shorts title fallback:", repr(e))

    if game_name == "Gran Turismo":
        title = "Gran Turismo en modo precisión total"
    elif game_name == "Minecraft":
        title = "Minecraft acaba de regalar puro cine"
    elif game_name == "EA Sports FC":
        title = "Esto en FC te parte la comunidad"
    else:
        title = pick_from_map(GAME_HOOKS, game_name)

    if "#Shorts" not in title:
        title = f"{title} #Shorts"
    return title[:100]


def build_shorts_description_from_meta(key, meta, item):
    game_name = resolve_game_name(key, meta)
    score = safe_float(item.get("candidate_score"), 0.0)
    emotion = str(item.get("emotion") or "").strip().lower()
    intensity = str(item.get("intensity") or "").strip().lower()

    if OPENAI_API_KEY:
        prompt = f"""
Escribe descripción para YouTube Shorts gamer LATAM.

Juego: {game_name}
Score: {score}
Emotion: {emotion}
Intensity: {intensity}
Archivo: {os.path.basename(key)}

Reglas:
- 2 líneas + hashtags
- no sonar genérico
- respetar el juego
- máximo 350 caracteres
- devolver solo el texto
"""
        try:
            text = openai_text(prompt).strip()
            if text:
                return text[:5000]
        except Exception as e:
            print("OpenAI shorts description fallback:", repr(e))

    context = pick_from_map(GAME_CONTEXTS, game_name)
    cta = pick_from_map(GAME_CTAS, game_name)
    hashtags = " ".join(GAME_HASHTAGS.get(game_name, GAME_HASHTAGS["Esports"]))

    return f"""{context}

{cta}

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
            params={
                "fields": "status_code",
                "access_token": IG_ACCESS_TOKEN,
            },
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
        data={
            "creation_id": container,
            "access_token": IG_ACCESS_TOKEN,
        },
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
        data={
            "upload_phase": "START",
            "access_token": FB_PAGE_ACCESS_TOKEN,
        },
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
        headers={
            "Authorization": f"OAuth {FB_PAGE_ACCESS_TOKEN}",
            "file_url": video_url,
        },
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
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"YT upload progress: {int(status.progress() * 100)}%")

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


def should_skip_for_score(item):
    score = safe_float(item.get("candidate_score"), 0.0)
    if allow_low_score(item):
        return False
    return score < B_MIN_CANDIDATE_SCORE


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

    if isinstance(meta, dict):
        caption = meta.get("caption")
        shorts_title = meta.get("shorts_title")
        shorts_description = meta.get("shorts_description")

    if brief:
        print("BRIEF TXT DETECTADO -> modo campaign/priority")
        if not caption:
            caption = build_campaign_caption_from_brief(key, meta, brief)
        if not shorts_title:
            shorts_title = build_campaign_shorts_title_from_brief(key, meta, brief)
        if not shorts_description:
            shorts_description = build_campaign_shorts_description_from_brief(key, meta, brief)

    if not caption:
        caption = build_caption_from_meta(key, meta, item)
    if not shorts_title:
        shorts_title = build_shorts_title_from_meta(key, meta, item)
    if not shorts_description:
        shorts_description = build_shorts_description_from_meta(key, meta, item)

    print("PUBLICANDO VIDEO:")
    print("KEY:", key)
    print("URL:", public_url)
    print("SOURCE_GROUP:", item.get("source_group"))
    print("TARGET_PLATFORMS:", target_platforms)
    print("TXT KEY:", item.get("brief_txt_key"))
    print("BRIEF PARSED:", item.get("brief"))
    print("CAPTION:\n", caption)
    print("SHORTS TITLE:", shorts_title)

    results = {
        "instagram": None,
        "facebook": None,
        "youtube_shorts": None,
    }

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
    meta_key, meta = load_meta_for_video_key(key)
    txt_key, brief_raw_text, brief = load_and_parse_sidecar_brief(key)

    source_group = resolve_source_group(key, meta)
    game_name = resolve_game_name(key, meta)

    candidate_score = None
    emotion = None
    intensity = None
    moment_type = None

    if isinstance(meta, dict):
        candidate_score = meta.get("candidate_score")
        emotion = meta.get("emotion")
        intensity = meta.get("intensity")
        moment_type = meta.get("moment_type")

    if isinstance(meta, dict):
        source_clip_key = meta.get("source_clip_key")
        if source_clip_key:
            clip_meta = load_clip_meta_from_source_clip_key(source_clip_key)
            if isinstance(clip_meta, dict):
                if candidate_score is None:
                    candidate_score = clip_meta.get("candidate_score")
                if emotion is None:
                    emotion = clip_meta.get("emotion")
                if intensity is None:
                    intensity = clip_meta.get("intensity")
                if moment_type is None:
                    moment_type = clip_meta.get("moment_type")

    candidate_score = safe_float(candidate_score, 0.0)

    if is_priority_key(key) and brief:
        candidate_score = max(candidate_score, 999.0)

    return {
        "key": key,
        "meta_key": meta_key,
        "meta": meta or {},
        "brief_txt_key": txt_key if brief_raw_text else None,
        "brief_text_raw": brief_raw_text,
        "brief": brief,
        "source_group": source_group,
        "game_name": game_name,
        "candidate_score": candidate_score,
        "emotion": emotion,
        "intensity": intensity,
        "moment_type": moment_type,
    }


def sort_queue_items(items):
    return sorted(
        items,
        key=lambda x: (
            safe_float(x.get("candidate_score"), 0.0),
            x["key"],
        ),
        reverse=True,
    )


def run_mode_b():
    print("===== MODE B (PUBLISHER) START =====")
    print("MODE B VERSION: META_AWARE_SOURCE_GROUP_V5_GAME_AWARE")
    print("B_MAX_PUBLISH_PER_RUN:", B_MAX_PUBLISH_PER_RUN)
    print("B_AVOID_SAME_SOURCE_PER_RUN:", B_AVOID_SAME_SOURCE_PER_RUN)
    print("B_BLOCK_IF_SOURCE_ALREADY_PUBLISHED:", B_BLOCK_IF_SOURCE_ALREADY_PUBLISHED)
    print("B_MIN_CANDIDATE_SCORE:", B_MIN_CANDIDATE_SCORE)
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
    used_source_groups_this_run = set()

    for item in queue_items:
        if count >= B_MAX_PUBLISH_PER_RUN:
            break

        key = item["key"]
        source_group = item.get("source_group") or "unknown"

        if B_AVOID_SAME_SOURCE_PER_RUN and source_group in used_source_groups_this_run:
            print("SKIP same source group in this run:", source_group, "|", key)
            continue

        if should_skip_for_score(item):
            print(
                "SKIP low candidate_score:",
                key,
                "| score:", item.get("candidate_score"),
                "| min:", B_MIN_CANDIDATE_SCORE,
            )
            continue

        process, missing_platforms = should_process_item(state, item)
        if not process:
            print("SKIP already published / blocked by source group:", key)
            continue

        print("Procesando:", key)
        print("SOURCE GROUP:", source_group)
        print("GAME DETECTED:", item.get("game_name"))
        print("CANDIDATE SCORE:", item.get("candidate_score"))
        print("EMOTION:", item.get("emotion"))
        print("INTENSITY:", item.get("intensity"))
        print("MOMENT TYPE:", item.get("moment_type"))
        print("META KEY:", item.get("meta_key"))
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
                    "game_name": item.get("game_name"),
                    "emotion": item.get("emotion"),
                    "intensity": item.get("intensity"),
                    "moment_type": item.get("moment_type"),
                    "brief_txt_key": item.get("brief_txt_key"),
                    "brief": item.get("brief"),
                    "platform_results": result,
                },
            )

            save_state(state)

            if success_any:
                used_source_groups_this_run.add(source_group)
                count += 1

        except Exception as e:
            print("ERROR publicando:", repr(e))

    save_state(state)

    print("Publicados en esta corrida:", count)
    print("===== MODE B DONE =====")


if __name__ == "__main__":
    run_mode_b()

# ===== FIN: ugc_mode_b.py =====
