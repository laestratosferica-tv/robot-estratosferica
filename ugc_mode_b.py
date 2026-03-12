import os
import re
import json
import time
import random
import tempfile

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

ENABLE_INSTAGRAM = env_bool("ENABLE_INSTAGRAM", True)
ENABLE_FACEBOOK = env_bool("ENABLE_FACEBOOK", True)
ENABLE_TIKTOK = env_bool("ENABLE_TIKTOK", False)
ENABLE_SHORTS = env_bool("ENABLE_SHORTS", True)

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

    items.sort(
        key=lambda x: x["last_modified"] or 0,
        reverse=True,
    )

    return [x["key"] for x in items]


def load_state():
    try:
        obj = r2().get_object(Bucket=BUCKET_NAME, Key=STATE_KEY)
        st = json.loads(obj["Body"].read())
    except Exception:
        st = {}

    if not isinstance(st, dict):
        st = {}

    old_published = st.get("published", {})

    if isinstance(old_published, list):
        st["published"] = {
            "instagram": list(old_published),
            "facebook": list(old_published),
            "youtube_shorts": [],
        }
    elif isinstance(old_published, dict):
        st["published"] = old_published
    else:
        st["published"] = {}

    st["published"].setdefault("instagram", [])
    st["published"].setdefault("facebook", [])
    st["published"].setdefault("youtube_shorts", [])

    if not isinstance(st["published"]["instagram"], list):
        st["published"]["instagram"] = []
    if not isinstance(st["published"]["facebook"], list):
        st["published"]["facebook"] = []
    if not isinstance(st["published"]["youtube_shorts"], list):
        st["published"]["youtube_shorts"] = []

    return st


def save_state(st):
    if not isinstance(st, dict):
        st = {}

    st.setdefault("published", {})
    st["published"].setdefault("instagram", [])
    st["published"].setdefault("facebook", [])
    st["published"].setdefault("youtube_shorts", [])

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


def extract_source_group(key):
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


def diversify_queue(queue):
    if not B_AVOID_SAME_SOURCE_PER_RUN:
        return queue

    groups = {}
    order = []

    for key in queue:
        group = extract_source_group(key)

        if group not in groups:
            groups[group] = []
            order.append(group)

        groups[group].append(key)

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


def detect_game_from_key(key):
    text = (key or "").lower().replace("_", " ").replace("-", " ")

    checks = [
        ("valorant", "Valorant"),
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
        ("fc", "EA Sports FC"),
        ("f1", "F1"),
        ("gran turismo", "Gran Turismo"),
    ]

    for needle, label in checks:
        if needle in text:
            return label

    return "Esports"


GAME_HOOKS = {
    "Valorant": [
        "Esto no fue una play normal.",
        "Hay rounds que definen mapas y rounds que cambian todo.",
        "Si esto no te levanta de la silla, nada lo hace.",
    ],
    "CS2": [
        "Esto fue puro timing y sangre fría.",
        "Hay clips buenos y luego está esto.",
        "Si pestañeaste, te perdiste la jugada.",
    ],
    "League of Legends": [
        "Hay teamfights que explican por qué este juego sigue siendo cine.",
        "Esto fue una locura total de principio a fin.",
        "La diferencia entre competir y dominar se ve aquí.",
    ],
    "Fortnite": [
        "Fortnite en modo absolutamente ridículo.",
        "Esto es de esas jugadas que revientan la partida.",
        "Hay mecánicas y luego está este nivel de locura.",
    ],
    "Warzone": [
        "Esto no se gana normalmente.",
        "Hay cierres sucios… y luego está este.",
        "Warzone a veces parece un caos. Esto ya fue arte.",
    ],
    "Apex Legends": [
        "Movement, lectura y sangre fría. Todo en una sola play.",
        "Esto es Apex en su versión más salvaje.",
        "Hay finales tensos y luego está esto.",
    ],
    "Minecraft": [
        "Minecraft también puede ser puro cine.",
        "Esto no tenía ningún sentido y aun así pasó.",
        "Hay momentos que parecen editados… pero no.",
    ],
    "EA Sports FC": [
        "Esto es de esos goles que encienden a toda la comunidad.",
        "Hay jugadas bonitas y luego están las que humillan.",
        "No me puedes decir que esto es normal.",
    ],
    "F1": [
        "Esto fue una barbaridad de precisión.",
        "Hay adelantamientos buenos y luego está esta locura.",
        "F1 cuando se pone serio es una obra de arte.",
    ],
    "Gran Turismo": [
        "Esto es manejo fino del bueno.",
        "Hay vueltas limpias y luego está esta joya.",
        "Gran Turismo también sabe regalar cine puro.",
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
        "La comunidad siempre dice que estas rondas separan al bueno del que de verdad compite.",
    ],
    "CS2": [
        "En CS2, una decisión así vale más que mil highlights vacíos.",
        "Esto es lo que pasa cuando aim, timing y cabeza se alinean.",
        "Si entiendes CS2, sabes por qué esta jugada pesa tanto.",
    ],
    "League of Legends": [
        "En LoL, una sola secuencia puede reventar todo el mapa.",
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
        "Este tipo de clip es el que pone a debatir a toda la escena.",
        "Aquí no fue solo suerte: hubo timing, decisión y sangre fría.",
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
        "En FC, una jugada así te pone a discutir con cualquiera en comentarios.",
        "Esto es justo el tipo de acción que parte a la comunidad en dos.",
        "Hay goles que valen uno. Este vale conversación toda la semana.",
    ],
    "F1": [
        "En F1, una maniobra así no se regala: se gana.",
        "Esto es precisión pura con presión máxima.",
        "Este tipo de adelantamiento explica por qué la escena engancha tanto.",
    ],
    "Gran Turismo": [
        "Esto es de lo más fino que puedes ver en un clip corto.",
        "Hay técnica, hay lectura, y luego hay esta belleza.",
        "Gran Turismo sabe regalar momentos de locura elegante.",
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
        "¿Tú dirías que aquí se ganó la serie mentalmente?",
    ],
    "Fortnite": [
        "¿De verdad sigue habiendo alguien mejor que Bugha en esto?",
        "¿Esto fue locura mecánica o pura descoordinación rival?",
        "¿La mejor play o una de esas que se inflan por el replay?",
    ],
    "Warzone": [
        "¿Esto fue puro control o simple caos favorable?",
        "¿Skill brutal o suerte con esteroides?",
        "¿Tú lo llamas clutch o milagro?",
    ],
    "Apex Legends": [
        "¿Esto fue movement de élite o error del lobby?",
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
        "¿Esto fue limpieza total o el rival se equivocó feo?",
        "¿Manejo perfecto o clip sobrevalorado?",
        "¿Tú lo pones como top play o no tanto?",
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
    "Minecraft": ["#Minecraft", "#MinecraftLATAM", "#GamingLATAM", "#EsportsLATAM", "#MinecraftClips"],
    "EA Sports FC": ["#EASportsFC", "#FC", "#FCLATAM", "#GamingLATAM", "#EsportsLATAM"],
    "F1": ["#F1", "#SimRacing", "#EsportsLATAM", "#GamingLATAM", "#F1Esports"],
    "Gran Turismo": ["#GranTurismo", "#SimRacing", "#EsportsLATAM", "#GamingLATAM", "#GTLATAM"],
    "Esports": ["#EsportsLATAM", "#GamingLATAM", "#Esports", "#Gaming", "#ReelsGaming"],
}


def pick_from_map(mapping, game_name):
    options = mapping.get(game_name) or mapping.get("Esports") or []
    return random.choice(options) if options else ""


def build_caption(key):
    game_name = detect_game_from_key(key)
    hook = pick_from_map(GAME_HOOKS, game_name)
    context = pick_from_map(GAME_CONTEXTS, game_name)
    cta = pick_from_map(GAME_CTAS, game_name)
    hashtags = " ".join(GAME_HASHTAGS.get(game_name, GAME_HASHTAGS["Esports"]))

    caption = f"""{hook}

{context}

🔥 {cta}

{hashtags}"""

    return caption.strip()


def build_shorts_title(key):
    game_name = detect_game_from_key(key)
    hook = pick_from_map(GAME_HOOKS, game_name) or "Clip brutal de esports"
    title = hook.replace("\n", " ").strip()

    if "#Shorts" not in title:
        title = f"{title} #Shorts"

    return title[:100]


def build_shorts_description(key):
    game_name = detect_game_from_key(key)
    context = pick_from_map(GAME_CONTEXTS, game_name)
    cta = pick_from_map(GAME_CTAS, game_name)
    hashtags = " ".join(GAME_HASHTAGS.get(game_name, GAME_HASHTAGS["Esports"]))

    text = f"""{context}

{cta}

{hashtags}"""
    return text[:5000].strip()


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

        time.sleep(3)

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


def publish(key, target_platforms=None):
    if target_platforms is None:
        target_platforms = ["instagram", "facebook", "youtube_shorts"]

    public_url = f"{R2_PUBLIC_BASE_URL}/{key}"
    caption = build_caption(key)
    shorts_title = build_shorts_title(key)
    shorts_description = build_shorts_description(key)

    print("PUBLICANDO VIDEO:")
    print("KEY:", key)
    print("URL:", public_url)
    print("TARGET_PLATFORMS:", target_platforms)
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


def should_process_key(st, key):
    wanted = []

    if ENABLE_INSTAGRAM and not is_published_on(st, "instagram", key):
        wanted.append("instagram")

    if ENABLE_FACEBOOK and not is_published_on(st, "facebook", key):
        wanted.append("facebook")

    if ENABLE_SHORTS and not is_published_on(st, "youtube_shorts", key):
        wanted.append("youtube_shorts")

    return len(wanted) > 0, wanted


def run_mode_b():
    print("===== MODE B (PUBLISHER) START =====")
    print("MODE B VERSION: REAL_PUBLISH_DIVERSIFIED_V7_SHORTS_PUBLIC")
    print("B_MAX_PUBLISH_PER_RUN:", B_MAX_PUBLISH_PER_RUN)
    print("B_AVOID_SAME_SOURCE_PER_RUN:", B_AVOID_SAME_SOURCE_PER_RUN)
    print("B_ONLY_KEYS_CONTAIN:", B_ONLY_KEYS_CONTAIN or "(vacío)")
    print("PREFIX_PRIORITY:", PREFIX_PRIORITY)
    print("PREFIX_MANUAL:", PREFIX_MANUAL)
    print("PREFIX_AUTO:", PREFIX_AUTO)
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

    queue = []
    queue.extend(priority)
    queue.extend(manual)
    queue.extend(auto)

    if B_ONLY_KEYS_CONTAIN:
        queue = [k for k in queue if B_ONLY_KEYS_CONTAIN in k]
        print("Queue tras filtro:", len(queue))

    queue = diversify_queue(queue)

    print("Queue final diversificada:", len(queue))

    count = 0

    for key in queue:
        if count >= B_MAX_PUBLISH_PER_RUN:
            break

        process, missing_platforms = should_process_key(state, key)
        if not process:
            print("SKIP already published in enabled platforms:", key)
            continue

        print("Procesando:", key)
        print("SOURCE GROUP:", extract_source_group(key))
        print("GAME DETECTED:", detect_game_from_key(key))
        print("MISSING PLATFORMS:", missing_platforms)

        try:
            result = publish(key, target_platforms=missing_platforms)

            success_any = False

            if "instagram" in missing_platforms and result.get("instagram", {}).get("ok"):
                mark_published_on(state, "instagram", key)
                success_any = True

            if "facebook" in missing_platforms and result.get("facebook", {}).get("ok"):
                mark_published_on(state, "facebook", key)
                success_any = True

            if "youtube_shorts" in missing_platforms and result.get("youtube_shorts", {}).get("ok"):
                mark_published_on(state, "youtube_shorts", key)
                success_any = True

            save_state(state)

            if success_any:
                count += 1

        except Exception as e:
            print("ERROR publicando:", repr(e))

    save_state(state)

    print("Publicados en esta corrida:", count)
    print("===== MODE B DONE =====")


if __name__ == "__main__":
    run_mode_b()
