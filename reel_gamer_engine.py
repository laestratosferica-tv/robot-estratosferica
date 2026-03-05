# reel_gamer_engine.py
import os
import json
import random
import tempfile
import subprocess
from typing import Dict, Any, Optional, Tuple, List

import requests

# -------------------------
# ENV helpers
# -------------------------
def env_nonempty(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v or not v.strip():
        return default
    try:
        return int(v.strip())
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v or not v.strip():
        return default
    try:
        return float(v.strip())
    except Exception:
        return default

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

# -------------------------
# Defaults / tuning
# -------------------------
HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)

REEL_W = env_int("REEL_W", 1080)
REEL_H = env_int("REEL_H", 1920)
REEL_FPS = env_int("REEL_FPS", 30)
REEL_SECONDS = env_int("REEL_SECONDS", 8)

FONT_BOLD = env_nonempty("FONT_BOLD", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

# Brand assets (optional)
ASSET_LOGO = env_nonempty("ASSET_LOGO", "assets/logo.png")
LOGO_PROB = env_float("LOGO_PROBABILITY", 0.25)  # más bajo = menos institucional
LOGO_MODE = env_nonempty("LOGO_MODE", "auto")    # auto|none|small|big

# FX probabilities
GLITCH_PROB = env_float("GLITCH_PROBABILITY", 0.35)
HUD_PROB = env_float("HUD_PROBABILITY", 0.60)
SCANLINES_PROB = env_float("SCANLINES_PROBABILITY", 0.45)

# Text behavior
MAX_HOOK_CHARS = env_int("MAX_HOOK_CHARS", 18)         # HOOK corto
MAX_BODY_CHARS = env_int("MAX_BODY_CHARS", 48)         # 1 frase
MAX_CTA_CHARS  = env_int("MAX_CTA_CHARS", 26)          # pregunta corta
TEXT_MAX_LINES = env_int("TEXT_MAX_LINES", 2)          # 1-2 líneas

# -------------------------
# Style library (random look)
# -------------------------
STYLE_LIBRARY: List[Dict[str, Any]] = [
    {"name": "cyber_neon", "box_op": 0.40, "contrast": 1.10, "sat": 1.30, "shake": 0.0022},
    {"name": "esports_broadcast", "box_op": 0.48, "contrast": 1.08, "sat": 1.20, "shake": 0.0012},
    {"name": "arcade_glitch", "box_op": 0.38, "contrast": 1.14, "sat": 1.38, "shake": 0.0028},
    {"name": "minimal_punch", "box_op": 0.32, "contrast": 1.06, "sat": 1.15, "shake": 0.0008},
]

def weighted_pick(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # simple uniform; you can add weights later
    return random.choice(items)

# -------------------------
# OpenAI text director (viral + corto)
# -------------------------
def openai_text(prompt: str) -> str:
    api_key = env_nonempty("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY")
    model = env_nonempty("OPENAI_MODEL", "gpt-4.1-mini")
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": prompt}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        # fallback chat.completions
        url2 = "https://api.openai.com/v1/chat/completions"
        payload2 = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        r2 = requests.post(url2, headers=headers, json=payload2, timeout=60)
        r2.raise_for_status()
        j2 = r2.json()
        return (j2["choices"][0]["message"]["content"] or "").strip()

    j = r.json()
    out = j.get("output_text")
    if out:
        return str(out).strip()

    texts = []
    for c in j.get("output", []) or []:
        for part in c.get("content", []) or []:
            if part.get("type") == "output_text" and part.get("text"):
                texts.append(part["text"])
    return "\n".join(texts).strip()

def _clip(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s[:n].rstrip()

def ai_viral_plan(title: str, link: str = "") -> Dict[str, str]:
    """
    Devuelve HOOK / BODY / CTA en español, MUY corto, polémico.
    """
    prompt = f"""
Eres editor viral GAMING/ESPORTS LATAM.
Objetivo: comentarios. Sin insultos.

Genera SOLO estas 3 líneas (sin nada más):
HOOK: (máx {MAX_HOOK_CHARS} caracteres)
BODY: (máx {MAX_BODY_CHARS} caracteres)
CTA:  (máx {MAX_CTA_CHARS} caracteres, pregunta)

Noticia/tema:
{title}

Link (referencia, no lo copies): {link}
"""
    try:
        raw = openai_text(prompt)
    except Exception:
        raw = ""

    hook = body = cta = ""
    for line in (raw or "").splitlines():
        line = line.strip()
        if line.upper().startswith("HOOK:"):
            hook = line.split(":", 1)[1].strip()
        elif line.upper().startswith("BODY:"):
            body = line.split(":", 1)[1].strip()
        elif line.upper().startswith("CTA:"):
            cta = line.split(":", 1)[1].strip()

    # fallbacks
    hook = _clip(hook or "ESPORTS DRAMA", MAX_HOOK_CHARS)
    body = _clip(body or _clip(title, MAX_BODY_CHARS), MAX_BODY_CHARS)
    cta  = _clip(cta  or "¿TÚ QUÉ OPINAS?", MAX_CTA_CHARS)

    return {"hook": hook.upper(), "body": body.upper(), "cta": cta.upper()}

# -------------------------
# Text layout helpers
# -------------------------
def wrap_lines(text: str, max_chars_line: int = 16, max_lines: int = 2) -> str:
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
        if len(cur) + 1 + len(w) <= max_chars_line:
            cur = cur + " " + w
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break
    if len(lines) < max_lines and cur:
        lines.append(cur)
    lines = lines[:max_lines]
    return "\n".join(lines).strip()

# -------------------------
# Core renderer: FULLSCREEN image + punch overlays
# -------------------------
def render_gamer_reel_from_image_bytes(
    *,
    headline: str,
    link: str,
    image_bytes: bytes,
    image_ext: str = ".jpg",
    seconds: Optional[int] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    FULLSCREEN: la foto es el recurso principal.
    Anim: zoom + (glitch/hud/scanlines opcional) + textos cortos.
    """
    if not os.path.exists(FONT_BOLD):
        raise RuntimeError(f"FONT_BOLD no existe: {FONT_BOLD}")

    seconds = int(seconds or REEL_SECONDS)
    style = weighted_pick(STYLE_LIBRARY)
    plan = ai_viral_plan(headline, link)

    # Random toggles
    enable_glitch = random.random() < GLITCH_PROB
    enable_hud = random.random() < HUD_PROB
    enable_scan = random.random() < SCANLINES_PROB

    # Logo decision (menos institucional)
    logo_mode = (LOGO_MODE or "auto").lower()
    use_logo = False
    if logo_mode == "none":
        use_logo = False
    elif logo_mode in ("small", "big"):
        use_logo = os.path.exists(ASSET_LOGO)
    else:
        use_logo = (random.random() < max(0.0, min(1.0, LOGO_PROB))) and os.path.exists(ASSET_LOGO)
        logo_mode = "small" if random.random() < 0.75 else "big"

    # Text positions random (para que no se vea plantilla)
    # top or mid-left
    variant = random.choice(["top_left", "mid_left", "bottom_left"])
    if variant == "top_left":
        hook_y, body_y, cta_y = 220, 340, 1580
    elif variant == "mid_left":
        hook_y, body_y, cta_y = 740, 880, 1580
    else:
        hook_y, body_y, cta_y = 1220, 1380, 1580

    hook = wrap_lines(plan["hook"], max_chars_line=14, max_lines=1)
    body = wrap_lines(plan["body"], max_chars_line=18, max_lines=TEXT_MAX_LINES)
    cta  = wrap_lines(plan["cta"],  max_chars_line=18, max_lines=1)

    # Make video
    with tempfile.TemporaryDirectory() as td:
        img_path = os.path.join(td, f"img{image_ext if image_ext.startswith('.') else '.jpg'}")
        out_mp4 = os.path.join(td, "out.mp4")
        hook_txt = os.path.join(td, "hook.txt")
        body_txt = os.path.join(td, "body.txt")
        cta_txt  = os.path.join(td, "cta.txt")

        with open(img_path, "wb") as f:
            f.write(image_bytes)
        with open(hook_txt, "w", encoding="utf-8") as f:
            f.write(hook)
        with open(body_txt, "w", encoding="utf-8") as f:
            f.write(body)
        with open(cta_txt, "w", encoding="utf-8") as f:
            f.write(cta)

        # Zoompan tuning
        # más punch con shake (muy leve)
        shake = float(style.get("shake", 0.0015))
        contrast = float(style.get("contrast", 1.10))
        sat = float(style.get("sat", 1.25))
        box_op = float(style.get("box_op", 0.42))

        # Build filter_complex
        # 0: image (loop)
        # 1: logo (optional)
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner",
            "-loglevel", "error",
            "-loop", "1", "-i", img_path,
        ]
        if use_logo:
            cmd += ["-i", ASSET_LOGO]

        # Base fullscreen fill + cinematic zoom
        frames = seconds * REEL_FPS
        # zoom increases slowly; crop stays centered but with tiny shake
        zexpr = "min(1.22,1+0.0009*on)"
        xexpr = f"(iw/2)-(iw/zoom/2)+{shake}*iw*sin(2*PI*on/17)"
        yexpr = f"(ih/2)-(ih/zoom/2)+{shake}*ih*cos(2*PI*on/19)"

        vf = []
        vf.append(
            f"[0:v]scale={REEL_W}:{REEL_H}:force_original_aspect_ratio=increase,"
            f"crop={REEL_W}:{REEL_H},"
            f"zoompan=z='{zexpr}':x='{xexpr}':y='{yexpr}':d={frames}:s={REEL_W}x{REEL_H}:fps={REEL_FPS},"
            f"eq=contrast={contrast}:saturation={sat},"
            f"format=rgba[v0]"
        )

        # Scanlines (light)
        if enable_scan:
            vf.append(
                f"color=c=white@0.035:s={REEL_W}x{REEL_H}:r={REEL_FPS}:d={seconds}[sl];"
                f"[v0][sl]blend=all_mode=multiply:all_opacity=0.55[v1]"
            )
            base = "[v1]"
        else:
            base = "[v0]"

        # HUD overlay (cheap but effective)
        if enable_hud:
            vf.append(
                f"{base}"
                f"drawbox=x=40:y=430:w={REEL_W-80}:h=12:color=white@0.18:t=fill,"
                f"drawbox=x=40:y=430:w=12:h={REEL_H-860}:color=white@0.10:t=fill,"
                f"drawbox=x={REEL_W-52}:y=430:w=12:h={REEL_H-860}:color=white@0.10:t=fill,"
                f"drawbox=x=40:y={REEL_H-430}:w={REEL_W-80}:h=12:color=white@0.12:t=fill"
                f"[v2]"
            )
            base2 = "[v2]"
        else:
            base2 = base

        # Glitch tint occasionally (subtle)
        if enable_glitch:
            vf.append(f"{base2}hue=h=2:s=1.05,eq=contrast={contrast+0.03}:saturation={sat+0.08}[v3]")
            base3 = "[v3]"
        else:
            base3 = base2

        # Logo overlay (top-right small / center big)
        if use_logo:
            if logo_mode == "big":
                vf.append(f"[1:v]scale=560:-1,format=rgba[lg];{base3}[lg]overlay=(W-w)/2:120:format=auto[v4]")
                base4 = "[v4]"
            else:
                vf.append(f"[1:v]scale=240:-1,format=rgba[lg];{base3}[lg]overlay=W-w-60:80:format=auto[v4]")
                base4 = "[v4]"
        else:
            base4 = base3

        # Text overlays: hook, body, cta (entrada con fade)
        # Hook BIG, body medium, CTA medium
        text = (
            f"{base4}"
            f"drawtext=fontfile={FONT_BOLD}:textfile={hook_txt}:"
            f"x=70:y={hook_y}:fontsize=86:fontcolor=white:"
            f"borderw=2:bordercolor=black@0.6:"
            f"box=1:boxcolor=black@{box_op}:boxborderw=22:"
            f"alpha='if(lt(t,0.15),0, if(lt(t,0.40),(t-0.15)/0.25,1))',"
            f"drawtext=fontfile={FONT_BOLD}:textfile={body_txt}:"
            f"x=70:y={body_y}:fontsize=58:fontcolor=white:"
            f"borderw=2:bordercolor=black@0.5:"
            f"box=1:boxcolor=black@{box_op-0.06}:boxborderw=20:"
            f"alpha='if(lt(t,0.35),0, if(lt(t,0.65),(t-0.35)/0.30,1))',"
            f"drawtext=fontfile={FONT_BOLD}:textfile={cta_txt}:"
            f"x=70:y={cta_y}:fontsize=56:fontcolor=white:"
            f"borderw=2:bordercolor=black@0.5:"
            f"box=1:boxcolor=black@0.30:boxborderw=18:"
            f"alpha='if(lt(t,0.85),0, if(lt(t,1.10),(t-0.85)/0.25,1))'"
            f"[vout]"
        )

        # Important: filter graph must be single string.
        # We constructed vf parts; join with ';' except first already sets [v0]
        # We'll build as:
        # 1) first line produces [v0]
        # 2) following lines reference [v0] or produce [v1],[v2]...
        parts = []
        parts.append(vf[0])
        for p in vf[1:]:
            parts.append(p)
        parts.append(text)

        filter_complex = ";".join(parts)

        cmd += [
            "-t", str(seconds),
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            out_mp4,
        ]

        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
        if p.returncode != 0:
            raise RuntimeError(f"FFmpeg falló:\n{(p.stderr or '')[:4000]}")

        with open(out_mp4, "rb") as f:
            mp4_bytes = f.read()

    meta = {
        "style": style,
        "plan": plan,
        "enable_glitch": enable_glitch,
        "enable_hud": enable_hud,
        "enable_scanlines": enable_scan,
        "use_logo": use_logo,
        "logo_mode": logo_mode,
        "variant": variant,
        "seconds": seconds,
        "fps": REEL_FPS,
    }
    return mp4_bytes, meta
