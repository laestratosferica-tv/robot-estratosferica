# reel_gamer_engine.py
import os
import json
import random
import tempfile
import subprocess
from typing import Dict, Any, Optional, Tuple

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

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v or not v.strip():
        return default
    try:
        return float(v.strip())
    except Exception:
        return default


# -------------------------
# Defaults
# -------------------------
HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 30.0)
REEL_W = env_int("REEL_W", 1080)
REEL_H = env_int("REEL_H", 1920)
REEL_FPS = env_int("REEL_FPS", 30)

FONT_BOLD = env_nonempty("FONT_BOLD", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

# Logo rule (percentages must sum ~100)
LOGO_NONE_PCT = env_int("LOGO_NONE_PCT", 60)
LOGO_SMALL_PCT = env_int("LOGO_SMALL_PCT", 30)
LOGO_BIG_PCT = env_int("LOGO_BIG_PCT", 10)

# Music always?
REEL_MUSIC_ALWAYS = env_bool("REEL_MUSIC_ALWAYS", True)
REEL_MUSIC_VOLUME = env_float("REEL_MUSIC_VOLUME", 0.15)

# Voice sometimes?
REEL_VOICE_PCT = env_int("REEL_VOICE_PCT", 30)  # 0..100
REEL_VOICE_VOLUME = env_float("REEL_VOICE_VOLUME", 1.0)

# If you want: add SFX later
REEL_GLITCH_PCT = env_int("REEL_GLITCH_PCT", 60)  # 0..100
REEL_SCANLINES_PCT = env_int("REEL_SCANLINES_PCT", 70)  # 0..100


# -------------------------
# Style library (variedad gamer)
# -------------------------
STYLE_LIBRARY = [
    {
        "name": "esports_broadcast",
        "weight": 22,
        "bg_mode": "gradient",
        "accent": "magenta",
        "hud": True,
        "glitch": False,
        "scanlines": True,
        "headline_box_opacity": 0.55,
    },
    {
        "name": "cyberpunk_neon",
        "weight": 22,
        "bg_mode": "noise",
        "accent": "neon",
        "hud": True,
        "glitch": True,
        "scanlines": True,
        "headline_box_opacity": 0.50,
    },
    {
        "name": "arcade_pixel",
        "weight": 18,
        "bg_mode": "noise",
        "accent": "pixel",
        "hud": False,
        "glitch": True,
        "scanlines": True,
        "headline_box_opacity": 0.60,
    },
    {
        "name": "fifa_card_drop",
        "weight": 20,
        "bg_mode": "gradient",
        "accent": "gold",
        "hud": True,
        "glitch": False,
        "scanlines": False,
        "headline_box_opacity": 0.55,
    },
    {
        "name": "loot_drop_legendary",
        "weight": 18,
        "bg_mode": "gradient",
        "accent": "legendary",
        "hud": True,
        "glitch": True,
        "scanlines": False,
        "headline_box_opacity": 0.50,
    },
]


def weighted_pick(items):
    total = sum(max(0, int(x.get("weight", 1))) for x in items)
    r = random.randint(1, max(1, total))
    acc = 0
    for it in items:
        w = max(0, int(it.get("weight", 1)))
        acc += w
        if r <= acc:
            return it
    return items[0]


def pick_logo_mode() -> str:
    modes = (["none"] * LOGO_NONE_PCT) + (["small"] * LOGO_SMALL_PCT) + (["big"] * LOGO_BIG_PCT)
    return random.choice(modes) if modes else "none"


# -------------------------
# OpenAI text “director”
# (Usa tu mismo Responses API por HTTP como en Modo B, cero SDK required)
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

    # fallback parse
    texts = []
    for c in j.get("output", []) or []:
        for part in c.get("content", []) or []:
            if part.get("type") == "output_text" and part.get("text"):
                texts.append(part["text"])
    return "\n".join(texts).strip()


def ai_director_plan(headline: str, link: str) -> Dict[str, Any]:
    """
    La IA NO genera el video; genera el "plan" (estilo + copy en pantalla + energía).
    """
    base_style = weighted_pick(STYLE_LIBRARY)
    logo_mode = pick_logo_mode()

    prompt = f"""
Eres director creativo para Reels GAMER (LATAM). Tu trabajo es dar un PLAN para que ffmpeg lo anime.
No inventes marcas ni datos. Mantén el headline fiel.
Devuelve SOLO JSON válido.

HEADLINE: {headline}
LINK: {link}

Elige:
- on_screen_headline: versión corta del headline (máx 60 chars)
- on_screen_kicker: 2-4 palabras tipo "BREAKING", "RUMOR", "UPDATE", "OFICIAL" (en español)
- cta: 1 frase corta (máx 32 chars) tipo "Comenta tu opinión"
- energy: number 1..10 (más alto = más punch)
- text_style: "bold"|"ultra"|"minimal"
- mood_music: "hype"|"trap"|"phonk"|"electro"
"""
    try:
        raw = openai_text(prompt)
        j = json.loads(raw)
    except Exception:
        # fallback safe
        j = {
            "on_screen_headline": (headline or "")[:60],
            "on_screen_kicker": "UPDATE",
            "cta": "Comenta tu opinión",
            "energy": 7,
            "text_style": "bold",
            "mood_music": "hype",
        }

    # merge with base style
    j["style"] = base_style
    j["logo_mode"] = logo_mode
    return j


# -------------------------
# Utilities
# -------------------------
def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(f"Falta {label} en repo: {path}")

def _safe_text(s: str, max_len: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s


# -------------------------
# Core: render gamer reel from ONE image
# -------------------------
def render_gamer_reel_mp4_bytes(
    *,
    headline: str,
    link: str,
    news_image_path: str,
    bg_path: str,
    logo_path: str,
    seconds: int,
    music_path: Optional[str] = None,
    voice_mp3_path: Optional[str] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Returns (mp4_bytes, plan_used)
    """
    _require_file(bg_path, "ASSET_BG")
    _require_file(logo_path, "ASSET_LOGO")
    _require_file(news_image_path, "news_image_path")
    _require_file(FONT_BOLD, "FONT_BOLD")

    plan = ai_director_plan(headline=headline, link=link)
    style = plan.get("style") or {}
    logo_mode = plan.get("logo_mode", "none")

    on_head = _safe_text(plan.get("on_screen_headline", headline), 80)
    kicker = _safe_text(plan.get("on_screen_kicker", "UPDATE"), 16).upper()
    cta = _safe_text(plan.get("cta", "Sigue para más"), 40)

    energy = int(plan.get("energy", 7))
    energy = max(1, min(10, energy))

    # Motion tuning
    # zoom: higher energy => more aggressive zoom
    zoom_start = 1.05 + (energy * 0.01)   # 1.06..1.15
    zoom_end = 1.20 + (energy * 0.015)    # 1.215..1.35

    enable_glitch = bool(style.get("glitch", False)) and (random.randint(1, 100) <= REEL_GLITCH_PCT)
    enable_scanlines = bool(style.get("scanlines", False)) and (random.randint(1, 100) <= REEL_SCANLINES_PCT)
    enable_hud = bool(style.get("hud", False))

    # Logo size/alpha by mode
    if logo_mode == "big":
        logo_scale = 760
        logo_alpha = 1.0
        logo_y = 120
    elif logo_mode == "small":
        logo_scale = 420
        logo_alpha = 0.9
        logo_y = 140
    else:
        logo_scale = 10
        logo_alpha = 0.0
        logo_y = 140

    headline_box_op = float(style.get("headline_box_opacity", 0.55))
    headline_box_op = max(0.25, min(0.75, headline_box_op))

    # Build audio flags
    music_ok = REEL_MUSIC_ALWAYS and bool(music_path) and os.path.exists(music_path)
    voice_ok = bool(voice_mp3_path) and os.path.exists(voice_mp3_path)

    with tempfile.TemporaryDirectory() as td:
        out_mp4 = os.path.join(td, "out.mp4")
        title_txt = os.path.join(td, "title.txt")
        cta_txt = os.path.join(td, "cta.txt")
        kicker_txt = os.path.join(td, "kicker.txt")

        with open(title_txt, "w", encoding="utf-8") as f:
            f.write(on_head)
        with open(cta_txt, "w", encoding="utf-8") as f:
            f.write(cta)
        with open(kicker_txt, "w", encoding="utf-8") as f:
            f.write(kicker)

        # Inputs:
        # 0: base color
        # 1: bg image
        # 2: news image
        # 3: logo
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=black:s={REEL_W}x{REEL_H}:r={REEL_FPS}:d={int(seconds)}",
            "-i", bg_path,
            "-i", news_image_path,
            "-i", logo_path,
        ]
        # optional audio inputs after video inputs
        # music: idx 4
        if music_ok:
            cmd += ["-i", music_path]
        # voice: next
        if voice_ok:
            cmd += ["-i", voice_mp3_path]

        # Animated background: base bg + subtle movement + noise/scanlines
        # News image: zoompan (Ken Burns) + slight rotate/shake with random noise (cheap trick)
        # HUD: overlay some moving boxes/lines made with drawbox + noise

        # Background pipeline
        # bg image scale to full
        fc = []
        fc.append(f"[1:v]scale={REEL_W}:{REEL_H},format=rgba[bg]")

        # add subtle animated noise layer (optional)
        # using geq noise is expensive; we do a lightweight noise source
        fc.append(f"noise=alls=20:allf=t+u,format=rgba[no]")

        # compose bg + noise at low alpha
        fc.append(f"[bg][no]overlay=0:0:format=auto:alpha=0.08[bg2]")

        # scanlines overlay
        if enable_scanlines:
            # create moving scanlines
            fc.append(f"color=c=black@0.0:s={REEL_W}x{REEL_H}:r={REEL_FPS}:d={int(seconds)}[sl0]")
            # draw many thin boxes via drawbox is heavy; simulate via blend with a generated pattern
            # simplest: use 'tblend' with noise - looks like CRT
            fc.append(f"[bg2]tblend=all_mode=average,eq=contrast=1.05:saturation=1.15[bg3]")
        else:
            fc.append(f"[bg2]eq=contrast=1.06:saturation=1.18[bg3]")

        # base over color
        fc.append(f"[0:v][bg3]overlay=0:0:format=auto[v0]")

        # News image: scale then zoompan
        # Keep it centered around mid-upper
        news_w = REEL_W - 140
        fc.append(f"[2:v]scale={news_w}:-1,format=rgba[news]")
        # zoompan: frames = seconds*fps
        frames = int(seconds * REEL_FPS)
        # zoom expression: from zoom_start to zoom_end across frames
        zexpr = f"if(lte(on,0),{zoom_start},min({zoom_end},zoom+0.0015))"
        fc.append(
            f"[news]zoompan=z='{zexpr}':d={frames}:s={news_w}x{int(REEL_H*0.42)}:fps={REEL_FPS}[news_z]"
        )

        # place news image
        news_y = 520
        fc.append(f"[v0][news_z]overlay=(W-w)/2:{news_y}:format=auto[v1]")

        # HUD / frames / accents
        if enable_hud:
            # a few animated boxes
            fc.append(
                f"[v1]drawbox=x=40:y=470:w={REEL_W-80}:h={int(REEL_H*0.46)}:color=white@0.08:t=4,"
                f"drawbox=x=60:y=490:w={REEL_W-120}:h={int(REEL_H*0.46)-40}:color=white@0.05:t=2"
                f"[v2]"
            )
        else:
            fc.append(f"[v1]copy[v2]")

        # Logo overlay (if any)
        if logo_alpha > 0.01:
            fc.append(f"[3:v]scale={logo_scale}:-1,format=rgba,colorchannelmixer=aa={logo_alpha}[lg]")
            fc.append(f"[v2][lg]overlay=(W-w)/2:{logo_y}:format=auto[v3]")
        else:
            fc.append(f"[v2]copy[v3]")

        # Kicker (small) + headline + CTA
        # Use boxed text; looks broadcast.
        # headline position:
        head_y = 1320
        cta_y = 1540

        # glitch effect (cheap): occasionally add slight color shift via eq/hue
        if enable_glitch:
            fc.append(f"[v3]hue=h=2:s=1.1,eq=contrast=1.08:saturation=1.25[v3g]")
            base_for_text = "[v3g]"
        else:
            base_for_text = "[v3]"

        draw = (
            f"{base_for_text}"
            f"drawtext=fontfile={FONT_BOLD}:textfile={kicker_txt}:x=60:y=460:fontsize=44:fontcolor=white:box=1:boxcolor=black@0.35:boxborderw=18,"
            f"drawtext=fontfile={FONT_BOLD}:textfile={title_txt}:x=60:y={head_y}:fontsize=52:fontcolor=white:box=1:boxcolor=black@{headline_box_op}:boxborderw=26,"
            f"drawtext=fontfile={FONT_BOLD}:textfile={cta_txt}:x=60:y={cta_y}:fontsize=44:fontcolor=white:box=1:boxcolor=black@0.35:boxborderw=18"
            f"[vout]"
        )
        fc.append(draw)

        filter_complex = ";".join(fc)

        cmd += ["-filter_complex", filter_complex, "-map", "[vout]"]

        # AUDIO MIX:
        # If music only -> keep music
        # If music + voice -> mix
        # If none -> silent
        audio_inputs = []
        music_index = None
        voice_index = None
        next_audio_idx = 4

        if music_ok:
            music_index = next_audio_idx
            next_audio_idx += 1
        if voice_ok:
            voice_index = next_audio_idx
            next_audio_idx += 1

        if music_index is not None or voice_index is not None:
            # build amix
            aparts = []
            mix_ins = []
            if music_index is not None:
                aparts.append(f"[{music_index}:a]volume={REEL_MUSIC_VOLUME}[am]")
                mix_ins.append("[am]")
            if voice_index is not None:
                aparts.append(f"[{voice_index}:a]volume={REEL_VOICE_VOLUME}[av]")
                mix_ins.append("[av]")

            if len(mix_ins) == 1:
                # single audio
                fc_audio = ";".join(aparts) + f";{mix_ins[0]}anull[aout]"
            else:
                fc_audio = ";".join(aparts) + f";{''.join(mix_ins)}amix=inputs={len(mix_ins)}:duration=first:dropout_transition=2[aout]"

            cmd += ["-filter_complex", filter_complex + ";" + fc_audio, "-map", "[vout]", "-map", "[aout]"]
            cmd += ["-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]

        cmd += [
            "-t", str(int(seconds)),
            "-r", str(REEL_FPS),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-shortest",
            out_mp4,
        ]

        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg gamer reel failed:\n{(p.stderr or '')[:4000]}")

        with open(out_mp4, "rb") as f:
            mp4_bytes = f.read()

    return mp4_bytes, plan


def should_add_voice() -> bool:
    return random.randint(1, 100) <= max(0, min(100, REEL_VOICE_PCT))
