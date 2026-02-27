import os
import json
import tempfile
import subprocess
from typing import Optional, Tuple
import requests

# -------------------------
# ENV
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1").strip()

REEL_W = int(os.getenv("REEL_W", "1080"))
REEL_H = int(os.getenv("REEL_H", "1920"))
REEL_SECONDS = int(os.getenv("REEL_SECONDS", "8"))

FONT_BOLD = os.getenv("FONT_BOLD", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
ASSET_LOGO = os.getenv("ASSET_LOGO", "assets/logo.png")

# logo rules
LOGO_NONE_PCT = int(os.getenv("LOGO_NONE_PCT", "60"))
LOGO_SMALL_PCT = int(os.getenv("LOGO_SMALL_PCT", "30"))
LOGO_BIG_PCT = int(os.getenv("LOGO_BIG_PCT", "10"))

# -------------------------
# OpenAI Image (HTTP)
# -------------------------
def openai_generate_image_png(prompt: str) -> bytes:
    """
    Generates a single PNG using OpenAI Images API (best-effort).
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY")
    url = "https://api.openai.com/v1/images"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    # NOTE: This API shape may vary by account/model; this is a practical best-effort.
    data = {
        "model": OPENAI_IMAGE_MODEL,
        "prompt": prompt,
        "size": "1024x1024",
        "n": 1,
        "response_format": "b64_json",
    }

    r = requests.post(url, headers=headers, json=data, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI image API error: {r.status_code} {r.text[:600]}")

    j = r.json()
    # Most common format: data[0].b64_json
    import base64
    b64 = j["data"][0].get("b64_json")
    if not b64:
        raise RuntimeError(f"OpenAI image API unexpected response: {str(j)[:600]}")
    return base64.b64decode(b64)


# -------------------------
# Visual prompt builder
# -------------------------
def build_bg_prompt(headline: str, category: str) -> str:
    """
    AI-first: generates a background that FEELS like gaming/esports.
    No copyrighted logos, no real team names.
    """
    headline = (headline or "").strip().replace("\n", " ")[:140]
    category = (category or "esports").strip().lower()

    # category -> style nudges
    style = {
        "breaking": "dramatic neon, high contrast, urgent breaking news vibe, particles, glow",
        "result": "scoreboard vibe, clean esports UI, bold shapes, modern arena lighting",
        "transfer": "futuristic card vibe, collectible UI, holographic accents, premium neon",
        "drama": "dark cyber mood, glitch, warning stripes, tense atmosphere",
        "announcement": "premium minimal neon, clean gradients, subtle glow, modern tech",
    }.get(category, "neon esports, modern gaming UI, cyberpunk accents, clean but energetic")

    return f"""
Create a vertical background image for an esports/gaming news reel.
Style: {style}
Constraints:
- No real logos, no copyrighted characters, no real team names.
- Must be clean enough for text overlay, with a readable central area.
- Feel premium, modern, LATAM esports media.
Text context (do not render exact text): "{headline}"
Output: a single background image.
""".strip()


# -------------------------
# Video maker (ffmpeg)
# -------------------------
def make_news_reel_mp4(
    headline: str,
    cta: str = "Síguenos para más",
    category: str = "esports",
    logo_mode: str = "auto",  # auto | none | small | big
) -> bytes:
    """
    Creates a short 1080x1920 mp4 using:
    - AI-generated background (png)
    - Animated text (simple fade/slide)
    - Optional logo
    """
    if not os.path.exists(FONT_BOLD):
        raise RuntimeError(f"FONT_BOLD no existe: {FONT_BOLD}")

    headline = (headline or "").strip().replace("\n", " ")[:80]
    cta = (cta or "").strip()[:40]

    # Decide logo mode
    import random
    if logo_mode == "auto":
        roll = random.randint(1, 100)
        if roll <= LOGO_NONE_PCT:
            logo_mode = "none"
        elif roll <= LOGO_NONE_PCT + LOGO_SMALL_PCT:
            logo_mode = "small"
        else:
            logo_mode = "big"

    # Generate background via OpenAI
    bg_prompt = build_bg_prompt(headline, category)
    bg_png = openai_generate_image_png(bg_prompt)

    with tempfile.TemporaryDirectory() as td:
        bg_path = os.path.join(td, "bg.png")
        out_mp4 = os.path.join(td, "out.mp4")
        title_txt = os.path.join(td, "title.txt")
        cta_txt = os.path.join(td, "cta.txt")

        with open(bg_path, "wb") as f:
            f.write(bg_png)

        with open(title_txt, "w", encoding="utf-8") as f:
            f.write(headline)

        with open(cta_txt, "w", encoding="utf-8") as f:
            f.write(cta)

        # Logo input only if needed and exists
        use_logo = (logo_mode != "none") and os.path.exists(ASSET_LOGO)

        # Base: background scaled to 1080x1920 + subtle zoom animation
        # Text: hook + CTA with fade in
        # (Simple but effective; we can get fancier later.)
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-loop", "1", "-i", bg_path,
        ]
        if use_logo:
            cmd += ["-i", ASSET_LOGO]

        # zoompan for subtle motion
        # drawtext for headline and cta
        logo_filter = ""
        if use_logo:
            if logo_mode == "small":
                logo_w = 220
                logo_y = 80
            else:  # big
                logo_w = 520
                logo_y = 140
            logo_filter = (
                f";[1:v]scale={logo_w}:-1,format=rgba[logo]"
                f";[v0][logo]overlay=(W-w)/2:{logo_y}:format=auto[v1]"
            )
            v_in = "[v1]"
        else:
            v_in = "[v0]"

        vf = (
            f"[0:v]scale={REEL_W}:{REEL_H},"
            f"zoompan=z='min(1.08,1+0.0008*on)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={REEL_W}x{REEL_H},"
            f"fps=30,format=rgba[v0]"
            f"{logo_filter}"
            f";{v_in}"
            f"drawtext=fontfile={FONT_BOLD}:textfile={title_txt}:"
            f"x=80:y=1220:fontsize=64:fontcolor=white:"
            f"box=1:boxcolor=black@0.45:boxborderw=26:"
            f"alpha='if(lt(t,0.4),0, if(lt(t,0.9),(t-0.4)/0.5,1))',"
            f"drawtext=fontfile={FONT_BOLD}:textfile={cta_txt}:"
            f"x=80:y=1540:fontsize=44:fontcolor=white:"
            f"box=1:boxcolor=black@0.30:boxborderw=18:"
            f"alpha='if(lt(t,1.0),0, if(lt(t,1.4),(t-1.0)/0.4,1))'"
            f"[vout]"
        )

        cmd += [
            "-t", str(REEL_SECONDS),
            "-filter_complex", vf,
            "-map", "[vout]",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            out_mp4,
        ]

        p = subprocess.run(cmd, capture_output=True, text=True, timeout=240, check=False)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg falló:\n{(p.stderr or '')[:2000]}")

        with open(out_mp4, "rb") as f:
            return f.read()
