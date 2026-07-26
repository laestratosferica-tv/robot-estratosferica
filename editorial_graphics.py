import io
import os
import textwrap
from typing import Optional

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


PILLAR_COLORS = {
    "gaming": "#A855F7",
    "technology": "#00D4FF",
    "advertising": "#FF3D8D",
    "fashion": "#FF7A00",
    "gastronomy": "#FFCB45",
    "lifestyle": "#3DE2B4",
    "luxury": "#D8B45B",
    "monetization": "#B7FF3C",
}

PILLAR_LABELS = {
    "gaming": "GAMING + ESPORTS",
    "technology": "TECNOLOGÍA + IA",
    "advertising": "PUBLICIDAD + CREATIVIDAD",
    "fashion": "MODA + CULTURA URBANA",
    "gastronomy": "GASTRONOMÍA + EXPERIENCIAS",
    "lifestyle": "ESTILO DE VIDA DIGITAL",
    "luxury": "LUJO CONTEMPORÁNEO",
    "monetization": "NEGOCIOS + OPORTUNIDADES",
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def _fit_headline(draw: ImageDraw.ImageDraw, text: str, max_width: int, max_lines: int = 4):
    clean = " ".join((text or "MIRA ESTO").upper().split())
    for size in range(88, 43, -4):
        font = _font(size, bold=True)
        approx_chars = max(10, int(max_width / (size * 0.58)))
        lines = textwrap.wrap(clean, width=approx_chars)
        if len(lines) <= max_lines and all(draw.textbbox((0, 0), line, font=font)[2] <= max_width for line in lines):
            return font, lines
    return _font(44, bold=True), textwrap.wrap(clean, width=24)[:max_lines]


def build_threads_card(
    image_bytes: bytes,
    headline: str,
    badge_text: str,
    pillar: str,
    logo_path: Optional[str] = None,
    width: int = 1080,
    height: int = 1350,
) -> bytes:
    source = Image.open(io.BytesIO(image_bytes))
    background = _cover(source, (width, height))
    background = ImageEnhance.Contrast(background).enhance(1.08)
    background = ImageEnhance.Color(background).enhance(0.92)

    blurred = background.filter(ImageFilter.GaussianBlur(18))
    background = Image.blend(background, blurred, 0.16).convert("RGBA")

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for y in range(height):
        alpha = int(25 + (190 * (y / height) ** 1.7))
        od.line((0, y, width, y), fill=(5, 6, 14, min(225, alpha)))
    background = Image.alpha_composite(background, overlay)

    draw = ImageDraw.Draw(background)
    color = PILLAR_COLORS.get(pillar, PILLAR_COLORS["gaming"])
    label = PILLAR_LABELS.get(pillar, PILLAR_LABELS["gaming"])

    draw.rectangle((0, 0, 18, height), fill=color)
    draw.rounded_rectangle((58, 58, 380, 118), radius=20, fill=(8, 10, 22, 225), outline=color, width=3)
    draw.text((82, 72), "LA ESTRATOSFÉRICA", font=_font(27, bold=True), fill="white")

    badge = (badge_text or "HOT").upper()[:18]
    badge_font = _font(28, bold=True)
    badge_box = draw.textbbox((0, 0), badge, font=badge_font)
    badge_w = badge_box[2] - badge_box[0]
    draw.rounded_rectangle((width - badge_w - 132, 62, width - 62, 116), radius=18, fill=color)
    draw.text((width - badge_w - 98, 72), badge, font=badge_font, fill="#080A16")

    font, lines = _fit_headline(draw, headline, width - 136)
    line_h = int(font.size * 1.05)
    y = height - 455
    for line in lines:
        draw.text((70, y + 5), line, font=font, fill=(0, 0, 0, 170), stroke_width=5, stroke_fill=(0, 0, 0, 170))
        draw.text((70, y), line, font=font, fill="white", stroke_width=1, stroke_fill="white")
        y += line_h

    label_font = _font(24, bold=True)
    label_box = draw.textbbox((0, 0), label, font=label_font)
    label_w = label_box[2] - label_box[0]
    draw.rounded_rectangle((70, height - 105, 120 + label_w, height - 55), radius=16, fill=(8, 10, 22, 225), outline=color, width=2)
    draw.text((94, height - 94), label, font=label_font, fill=color)

    if logo_path and os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert("RGBA")
            logo.thumbnail((120, 120), Image.Resampling.LANCZOS)
            background.alpha_composite(logo, (width - logo.width - 62, height - logo.height - 48))
        except Exception:
            pass

    output = io.BytesIO()
    background.convert("RGB").save(output, format="PNG", optimize=True)
    return output.getvalue()
