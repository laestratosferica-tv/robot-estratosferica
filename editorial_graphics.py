import io
import os
import textwrap
from typing import Optional

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from visual_identity import get_visual_direction


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


def _fit_headline(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_lines: int = 4,
    scale: float = 1.0,
):
    clean = " ".join((text or "MIRA ESTO").upper().split())
    start = max(58, int(88 * scale))
    stop = max(38, int(43 * scale))
    for size in range(start, stop, -4):
        font = _font(size, bold=True)
        approx_chars = max(10, int(max_width / (size * 0.58)))
        lines = textwrap.wrap(clean, width=approx_chars)
        if len(lines) <= max_lines and all(draw.textbbox((0, 0), line, font=font)[2] <= max_width for line in lines):
            return font, lines
    return _font(stop, bold=True), textwrap.wrap(clean, width=24)[:max_lines]


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def _mix_rgb(start: str, end: str, amount: float) -> tuple[int, int, int]:
    left = _hex_rgb(start)
    right = _hex_rgb(end)
    return tuple(round(a + (b - a) * amount) for a, b in zip(left, right))


def _draw_orbital_signature(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    gradient: list[str],
) -> None:
    box = (-150, 42, width + 150, 470)
    segments = 80
    for index in range(segments):
        progress = index / max(1, segments - 1)
        if progress < 0.5:
            color = _mix_rgb(gradient[0], gradient[1], progress * 2)
        else:
            color = _mix_rgb(gradient[1], gradient[2], (progress - 0.5) * 2)
        start = 198 + index * (144 / segments)
        end = start + (160 / segments)
        line_width = max(2, round(3 + 7 * progress))
        draw.arc(box, start=start, end=end, fill=color, width=line_width)
    end_color = _hex_rgb(gradient[-1])
    draw.ellipse((width - 86, 188, width - 62, 212), fill=end_color)


def _draw_category_texture(
    draw: ImageDraw.ImageDraw,
    layout: str,
    color: str,
    width: int,
    height: int,
) -> None:
    rgb = _hex_rgb(color)
    if layout in {"kinetic", "signal"}:
        for index in range(4):
            x = width - 250 + index * 38
            draw.line((x, 250, x + 130, 120), fill=(*rgb, 120), width=5)
    elif layout == "precision":
        for x in range(70, width, 110):
            draw.ellipse((x, 280, x + 3, 283), fill=(*rgb, 110))
    elif layout in {"editorial", "spacious"}:
        draw.line((70, 255, width - 70, 255), fill=(*rgb, 100), width=2)
    elif layout == "sensory":
        draw.ellipse((width - 210, 220, width - 70, 360), outline=(*rgb, 95), width=3)
    elif layout == "editorial_play":
        draw.line((70, 260, 195, 260), fill=(*rgb, 150), width=8)
        draw.line((205, 260, 270, 260), fill=(*rgb, 70), width=8)
    else:
        draw.rounded_rectangle(
            (70, 235, width - 70, height - 160),
            radius=34,
            outline=(*rgb, 65),
            width=2,
        )


def build_threads_card(
    image_bytes: bytes,
    headline: str,
    badge_text: str,
    pillar: str,
    logo_path: Optional[str] = None,
    trend_profile: str = "evergreen",
    width: int = 1080,
    height: int = 1350,
) -> bytes:
    direction = get_visual_direction(pillar, trend_profile)
    source = Image.open(io.BytesIO(image_bytes))
    background = _cover(source, (width, height))
    background = ImageEnhance.Contrast(background).enhance(direction["contrast"])
    background = ImageEnhance.Color(background).enhance(direction["saturation"])

    blurred = background.filter(ImageFilter.GaussianBlur(18))
    background = Image.blend(background, blurred, direction["blur_mix"]).convert("RGBA")

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for y in range(height):
        alpha = int(20 + (195 * (y / height) ** 1.7))
        od.line((0, y, width, y), fill=(5, 6, 14, min(225, alpha)))
    background = Image.alpha_composite(background, overlay)

    draw = ImageDraw.Draw(background)
    color = direction["accent"]
    secondary = direction["secondary"]
    label = PILLAR_LABELS.get(pillar, PILLAR_LABELS["gaming"])

    _draw_orbital_signature(
        draw,
        width,
        height,
        direction["brand"]["core_gradient"],
    )
    _draw_category_texture(draw, direction["layout"], secondary, width, height)
    draw.rounded_rectangle((58, 58, 424, 118), radius=20, fill=(8, 10, 22, 225), outline=color, width=2)
    draw.text((82, 69), "LETV  LA ESTRATOSFÉRICA", font=_font(25, bold=True), fill="white")

    badge = (badge_text or "HOT").upper()[:18]
    badge_font = _font(28, bold=True)
    badge_box = draw.textbbox((0, 0), badge, font=badge_font)
    badge_w = badge_box[2] - badge_box[0]
    draw.rounded_rectangle((width - badge_w - 132, 62, width - 62, 116), radius=18, fill=color)
    draw.text((width - badge_w - 98, 72), badge, font=badge_font, fill="#080A16")

    max_width = width - 210 if direction["layout"] == "spacious" else width - 136
    font, lines = _fit_headline(
        draw,
        headline,
        max_width,
        scale=direction["headline_scale"],
    )
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

    trend_accent = direction["trend"]["accent"]
    if direction["trend_profile"] != "evergreen":
        draw.line((width - 270, height - 64, width - 170, height - 64), fill=trend_accent, width=5)

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
