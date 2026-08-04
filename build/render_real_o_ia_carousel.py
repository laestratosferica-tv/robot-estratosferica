#!/usr/bin/env python3
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1080, 1350
FONT_B = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
FONT_R = "/System/Library/Fonts/Supplemental/Arial.ttf"
WHITE = "#FFFFFF"
CYAN = "#7DEBFF"
MAGENTA = "#FF28C9"
MUTED = "#D8D4E8"
INK = "#080316"

def font(size, bold=True):
    return ImageFont.truetype(FONT_B if bold else FONT_R, size)

def fit_base():
    im = Image.open(SRC).convert("RGB")
    scale = max(W / im.width, H / im.height)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
    left = (im.width - W) // 2
    top = (im.height - H) // 2
    im = im.crop((left, top, left + W, top + H))
    return ImageEnhance.Color(ImageEnhance.Contrast(im).enhance(1.05)).enhance(1.08)

def box(draw, xy, color, alpha=255, width=0):
    draw.rectangle(xy, fill=(*ImageColor.getrgb(color), alpha) if width == 0 else None,
                   outline=(*ImageColor.getrgb(color), alpha), width=width)

from PIL import ImageColor

def overlay(im, xy, color, alpha):
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(lay).rectangle(xy, fill=(*ImageColor.getrgb(color), alpha))
    return Image.alpha_composite(im.convert("RGBA"), lay)

def text(draw, xy, value, size, color=WHITE, bold=True, spacing=4):
    draw.multiline_text(xy, value, font=font(size, bold), fill=color, spacing=spacing)

def frame(im):
    d = ImageDraw.Draw(im)
    d.rectangle((42, 42, 1038, 1308), outline=(125, 235, 255, 120), width=3)
    return d

base = fit_base()

# 1 — portada
im = overlay(base, (0, 0, W, H), INK, 55)
d = frame(im)
d.rectangle((46, 48, 1034, 58), fill=MAGENTA)
text(d, (66, 80), "LA ESTRATOSFÉRICA  •  RETO VISUAL", 25, CYAN, False)
text(d, (66, 146), "UNO DE ESTOS", 82)
text(d, (66, 235), "SETUPS NO EXISTE", 76, MAGENTA)
d.rounded_rectangle((66, 1070, 1014, 1220), 16, fill=(8, 3, 22, 220))
text(d, (94, 1092), "MIRA LOS DETALLES", 48)
text(d, (94, 1160), "Desliza. El error está a simple vista.", 34, MUTED, False)
d.rectangle((66, 1260, 366, 1267), fill="#20E5F2")
im.convert("RGB").save(OUT / "01-portada.png", quality=95)

# 2 — A
im = overlay(base, (0, 0, 540, H), INK, 25)
d = frame(im)
d.rounded_rectangle((56, 62, 172, 178), 12, fill="#20E5F2")
text(d, (91, 68), "A", 76, INK)
text(d, (196, 76), "¿REAL?", 64)
d.rounded_rectangle((56, 1120, 1024, 1252), 14, fill=(8, 3, 22, 225))
text(d, (86, 1140), "BUSCA UNA PISTA", 48)
text(d, (86, 1200), "Reflejos  •  cables  •  controles", 31, MUTED, False)
im.convert("RGB").save(OUT / "02-opcion-a.png", quality=95)

# 3 — B
im = overlay(base, (540, 0, W, H), INK, 25)
d = frame(im)
d.rounded_rectangle((56, 62, 172, 178), 12, fill=MAGENTA)
text(d, (90, 68), "B", 76)
text(d, (196, 76), "¿IA?", 64)
d.rounded_rectangle((56, 1120, 1024, 1252), 14, fill=(8, 3, 22, 225))
text(d, (86, 1140), "DECIDE ANTES DE SEGUIR", 42)
text(d, (86, 1200), "¿A o B? Guarda tu respuesta.", 31, MUTED, False)
im.convert("RGB").save(OUT / "03-opcion-b.png", quality=95)

# Variantes limpias para el momento de decisión del video.
for letter, color, name in (("A", "#20E5F2", "video-02-a.png"), ("B", MAGENTA, "video-03-b.png")):
    im = overlay(base, (0, 0, W, H), INK, 68)
    d = frame(im)
    bbox = d.textbbox((0, 0), letter, font=font(390))
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.rounded_rectangle((W // 2 - 260, H // 2 - 290, W // 2 + 260, H // 2 + 290), 34,
                        fill=(8, 3, 22, 205), outline=color, width=8)
    d.text(((W - tw) // 2, (H - th) // 2 - 35), letter, font=font(390), fill=color)
    im.convert("RGB").save(OUT / name, quality=95)

# 4 — revelación
im = overlay(base, (0, 0, W, H), INK, 120)
d = frame(im)
text(d, (66, 76), "RESPUESTA", 28, CYAN, False)
text(d, (66, 135), "B FUE GENERADO", 72)
text(d, (66, 217), "CON IA", 96, MAGENTA)
d.rounded_rectangle((565, 420, 995, 780), 16, fill=(8, 3, 22, 225), outline=MAGENTA, width=4)
text(d, (600, 452), "PISTAS", 44, CYAN)
text(d, (600, 530), "• cableado imposible\n• conexión sin lógica\n• control inconsistente", 31, WHITE, False, 26)
text(d, (66, 1120), "¿TE ENGAÑÓ?", 66)
text(d, (68, 1205), "Cuéntanos qué elegiste.", 36, MUTED, False)
im.convert("RGB").save(OUT / "04-revelacion.png", quality=95)

# 5 — contexto y CTA
im = Image.new("RGB", (W, H), INK).convert("RGBA")
d = ImageDraw.Draw(im)
d.rounded_rectangle((42, 42, 1038, 1308), 24, fill="#24113E", outline="#7DEBFF", width=3)
d.rectangle((42, 42, 1038, 52), fill="#20E5F2")
text(d, (68, 100), "TRANSPARENCIA DIGITAL", 28, CYAN, False)
text(d, (68, 200), "LA IA DEBE", 78)
text(d, (68, 286), "IDENTIFICARSE", 82, MAGENTA)
text(d, (70, 460), "Meta avanza en reglas para identificar\ncontenido generado con inteligencia artificial.", 38, WHITE, False, 14)
d.rounded_rectangle((68, 640, 1012, 830), 16, fill=(8, 3, 22, 190))
text(d, (102, 692), "VER  →  ENTENDER  →  DECIDIR", 42, CYAN)
text(d, (68, 1005), "¿DEBERÍA MARCARSE SIEMPRE?", 50)
text(d, (70, 1090), "Comenta: SÍ o DEPENDE", 40, MUTED, False)
text(d, (70, 1240), "Fuente oficial: Meta", 25, "#8F88A5", False)
im.convert("RGB").save(OUT / "05-conversacion.png", quality=95)

print(f"Carrusel generado en {OUT}")
