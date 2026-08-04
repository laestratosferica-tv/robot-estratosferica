#!/usr/bin/env python3
"""Build and schedule the autonomous gamer-humor experiment for Aug 4–20, 2026."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/approved/scheduled/gamer-viral-daily-2026-08"
MANIFESTS = ROOT / "artifacts/publication-manifests"
QUEUE = ROOT / "config/scheduled_publications_v1.json"
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
TZ = timezone(timedelta(hours=-5))


PIECES = [
    ("vr-dignidad", "https://videos.pexels.com/video-files/6499156/6499156-uhd_2560_1440_25fps.mp4", "https://www.pexels.com/video/people-playing-video-game-in-virtual-reality-mode-6499156/", "Tima Miroshnichenko", "TU CUERPO SIGUE AQUÍ", "Tu dignidad ya entró al juego.", "Juan", "human"),
    ("enemigo-invisible", "https://videos.pexels.com/video-files/7986123/7986123-hd_1920_1080_25fps.mp4", "https://www.pexels.com/video/man-playing-while-using-virtual-reality-headset-7986123/", "Mikhail Nilov", "DESDE AFUERA SE VE RARO", "Dentro del visor, él está salvando el mundo.", "Paulina", "machine"),
    ("celebracion-mundial", "https://videos.pexels.com/video-files/35782229/15170238_1440_2560_25fps.mp4", "https://www.pexels.com/video/competitive-gamer-celebrating-victory-35782229/", "Themba Mtegha", "GANÓ UNA PARTIDA NORMAL", "Su cerebro entendió: final mundial.", "Juan", "human"),
    ("control-en-peligro", "https://videos.pexels.com/video-files/35782226/15170234_1440_2560_25fps.mp4", "https://www.pexels.com/video/frustrated-gamer-with-red-mood-lighting-35782226/", "Themba Mtegha", "PERDIÓ LA PARTIDA", "El control acaba de entrar en zona de riesgo.", "Paulina", "machine"),
    ("retro-sin-lag", "https://videos.pexels.com/video-files/8888991/8888991-uhd_2732_1440_25fps.mp4", "https://www.pexels.com/video/friends-playing-video-games-8888991/", "MART PRODUCTION", "ANTES NO HABÍA LAG", "Había un amigo tapando la pantalla.", "Juan", "human"),
    ("culpa-del-control", "https://videos.pexels.com/video-files/7668244/7668244-hd_1080_1920_25fps.mp4", "https://www.pexels.com/video/friends-playing-a-video-game-7668244/", "Pavel Danilyuk", "TODOS TENEMOS ESE AMIGO", "Pierde él. La culpa es del control.", "Paulina", "alien"),
    ("estrategia-tres-segundos", "https://videos.pexels.com/video-files/7856590/7856590-uhd_2732_1440_25fps.mp4", "https://www.pexels.com/video/people-playing-video-games-7856590/", "Ron Lach", "HABÍA UNA ESTRATEGIA", "Duró exactamente tres segundos.", "Juan", "machine"),
    ("cuatro-cerebros", "https://videos.pexels.com/video-files/8128212/8128212-uhd_2560_1440_25fps.mp4", "https://www.pexels.com/video/friends-playing-video-games-8128212/", "Alena Darmel", "CUATRO CEREBROS", "Una sola idea. Y era malísima.", "Paulina", "human"),
    ("vr-nueva-generacion", "https://videos.pexels.com/video-files/8174485/8174485-uhd_2560_1440_25fps.mp4", "https://www.pexels.com/video/boy-playing-games-using-virtual-reality-8174485/", "Kampus Production", "ELLOS YA NO MIRAN EL JUEGO", "Ahora entran directamente.", "Juan", "alien"),
    ("gamer-sin-pantalla", "https://videos.pexels.com/video-files/12715357/12715357-uhd_1440_2732_30fps.mp4", "https://www.pexels.com/video/low-angle-view-of-a-young-man-playing-video-games-12715357/", "ROMAN ODINTSOV", "NO HAY PANTALLA", "Pero juega con una confianza preocupante.", "Paulina", "human"),
    ("jefe-final-sala", "https://videos.pexels.com/video-files/7774360/uhd_30fps.mp4", "https://www.pexels.com/video/kids-playing-virtual-reality-game-7774360/", "Artem Podrez", "EL JEFE FINAL NO ESTABA EN VR", "Estaba caminando por la sala.", "Juan", "machine"),
    ("ultima-partida", "https://videos.pexels.com/video-files/7774460/7774460-uhd_2560_1440_30fps.mp4", "https://www.pexels.com/video/a-young-man-playing-a-video-game-with-a-wireless-gaming-controller-7774460/", "Artem Podrez", "DIJO: UNA ÚLTIMA", "Y activó modo campeonato mundial.", "Paulina", "human"),
    ("buscando-salida-vr", "https://videos.pexels.com/video-files/7986127/7986127-hd_1080_1920_25fps.mp4", "https://www.pexels.com/video/person-wearing-virtual-reality-headset-7986127/", "Mikhail Nilov", "ENTRÓ CINCO MINUTOS", "Todavía está buscando la salida.", "Juan", "alien"),
    ("sonrisa-antes-susto", "https://videos.pexels.com/video-files/5213007/5213007-uhd_1440_2560_25fps.mp4", "https://www.pexels.com/video/a-man-playing-a-virtual-reality-game-5213007/", "Tima Miroshnichenko", "MIRA ESA SONRISA", "Todavía no sabe lo que viene.", "Paulina", "machine"),
    ("silencio-equipo", "https://videos.pexels.com/video-files/7914800/7914800-hd_1920_1080_30fps.mp4", "https://www.pexels.com/video/men-losing-a-gaming-match-7914800/", "RDNE Stock project", "NADIE HABLA", "Todos saben exactamente quién falló.", "Juan", "human"),
    ("victoria-colectiva", "https://videos.pexels.com/video-files/7849252/7849252-uhd_1440_2732_25fps.mp4", "https://www.pexels.com/video/gamers-celebrating-a-win-7849252/", "Ron Lach", "LA JUGADA DURÓ SEGUNDOS", "La historia se contará durante años.", "Paulina", "machine"),
    ("vr-vista-lujo", "https://videos.pexels.com/video-files/8591732/8591732-uhd_2732_1440_25fps.mp4", "https://www.pexels.com/video/person-using-virtual-reality-headset-8591732/", "cottonbro studio", "PAGÓ POR ESA VISTA", "Y decidió reemplazarla con realidad virtual.", "Juan", "alien"),
]


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def make_overlay(path: Path, hook: str, payoff: str, author: str) -> None:
    image = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    hook_font = ImageFont.truetype(FONT, 66)
    payoff_font = ImageFont.truetype(FONT, 48)
    credit_font = ImageFont.truetype(FONT, 23)
    draw.rounded_rectangle((34, 82, 1046, 445), radius=28, fill=(9, 5, 31, 205), outline=(115, 40, 220, 230), width=3)
    draw.rounded_rectangle((34, 82, 50, 445), radius=8, fill=(239, 40, 200, 255))
    y = 122
    for line in fit_lines(draw, hook, hook_font, 910):
        draw.text((78, y), line, font=hook_font, fill="white", stroke_width=1, stroke_fill=(0, 0, 0, 170))
        y += 75
    y += 14
    for line in fit_lines(draw, payoff, payoff_font, 910):
        draw.text((78, y), line, font=payoff_font, fill=(69, 231, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0, 180))
        y += 57
    credit = f"VIDEO BASE: {author.upper()} / PEXELS"
    draw.rounded_rectangle((28, 1780, 1052, 1888), radius=22, fill=(7, 3, 24, 180))
    draw.text((48, 1818), credit, font=credit_font, fill=(255, 255, 255, 215))
    brand = "LA ESTRATOSFÉRICA"
    bw = draw.textbbox((0, 0), brand, font=credit_font)[2]
    draw.text((1032 - bw, 1818), brand, font=credit_font, fill="white")
    image.save(path)


def voice_filter(style: str) -> str:
    if style == "machine":
        return "highpass=f=100,lowpass=f=6500,aecho=0.8:0.22:28:0.10"
    if style == "alien":
        return "asetrate=48000*1.045,aresample=48000,aecho=0.8:0.20:45:0.12"
    return "highpass=f=90,lowpass=f=7800"


def render(index: int, piece: tuple[str, ...]) -> tuple[Path, dict]:
    slug, source, page, author, hook, payoff, voice, style = piece
    folder = OUT / f"{index:02d}-{slug}"
    folder.mkdir(parents=True, exist_ok=True)
    voice_path = folder / "voice.aiff"
    overlay_path = folder / "overlay.png"
    output = folder / f"{slug}.mp4"
    script = f"{hook}. {payoff}"
    run("say", "-v", voice, "-r", "205", "-o", str(voice_path), script)
    make_overlay(overlay_path, hook, payoff, author)
    duration = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(voice_path)
    ], text=True).strip()) + 1.0
    duration = max(7.0, min(duration, 12.5))
    vf = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
        "eq=contrast=1.07:saturation=1.12[base];"
        "[base][2:v]overlay=0:0:format=auto,"
        "fade=t=in:st=0:d=0.15,fade=t=out:st=" + f"{max(0.1, duration-0.25):.2f}" + ":d=0.25[v]"
    )
    af = f"[1:a]{voice_filter(style)},loudnorm=I=-16:TP=-1.5:LRA=7,apad=pad_dur={duration:.2f}[a]"
    run(
        "ffmpeg", "-y", "-v", "error", "-ss", "0", "-t", f"{duration:.2f}", "-i", source,
        "-i", str(voice_path), "-loop", "1", "-framerate", "30", "-i", str(overlay_path),
        "-filter_complex", vf + ";" + af,
        "-map", "[v]", "-map", "[a]", "-t", f"{duration:.2f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(output)
    )
    run("ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-")
    evidence = {
        "slug": slug,
        "category": "contenido_divertido",
        "category_label": "Contenido divertido",
        "source_page": page,
        "direct_asset": source,
        "author": author,
        "license": "Pexels License",
        "license_url": "https://www.pexels.com/license/",
        "credit": f"Video base: {author} / Pexels. Edición y narración: La Estratosférica.",
        "hook": hook,
        "payoff": payoff,
        "voice": voice,
        "voice_style": style,
        "duration_seconds": duration,
        "sha256": sha(output),
    }
    (folder / "SOURCE_AND_LICENSE.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    return output, evidence


def manifest(slug: str, platform: str, video: Path, evidence: dict, publish_day: date) -> Path:
    rel = video.relative_to(ROOT).as_posix()
    credit = evidence["credit"]
    caption = f"{evidence['hook'].capitalize()}. {evidence['payoff']}\n\n{credit}\n\n#Gaming #Gamer #HumorGamer #LaEstratosferica"
    mid = f"gamer-viral-{slug}-{platform}-{publish_day.isoformat()}"
    approval = f"autonomous-gamer-viral-experiment-2026-08-03-{slug}-{platform}"
    if platform == "youtube":
        data = {"schema": "approved_youtube_short_publication_v1", "slug": mid, "approval_id": approval,
                "video_path": rel, "video_sha256": evidence["sha256"], "title": f"{evidence['hook'].title()} #Shorts",
                "description": caption + "\n\n#Shorts", "privacy_status": "public"}
    elif platform == "threads":
        data = {"schema": "approved_social_post_v1", "slug": mid, "approval_id": approval, "platform": "threads",
                "post_type": "video", "asset_path": rel, "asset_sha256": evidence["sha256"], "text": caption}
    else:
        data = {"schema": "supervised_meta_publication_v1", "slug": mid, "platform": platform,
                "approval_id": approval, "caption": caption, "video_path": rel, "video_sha256": evidence["sha256"]}
    path = MANIFESTS / f"{mid}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    queue = json.loads(QUEUE.read_text())
    queue["items"] = [x for x in queue["items"] if not x["content_id"].startswith("gamer-viral-")]
    start = date(2026, 8, 4)
    for index, piece in enumerate(PIECES, start=1):
        publish_day = start + timedelta(days=index - 1)
        video, evidence = render(index, piece)
        slug = piece[0]
        for platform in ("instagram", "facebook", "threads", "youtube"):
            path = manifest(slug, platform, video, evidence, publish_day)
            queue["items"].append({
                "content_id": f"gamer-viral-{slug}-{platform}-{publish_day.isoformat()}",
                "manifest_path": path.relative_to(ROOT).as_posix(),
                "approval_id": f"autonomous-gamer-viral-experiment-2026-08-03-{slug}-{platform}",
                "publish_at": datetime.combine(publish_day, time(2, 0), TZ).isoformat(),
                "status": "approved",
                "enabled": True,
                "experiment": "gamer-viral-daily-2am-v1",
                "category": "contenido_divertido",
                "category_label": "Contenido divertido",
                "license_evidence_path": (video.parent / "SOURCE_AND_LICENSE.json").relative_to(ROOT).as_posix(),
            })
        print(f"[{index:02d}/{len(PIECES)}] {slug} -> {publish_day.isoformat()} 02:00")
    QUEUE.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
