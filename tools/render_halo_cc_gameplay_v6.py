#!/usr/bin/env python3
"""Render the internal Halo pilot with licensed support footage."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1080
HEIGHT = 1920
FPS = 30
FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")


@dataclass(frozen=True)
class Segment:
    source_start: float
    duration: float
    headline: str
    accent: str


SEGMENTS = (
    Segment(390.0, 0.9, "HALO CAMBIÓ", "#B5FF31"),
    Segment(430.0, 1.4, "3 MISIONES NUEVAS", "#35E6FF"),
    Segment(425.0, 2.0, "ANTES DEL INICIO", "#B5FF31"),
    Segment(435.0, 2.0, "OPERACIÓN METEORITO", "#FF4D8D"),
    Segment(445.0, 1.9, "MISIÓN CLANDESTINA", "#A779FF"),
    Segment(460.0, 1.4, "¿ENTRAS?", "#35E6FF"),
)


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def hex_rgba(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = value.lstrip("#")
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
        alpha,
    )


def create_overlay(
    destination: Path,
    headline: str,
    accent: str,
    active_index: int,
) -> None:
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    bold_30 = font(FONT_BOLD, 30)
    bold_72 = font(FONT_BOLD, 72)
    regular_25 = font(FONT_REGULAR, 25)
    regular_27 = font(FONT_REGULAR, 27)
    accent_color = hex_rgba(accent)

    draw.rounded_rectangle(
        (58, 84, 350, 138),
        radius=12,
        fill=(5, 12, 17, 190),
        outline=hex_rgba(accent, 220),
        width=2,
    )
    draw.text((82, 94), "RADAR // HALO", font=bold_30, fill=(245, 249, 250, 255))

    draw.rounded_rectangle(
        (58, 190, 1022, 354),
        radius=26,
        fill=(2, 7, 11, 190),
        outline=(255, 255, 255, 38),
        width=2,
    )
    draw.rounded_rectangle(
        (58, 190, 74, 354),
        radius=8,
        fill=accent_color,
    )
    headline_font = bold_72
    if len(headline) > 19:
        headline_font = font(FONT_BOLD, 62)
    text_box = draw.textbbox((0, 0), headline, font=headline_font)
    text_height = text_box[3] - text_box[1]
    text_y = 272 - text_height / 2 - text_box[1]
    draw.text(
        (104, text_y),
        headline,
        font=headline_font,
        fill=(248, 250, 251, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0, 190),
    )

    dot_start = 754
    for index in range(len(SEGMENTS)):
        dot_color = accent_color if index == active_index else (255, 255, 255, 75)
        left = dot_start + index * 43
        draw.rounded_rectangle(
            (left, 105, left + 28, 115),
            radius=5,
            fill=dot_color,
        )

    draw.rectangle((0, 1580, WIDTH, HEIGHT), fill=(1, 5, 8, 255))
    draw.rectangle((0, 1580, WIDTH, 1588), fill=accent_color)
    draw.text(
        (58, 1620),
        "IMÁGENES DE APOYO · HALO INFINITE",
        font=regular_27,
        fill=(247, 249, 250, 255),
    )
    draw.text(
        (58, 1664),
        "Xbox México · CC BY 3.0 · edición La Estratosférica",
        font=regular_25,
        fill=(195, 207, 212, 255),
    )
    draw.text(
        (58, 1710),
        "PROTOTIPO INTERNO · PUBLICACIÓN NO HABILITADA",
        font=bold_30,
        fill=accent_color,
    )

    image.save(destination)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def render(
    source: Path,
    narration: Path,
    output: Path,
    overlay_directory: Path,
) -> None:
    overlay_directory.mkdir(parents=True, exist_ok=True)
    overlays: list[Path] = []
    for index, segment in enumerate(SEGMENTS):
        overlay = overlay_directory / f"overlay-{index + 1}.png"
        create_overlay(overlay, segment.headline, segment.accent, index)
        overlays.append(overlay)

    command = ["ffmpeg", "-hide_banner", "-y"]
    for segment in SEGMENTS:
        command.extend(
            [
                "-ss",
                f"{segment.source_start:.3f}",
                "-t",
                f"{segment.duration:.3f}",
                "-i",
                str(source),
            ]
        )
    command.extend(["-i", str(narration)])
    for overlay in overlays:
        command.extend(
            [
                "-loop",
                "1",
                "-framerate",
                str(FPS),
                "-t",
                "9.6",
                "-i",
                str(overlay),
            ]
        )

    filters: list[str] = []
    for index in range(len(SEGMENTS)):
        filters.append(
            f"[{index}:v]fps={FPS},"
            f"scale=-2:{HEIGHT},crop={WIDTH}:{HEIGHT},setsar=1,"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
        filters.append(
            f"[{index}:a]aformat=sample_rates=48000:"
            f"channel_layouts=stereo,asetpts=PTS-STARTPTS[a{index}]"
        )

    video_inputs = "".join(f"[v{index}]" for index in range(len(SEGMENTS)))
    audio_inputs = "".join(f"[a{index}]" for index in range(len(SEGMENTS)))
    filters.append(
        f"{video_inputs}concat=n={len(SEGMENTS)}:v=1:a=0[vbase]"
    )
    filters.append(
        f"{audio_inputs}concat=n={len(SEGMENTS)}:v=0:a=1,"
        "volume=0.055,highpass=f=120,lowpass=f=6500[game]"
    )

    previous_video = "vbase"
    overlay_offset = len(SEGMENTS) + 1
    timeline_start = 0.0
    for index, segment in enumerate(SEGMENTS):
        timeline_end = timeline_start + segment.duration
        output_label = f"vo{index}"
        filters.append(
            f"[{previous_video}][{overlay_offset + index}:v]"
            f"overlay=0:0:enable='between(t,{timeline_start:.3f},"
            f"{timeline_end:.3f})'[{output_label}]"
        )
        previous_video = output_label
        timeline_start = timeline_end

    narration_index = len(SEGMENTS)
    filters.append(
        f"[{narration_index}:a]aresample=48000,"
        "acompressor=threshold=-18dB:ratio=3:attack=5:release=50,"
        "volume=1.18,apad=pad_dur=0.3,atrim=duration=9.6[voice]"
    )
    filters.append(
        "[game][voice]amix=inputs=2:duration=longest:normalize=0,"
        "volume=1.5,alimiter=limit=0.95[mix]"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            f"[{previous_video}]",
            "-map",
            "[mix]",
            "-t",
            "9.6",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    run(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--narration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overlay-directory", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(
        source=args.source,
        narration=args.narration,
        output=args.output,
        overlay_directory=args.overlay_directory,
    )


if __name__ == "__main__":
    main()
