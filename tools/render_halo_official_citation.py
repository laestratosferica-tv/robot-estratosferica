#!/usr/bin/env python3
"""Render a review-only Halo short using documented official trailer excerpts."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1080
HEIGHT = 1920
FPS = 30
DURATION = 9.6
TRANSITION = 0.12
GAMEPLAY_Y = 500
GAMEPLAY_HEIGHT = 608
FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")


@dataclass(frozen=True)
class Segment:
    source_start: float
    duration: float
    headline: str
    accent: str


SEGMENTS = (
    Segment(49.0, 1.02, "HALO CAMBIÓ", "#B5FF31"),
    Segment(51.0, 1.52, "3 MISIONES NUEVAS", "#35E6FF"),
    Segment(53.0, 2.12, "ANTES DEL INICIO", "#B5FF31"),
    Segment(56.5, 2.12, "JEFE + JOHNSON", "#FF4D8D"),
    Segment(65.0, 2.02, "MISIÓN CLANDESTINA", "#A779FF"),
    Segment(136.5, 1.4, "¿ENTRAS?", "#35E6FF"),
)

FOCUS_X = (0.56, 0.64, 0.67, 0.50, 0.63, 0.50)


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
    full_bleed: bool = False,
    public_candidate: bool = False,
) -> None:
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    accent_color = hex_rgba(accent)
    label_font = font(FONT_BOLD, 28)
    headline_font = font(FONT_BOLD, 78 if len(headline) < 18 else 64)
    source_font = font(FONT_REGULAR, 23)
    warning_font = font(FONT_BOLD, 24)

    if full_bleed:
        draw.rectangle((0, 0, WIDTH, 458), fill=(1, 5, 8, 178))
        draw.rectangle((0, 1628, WIDTH, HEIGHT), fill=(1, 5, 8, 205))
    draw.rounded_rectangle(
        (58, 92, 538, 148),
        radius=14,
        fill=(2, 8, 12, 210),
        outline=accent_color,
        width=2,
    )
    draw.text(
        (82, 102),
        (
            "OPERACIÓN: METEORITE"
            if public_candidate
            else "RADAR // CITA EDITORIAL"
        ),
        font=label_font,
        fill=(246, 249, 250, 255),
    )

    text_box = draw.textbbox((0, 0), headline, font=headline_font)
    text_width = text_box[2] - text_box[0]
    text_x = max(58, (WIDTH - text_width) // 2)
    draw.text(
        (text_x, 280),
        headline,
        font=headline_font,
        fill=(249, 251, 252, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 230),
    )
    draw.rounded_rectangle(
        (58, 402, 1022, 414),
        radius=6,
        fill=accent_color,
    )

    if not full_bleed:
        draw.rounded_rectangle(
            (40, GAMEPLAY_Y - 8, 1040, GAMEPLAY_Y + GAMEPLAY_HEIGHT + 8),
            radius=22,
            outline=hex_rgba(accent, 235),
            width=4,
        )

    draw.rounded_rectangle(
        (44, 1684, 1036, 1838),
        radius=22,
        fill=(1, 6, 10, 220),
        outline=(255, 255, 255, 38),
        width=2,
    )
    draw.text(
        (72, 1712),
        "Fuente audiovisual: HALO · tráiler oficial",
        font=source_font,
        fill=(233, 239, 241, 255),
    )
    draw.text(
        (72, 1754),
        "Fragmento transformado · audio original eliminado",
        font=source_font,
        fill=(181, 198, 205, 255),
    )
    draw.text(
        (72, 1795),
        (
            "LA ESTRATOSFÉRICA · RADAR GAMER"
            if public_candidate
            else "BORRADOR INTERNO · PUBLICACIÓN NO HABILITADA"
        ),
        font=warning_font,
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
    full_bleed: bool = False,
    public_candidate: bool = False,
) -> None:
    overlay_directory.mkdir(parents=True, exist_ok=True)
    overlays: list[Path] = []
    for index, segment in enumerate(SEGMENTS):
        overlay = overlay_directory / f"overlay-{index + 1}.png"
        create_overlay(
            overlay,
            segment.headline,
            segment.accent,
            full_bleed=full_bleed,
            public_candidate=public_candidate,
        )
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
                f"{DURATION:.3f}",
                "-i",
                str(overlay),
            ]
        )
    command.extend(
        [
            "-f",
            "lavfi",
            "-t",
            f"{DURATION:.3f}",
            "-i",
            "sine=frequency=58:sample_rate=48000",
        ]
    )

    filters: list[str] = []
    for index in range(len(SEGMENTS)):
        if full_bleed:
            focus = FOCUS_X[index]
            filters.append(
                f"[{index}:v]fps={FPS},"
                f"scale=-2:{HEIGHT},"
                f"crop={WIDTH}:{HEIGHT}:"
                f"x='min(max(iw*{focus:.2f}-{WIDTH / 2:.1f},0),"
                f"iw-{WIDTH})':y=0,"
                "eq=contrast=1.08:saturation=1.12,"
                f"setsar=1,setpts=PTS-STARTPTS[scene{index}]"
            )
        else:
            filters.append(
                f"[{index}:v]fps={FPS},split=2[bgraw{index}][fgraw{index}]"
            )
            filters.append(
                f"[bgraw{index}]"
                f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
                f"crop={WIDTH}:{HEIGHT},gblur=sigma=34,"
                "eq=brightness=-0.28:saturation=0.85,"
                f"setpts=PTS-STARTPTS[bg{index}]"
            )
            filters.append(
                f"[fgraw{index}]"
                f"scale={WIDTH}:{GAMEPLAY_HEIGHT}:"
                "force_original_aspect_ratio=increase,"
                f"crop={WIDTH}:{GAMEPLAY_HEIGHT},"
                "eq=contrast=1.08:saturation=1.12,"
                f"setpts=PTS-STARTPTS[fg{index}]"
            )
            filters.append(
                f"[bg{index}][fg{index}]"
                f"overlay=0:{GAMEPLAY_Y}:shortest=1,"
                f"setsar=1[scene{index}]"
            )

    previous_scene = "scene0"
    composed_duration = SEGMENTS[0].duration
    for index in range(1, len(SEGMENTS)):
        output_label = f"blend{index}"
        transition_offset = composed_duration - TRANSITION
        filters.append(
            f"[{previous_scene}][scene{index}]"
            f"xfade=transition=fade:duration={TRANSITION}:"
            f"offset={transition_offset:.3f},setsar=1[{output_label}]"
        )
        previous_scene = output_label
        composed_duration += SEGMENTS[index].duration - TRANSITION

    previous_video = previous_scene
    overlay_offset = len(SEGMENTS) + 1
    timeline_start = 0.0
    for index, segment in enumerate(SEGMENTS):
        timeline_end = (
            DURATION
            if index == len(SEGMENTS) - 1
            else timeline_start + segment.duration - TRANSITION
        )
        output_label = f"vo{index}"
        filters.append(
            f"[{previous_video}][{overlay_offset + index}:v]"
            f"overlay=0:0:enable='between(t,{timeline_start:.3f},"
            f"{timeline_end:.3f})'[{output_label}]"
        )
        previous_video = output_label
        timeline_start = timeline_end

    narration_index = len(SEGMENTS)
    hum_index = len(SEGMENTS) + 1 + len(SEGMENTS)
    filters.append(
        f"[{narration_index}:a]aresample=48000,"
        "acompressor=threshold=-18dB:ratio=3:attack=5:release=50,"
        f"volume=1.20,apad=pad_dur=0.3,atrim=duration={DURATION}[voice]"
    )
    filters.append(
        f"[{hum_index}:a]volume=0.035,"
        "afade=t=in:st=0:d=0.25,afade=t=out:st=9.1:d=0.5[hum]"
    )
    filters.append(
        "[voice][hum]amix=inputs=2:duration=longest:normalize=0,"
        "alimiter=limit=0.95[mix]"
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
            f"{DURATION:.3f}",
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
    parser.add_argument("--full-bleed", action="store_true")
    parser.add_argument("--public-candidate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(
        source=args.source,
        narration=args.narration,
        output=args.output,
        overlay_directory=args.overlay_directory,
        full_bleed=args.full_bleed,
        public_candidate=args.public_candidate,
    )


if __name__ == "__main__":
    main()
