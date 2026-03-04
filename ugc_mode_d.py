import os
import subprocess
import tempfile
import random
import hashlib
from datetime import datetime

from ugc_mode_b import (
    r2_client,
    s3_get_bytes,
    s3_put_bytes,
    r2_public_url,
    openai_text,
    ig_publish_reel,
)

BUCKET_NAME = os.getenv("BUCKET_NAME")

LIBRARY_PREFIX = "ugc/library/raw/"
VIRAL_PREFIX = "ugc/library/viral/"


def short_hash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def detect_segments(duration):

    segments = []

    if duration > 45:

        segments.append((5, 10))
        segments.append((duration/3, 10))
        segments.append((duration*0.7, 10))

    else:

        segments.append((0, 8))
        segments.append((duration/2, 8))

    return segments


def make_clip(input_path, output_path, start, duration):

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        input_path,
        "-t",
        str(duration),
        "-vf",
        "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    subprocess.run(cmd, check=True)


def generate_hook_caption(filename, style):

    prompt = f"""
Eres editor viral de esports.

Tipo de hook: {style}

Haz caption corto polémico para reel gaming.

Debe tener:

1 frase fuerte
1 pregunta final
hashtags gaming

video: {filename}
"""

    return openai_text(prompt)


def run_mode_d():

    print("===== UGC MODE D VIRAL ENGINE =====")

    s3 = r2_client()

    objs = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=LIBRARY_PREFIX
    ).get("Contents", [])

    if not objs:

        print("No videos en library")
        return

    for obj in objs:

        key = obj["Key"]

        if not key.endswith(".mp4"):
            continue

        print("Analizando video:", key)

        video_bytes = s3_get_bytes(key)

        with tempfile.TemporaryDirectory() as td:

            input_path = f"{td}/video.mp4"

            with open(input_path, "wb") as f:
                f.write(video_bytes)

            duration = 60

            segments = detect_segments(duration)

            hook_styles = [

                "drama",
                "controversia",
                "misterio",
                "hype",
                "humor"

            ]

            for i, seg in enumerate(segments):

                start, dur = seg

                output = f"{td}/clip{i}.mp4"

                make_clip(input_path, output, start, dur)

                style = random.choice(hook_styles)

                caption = generate_hook_caption(key, style)

                clip_bytes = open(output, "rb").read()

                clip_key = f"{VIRAL_PREFIX}{short_hash(key)}_{i}.mp4"

                s3_put_bytes(clip_key, clip_bytes, "video/mp4")

                url = r2_public_url(clip_key)

                ig_publish_reel(url, caption)

                print("Publicado clip viral:", url)

    print("===== MODE D DONE =====")
