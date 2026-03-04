import os
import random
import subprocess
import tempfile
import hashlib
from datetime import datetime

from ugc_mode_b import (
    r2_client,
    s3_get_bytes,
    s3_put_bytes,
    r2_public_url,
    openai_text,
    ig_publish,
    fb_publish,
    tiktok_publish,
    youtube_publish,
)

BUCKET_NAME = os.getenv("BUCKET_NAME")

LIBRARY_PREFIX = "ugc/library/raw/"
CLIPS_PREFIX = "ugc/library/clips/"


# ------------------------
# helper
# ------------------------

def short_hash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:10]


# ------------------------
# clip generator
# ------------------------

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


# ------------------------
# caption generator
# ------------------------

def generate_caption(filename, hook_type):

    prompt = f"""
Eres editor viral de esports.

Tipo de hook: {hook_type}

Genera caption corto para reel gaming polémico.

Incluye:

pregunta final
hashtags esports
tono fuerte

Archivo: {filename}
"""

    return openai_text(prompt)


# ------------------------
# clip positions
# ------------------------

def clip_positions(duration):

    return [

        (0, 8),
        (duration / 3, 8),
        (duration * 0.6, 8),
    ]


# ------------------------
# main
# ------------------------

def run_mode_c():

    print("UGC MODE C START")

    s3 = r2_client()

    objs = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=LIBRARY_PREFIX
    ).get("Contents", [])

    for obj in objs:

        key = obj["Key"]

        if not key.endswith(".mp4"):
            continue

        print("Analizando:", key)

        video_bytes = s3_get_bytes(key)

        with tempfile.TemporaryDirectory() as td:

            input_path = f"{td}/video.mp4"

            open(input_path, "wb").write(video_bytes)

            duration = 60

            clips = clip_positions(duration)

            hooks = [

                "drama",
                "controversia",
                "hype",

            ]

            for i, clip in enumerate(clips):

                start, dur = clip

                output = f"{td}/clip{i}.mp4"

                make_clip(input_path, output, start, dur)

                caption = generate_caption(key, hooks[i])

                clip_bytes = open(output, "rb").read()

                clip_key = f"{CLIPS_PREFIX}{short_hash(key)}_{i}.mp4"

                s3_put_bytes(clip_key, clip_bytes, "video/mp4")

                url = r2_public_url(clip_key)

                ig_publish(url, caption)

                fb_publish(url, caption)

                tiktok_publish(url, caption)

                try:
                    youtube_publish(output, caption)
                except:
                    pass

                print("Publicado clip:", url)

    print("UGC MODE C DONE")
