#!/usr/bin/env python3
"""Apply deterministic privacy hardening to the pinned Wan worker handler."""

from __future__ import annotations

import sys
from pathlib import Path


UNSAFE_LOG = '    logger.info(f"Received job input: {job_input}")'
SAFE_LOG = (
    '    logger.info("Received job input keys: %s", '
    'sorted(str(key) for key in job_input.keys()))'
)

UNSAFE_IMAGE_INPUT = '''    if "image_path" in job_input:
        image_path = process_input(job_input["image_path"], task_id, "input_image.jpg", "path")
    elif "image_url" in job_input:
        image_path = process_input(job_input["image_url"], task_id, "input_image.jpg", "url")
    elif "image_base64" in job_input:
        image_path = process_input(job_input["image_base64"], task_id, "input_image.jpg", "base64")
    else:
        # 기본값 사용
        image_path = "/example_image.png"
        logger.info("기본 이미지 파일을 사용합니다: /example_image.png")'''

SAFE_IMAGE_INPUT = '''    if "image_base64" not in job_input:
        raise ValueError("image_base64 is required; paths and URLs are disabled")
    image_path = process_input(
        job_input["image_base64"], task_id, "input_image.jpg", "base64"
    )'''

UNSAFE_END_INPUT = '''    if "end_image_path" in job_input:
        end_image_path_local = process_input(job_input["end_image_path"], task_id, "end_image.jpg", "path")
    elif "end_image_url" in job_input:
        end_image_path_local = process_input(job_input["end_image_url"], task_id, "end_image.jpg", "url")
    elif "end_image_base64" in job_input:
        end_image_path_local = process_input(job_input["end_image_base64"], task_id, "end_image.jpg", "base64")'''

SAFE_END_INPUT = '''    if "end_image_base64" in job_input:
        end_image_path_local = process_input(
            job_input["end_image_base64"], task_id, "end_image.jpg", "base64"
        )'''

UNSAFE_DECODE = "        decoded_data = base64.b64decode(base64_data)"
SAFE_DECODE = '''        max_base64_chars = int(os.getenv("MAX_IMAGE_BASE64_CHARS", "12582912"))
        if not isinstance(base64_data, str):
            raise ValueError("Base64 input must be a string")
        if len(base64_data) > max_base64_chars:
            raise ValueError("Base64 input exceeds the configured limit")
        decoded_data = base64.b64decode(base64_data, validate=True)
        if len(decoded_data) > 9 * 1024 * 1024:
            raise ValueError("Decoded image exceeds 9 MiB")'''

V014_UNSAFE_IMAGE_INPUT = '''    if "image_path" in inp and inp["image_path"]:
        src = inp["image_path"]
        ext = os.path.splitext(src)[1] or ".png"
        name = f"input_{uuid.uuid4().hex}{ext}"
        dst = os.path.join(INPUT_DIR, name)
        shutil.copyfile(src, dst)
        return name

    if "image_url" in inp and inp["image_url"]:
        resp = requests.get(inp["image_url"], timeout=30)
        resp.raise_for_status()
        data = resp.content
    elif "image_base64" in inp and inp["image_base64"]:
        raw = inp["image_base64"]
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        data = base64.b64decode(raw)
    else:
        return None'''

V014_SAFE_IMAGE_INPUT = '''    if "image_base64" not in inp or not inp["image_base64"]:
        return None
    raw = inp["image_base64"]
    if not isinstance(raw, str):
        raise ValueError("image_base64 must be a string")
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    max_base64_chars = int(os.getenv("MAX_IMAGE_BASE64_CHARS", "12582912"))
    if len(raw) > max_base64_chars:
        raise ValueError("Base64 input exceeds the configured limit")
    data = base64.b64decode(raw, validate=True)
    if len(data) > 9 * 1024 * 1024:
        raise ValueError("Decoded image exceeds 9 MiB")'''


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} match, found {count}")
    return source.replace(old, new, 1)


def harden_text(source: str) -> str:
    if V014_UNSAFE_IMAGE_INPUT in source:
        return replace_once(
            source,
            V014_UNSAFE_IMAGE_INPUT,
            V014_SAFE_IMAGE_INPUT,
            "v0.1.4 image input block",
        )
    source = replace_once(source, UNSAFE_LOG, SAFE_LOG, "unsafe input log")
    source = replace_once(source, UNSAFE_IMAGE_INPUT, SAFE_IMAGE_INPUT, "image input block")
    source = replace_once(source, UNSAFE_END_INPUT, SAFE_END_INPUT, "end-image input block")
    source = replace_once(source, UNSAFE_DECODE, SAFE_DECODE, "base64 decoder")
    return source


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: harden_worker.py /path/to/handler.py")
    handler = Path(sys.argv[1]).resolve()
    original = handler.read_text(encoding="utf-8")
    hardened = harden_text(original)
    handler.write_text(hardened, encoding="utf-8")
    print(f"Hardened worker handler: {handler}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
