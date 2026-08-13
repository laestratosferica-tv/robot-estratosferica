FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG WAN_REPOSITORY=https://github.com/Wan-Video/Wan2.2.git
ARG WAN_REVISION=42bf4cfaa384bc21833865abc2f9e6c0e67233dc

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WAN_SOURCE_DIR=/opt/Wan2.2 \
    WAN_MODEL_DIR=/runpod-volume/Wan2.2-TI2V-5B \
    WAN_OUTPUT_DIR=/tmp/wan-output \
    WAN_MAX_IMAGE_BYTES=9437184 \
    WAN_MAX_OUTPUT_BYTES=52428800

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git clone "${WAN_REPOSITORY}" "${WAN_SOURCE_DIR}" \
    && git -C "${WAN_SOURCE_DIR}" checkout --detach "${WAN_REVISION}" \
    && test "$(git -C "${WAN_SOURCE_DIR}" rev-parse HEAD)" = "${WAN_REVISION}" \
    && sed '/^torch/d;/^torchvision/d;/^torchaudio/d;/^flash_attn/d' "${WAN_SOURCE_DIR}/requirements.txt" > /tmp/wan-requirements.txt \
    && python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/wan-requirements.txt runpod==1.7.13 \
    && rm -rf "${WAN_SOURCE_DIR}/.git" /tmp/wan-requirements.txt

WORKDIR /app
COPY handler.py /app/handler.py
COPY download_model.py /app/download_model.py

RUN python -m py_compile /app/handler.py /app/download_model.py \
    && python /app/handler.py --self-test

LABEL org.opencontainers.image.title="La Estratosferica Wan official worker" \
      org.opencontainers.image.description="RunPod worker based only on official Wan2.2 TI2V-5B sources" \
      org.opencontainers.image.source="https://github.com/laestratosferica-tv/robot-estratosferica" \
      org.opencontainers.image.vendor="La Estratosferica" \
      ai.wan.repository="https://github.com/Wan-Video/Wan2.2" \
      ai.wan.revision="42bf4cfaa384bc21833865abc2f9e6c0e67233dc" \
      ai.wan.model="Wan-AI/Wan2.2-TI2V-5B"

CMD ["python", "-u", "/app/handler.py"]
