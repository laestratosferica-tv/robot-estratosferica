FROM registry.runpod.net/wlsdml1114-generate-video-ksampler-dockerfile:a9247705c

COPY harden_worker.py /opt/estratosferica/harden_worker.py
RUN python /opt/estratosferica/harden_worker.py /handler.py \
    && python -m py_compile /handler.py

LABEL org.opencontainers.image.title="La Estratosferica Wan private worker"
LABEL org.opencontainers.image.description="Hardened derivative worker for private character references"

