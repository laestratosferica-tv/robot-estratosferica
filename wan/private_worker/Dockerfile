FROM registry.runpod.net/averystormknight-hue-wan22-ti2v-runpod-main-dockerfile:81fc2044f

COPY harden_worker.py /opt/estratosferica/harden_worker.py
RUN python /opt/estratosferica/harden_worker.py /app/handler.py \
    && test -s /app/workflows/wan22_t2v_api.json \
    && python -m json.tool /app/workflows/wan22_t2v_api.json >/dev/null \
    && python -m py_compile /app/handler.py

LABEL org.opencontainers.image.title="La Estratosferica Wan private worker"
LABEL org.opencontainers.image.description="Hardened derivative of Wan2.2 TI2V Hub v0.1.4"
LABEL org.opencontainers.image.source="https://github.com/laestratosferica-tv/robot-estratosferica"
LABEL org.opencontainers.image.upstream.revision="81fc2044f"
