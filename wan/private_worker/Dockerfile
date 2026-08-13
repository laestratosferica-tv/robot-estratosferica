FROM wlsdml1114/engui_genai-base_blackwell:1.1 AS runtime

ARG COMFYUI_COMMIT=ddbaa8752874c275290d054ee4fddd6e004f5fdf
ARG COMFYUI_MANAGER_COMMIT=2f3fc41a13bdacff82055b402b445f64c1a27c25
ARG COMFYUI_GGUF_COMMIT=6ea2651e7df66d7585f6ffee804b20e92fb38b8a
ARG KJNODES_COMMIT=6ab7e8130e449ed2c0037589bcf84146ceb7fc9c
ARG VIDEO_HELPER_COMMIT=4ee72c065db22c9d96c2427954dc69e7b908444b
ARG FANTASY_TALKING_COMMIT=48dd427c5cf28cc849467dc97e54515286df0aa0
ARG WAN_BLOCKSWAP_COMMIT=5fa2ec0fa55879fe43a33e762fff91fc2c553a67
ARG WAN_WRAPPER_COMMIT=088128b224242e110d3906c6750e9a3a348a659b
ARG INTELLIGENT_VRAM_COMMIT=3a3fdb41c1b0e01545d9d394304adc846cdde52b
ARG AUTO_WAN_COMMIT=d4f7e6294fc8d1f38c8b3acdb520c64d983099a1
ARG ADAPTIVE_WINDOW_COMMIT=6c46e055f63b031324a0d19f6e2adebcbe76b90b
ARG UPSTREAM_WORKER_COMMIT=4b6d5ec27dae6409bd2011a96d8e819e67d4ebaa

ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir -U "huggingface_hub[hf_transfer]" \
    && pip install --no-cache-dir runpod websocket-client

WORKDIR /

# Every source dependency is pinned. A rebuild cannot silently change the
# runtime because a third-party default branch moved.
RUN git clone --filter=blob:none https://github.com/comfyanonymous/ComfyUI.git /ComfyUI \
    && git -C /ComfyUI checkout "$COMFYUI_COMMIT" \
    && pip install --no-cache-dir -r /ComfyUI/requirements.txt

RUN set -eux; \
    clone_pinned() { repo="$1"; dest="$2"; commit="$3"; \
      git clone --filter=blob:none "$repo" "$dest"; \
      git -C "$dest" checkout "$commit"; \
    }; \
    clone_pinned https://github.com/Comfy-Org/ComfyUI-Manager.git /ComfyUI/custom_nodes/ComfyUI-Manager "$COMFYUI_MANAGER_COMMIT"; \
    pip install --no-cache-dir -r /ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt; \
    clone_pinned https://github.com/city96/ComfyUI-GGUF.git /ComfyUI/custom_nodes/ComfyUI-GGUF "$COMFYUI_GGUF_COMMIT"; \
    pip install --no-cache-dir -r /ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt; \
    clone_pinned https://github.com/kijai/ComfyUI-KJNodes.git /ComfyUI/custom_nodes/ComfyUI-KJNodes "$KJNODES_COMMIT"; \
    pip install --no-cache-dir -r /ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt; \
    clone_pinned https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite "$VIDEO_HELPER_COMMIT"; \
    pip install --no-cache-dir -r /ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt; \
    clone_pinned https://github.com/kael558/ComfyUI-GGUF-FantasyTalking.git /ComfyUI/custom_nodes/ComfyUI-GGUF-FantasyTalking "$FANTASY_TALKING_COMMIT"; \
    pip install --no-cache-dir -r /ComfyUI/custom_nodes/ComfyUI-GGUF-FantasyTalking/requirements.txt; \
    clone_pinned https://github.com/orssorbit/ComfyUI-wanBlockswap.git /ComfyUI/custom_nodes/ComfyUI-wanBlockswap "$WAN_BLOCKSWAP_COMMIT"; \
    clone_pinned https://github.com/kijai/ComfyUI-WanVideoWrapper.git /ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper "$WAN_WRAPPER_COMMIT"; \
    pip install --no-cache-dir -r /ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt; \
    clone_pinned https://github.com/eddyhhlure1Eddy/IntelligentVRAMNode.git /ComfyUI/custom_nodes/IntelligentVRAMNode "$INTELLIGENT_VRAM_COMMIT"; \
    clone_pinned https://github.com/eddyhhlure1Eddy/auto_wan2.2animate_freamtowindow_server.git /ComfyUI/custom_nodes/auto_wan2.2animate_freamtowindow_server "$AUTO_WAN_COMMIT"; \
    clone_pinned https://github.com/eddyhhlure1Eddy/ComfyUI-AdaptiveWindowSize.git /tmp/ComfyUI-AdaptiveWindowSize "$ADAPTIVE_WINDOW_COMMIT"; \
    mkdir -p /ComfyUI/custom_nodes/ComfyUI-AdaptiveWindowSize; \
    cp -a /tmp/ComfyUI-AdaptiveWindowSize/ComfyUI-AdaptiveWindowSize/. /ComfyUI/custom_nodes/ComfyUI-AdaptiveWindowSize/; \
    rm -rf /tmp/ComfyUI-AdaptiveWindowSize

RUN mkdir -p /ComfyUI/models/diffusion_models /ComfyUI/models/loras \
      /ComfyUI/models/clip_vision /ComfyUI/models/text_encoders /ComfyUI/models/vae \
    && hf download Kijai/WanVideo_comfy_fp8_scaled I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors --local-dir /tmp/models \
    && mv /tmp/models/I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors /ComfyUI/models/diffusion_models/ \
    && hf download Kijai/WanVideo_comfy_fp8_scaled I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors --local-dir /tmp/models \
    && mv /tmp/models/I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors /ComfyUI/models/diffusion_models/ \
    && hf download lightx2v/Wan2.2-Lightning Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors --local-dir /tmp/models \
    && mv /tmp/models/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors /ComfyUI/models/loras/ \
    && hf download lightx2v/Wan2.2-Lightning Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors --local-dir /tmp/models \
    && mv /tmp/models/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors /ComfyUI/models/loras/ \
    && hf download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/clip_vision/clip_vision_h.safetensors --local-dir /tmp/models \
    && mv /tmp/models/split_files/clip_vision/clip_vision_h.safetensors /ComfyUI/models/clip_vision/ \
    && hf download Kijai/WanVideo_comfy umt5-xxl-enc-bf16.safetensors --local-dir /ComfyUI/models/text_encoders \
    && hf download Kijai/WanVideo_comfy Wan2_1_VAE_bf16.safetensors --local-dir /ComfyUI/models/vae \
    && rm -rf /tmp/models /root/.cache/huggingface

RUN git clone --filter=blob:none https://github.com/wlsdml1114/generate_video.git /opt/upstream-worker \
    && git -C /opt/upstream-worker checkout "$UPSTREAM_WORKER_COMMIT" \
    && cp /opt/upstream-worker/handler.py /handler.py \
    && cp /opt/upstream-worker/entrypoint.sh /entrypoint.sh \
    && cp /opt/upstream-worker/extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml \
    && cp /opt/upstream-worker/new_Wan22_api.json /new_Wan22_api.json \
    && cp /opt/upstream-worker/new_Wan22_flf2v_api.json /new_Wan22_flf2v_api.json \
    && rm -rf /opt/upstream-worker

COPY harden_worker.py /opt/estratosferica/harden_worker.py
RUN python /opt/estratosferica/harden_worker.py /handler.py \
    && python -m py_compile /handler.py \
    && chmod +x /entrypoint.sh

LABEL org.opencontainers.image.title="La Estratosferica Wan private worker"
LABEL org.opencontainers.image.description="Reproducible hardened Wan 2.2 worker built from pinned public sources"
LABEL org.opencontainers.image.source="https://github.com/laestratosferica-tv/robot-estratosferica"
LABEL org.opencontainers.image.upstream.revision="$UPSTREAM_WORKER_COMMIT"

CMD ["/entrypoint.sh"]
