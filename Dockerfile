FROM debian:bookworm-slim
ARG TARGETARCH
ARG TARGETVARIANT

# Install faster-whisper
WORKDIR /usr/src

COPY ./pyproject.toml ./
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
    \
    && python3 -m venv .venv \
    && .venv/bin/pip3 install --no-cache-dir -U \
        setuptools \
        wheel \
    && .venv/bin/pip3 install --no-cache-dir \
        --extra-index-url 'https://download.pytorch.org/whl/cpu' \
        'torch==2.6.0' \
    \
    && .venv/bin/pip3 install --no-cache-dir \
        --extra-index-url https://www.piwheels.org/simple \
        -e '.[zeroconf,transformers,sherpa,onnx-asr,qwen3-asr,hass]' \
    \
    && rm -rf /var/lib/apt/lists/*

COPY ./ ./

EXPOSE 10300

# The server only starts listening once the model is loaded, which means
# downloading it on first run -- hence the long start period, during which
# failures don't count against --retries. --retries covers the other direction:
# transcription runs on the event loop, so a check can time out behind a long
# request without the server being unhealthy.
HEALTHCHECK --interval=30s --timeout=20s --start-period=5m --retries=3 \
    CMD ["/usr/src/.venv/bin/python3", "-m", "wyoming_faster_whisper.health_check"]

ENTRYPOINT ["bash", "docker_run.sh"]
