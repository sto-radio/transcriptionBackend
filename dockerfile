FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/home/appuser/.cache \
    NLTK_DATA=/home/appuser/nltk_data \
    XDG_CONFIG_HOME=/home/appuser/.config \
    MPLCONFIGDIR=/home/appuser/.config/matplotlib \
    HOME=/home/appuser \
    TMPDIR=/var/tmp/pip-tmp \
    PIP_CACHE_DIR=/var/cache/pip \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser \
    && mkdir -p /app /home/appuser/.cache /home/appuser/.config /home/appuser/nltk_data /var/tmp/pip-tmp /var/cache/pip \
    && chmod 1777 /var/tmp/pip-tmp \
    && chown -R appuser:appuser /app /home/appuser

WORKDIR /app

COPY --chown=appuser:appuser app/requirements.txt .
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir --prefer-binary \
        torch==2.8.0 \
        torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python3 -m pip install --no-cache-dir --prefer-binary -r requirements.txt \
    && rm -rf /var/tmp/pip-tmp/* /var/cache/pip/*

COPY --chown=appuser:appuser app/ .
RUN chmod +x /app/entrypoint.sh /app/entrypoint-init.sh

USER appuser

CMD ["bash", "/app/entrypoint.sh"]
