FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/home/appuser/.cache \
    NLTK_DATA=/home/appuser/nltk_data \
    XDG_CONFIG_HOME=/home/appuser/.config \
    HOME=/home/appuser

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser && mkdir -p /app /home/appuser/.cache /home/appuser/.config /home/appuser/nltk_data \
    && chown -R appuser:appuser /app /home/appuser

WORKDIR /app

COPY --chown=appuser:appuser app/requirements.txt .
RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt

COPY --chown=appuser:appuser app/ .
RUN chmod +x /app/entrypoint.sh /app/entrypoint-init.sh

USER appuser

CMD ["bash", "/app/entrypoint.sh"]
