FROM python:3.9-slim

# Instala dependencias del sistema (FFmpeg para Whisper)
RUN apt-get update && apt-get install -y ffmpeg prelink && rm -rf /var/lib/apt/lists/*

# Crea usuario y directorios seguros
RUN useradd -m appuser && \
    mkdir -p /home/appuser/.cache /home/appuser/.config && \
    chown -R appuser:appuser /home/appuser

# Variables de entorno para cachés
ENV XDG_CACHE_HOME=/home/appuser/.cache \
    XDG_CONFIG_HOME=/home/appuser/.config \
    HOME=/home/appuser \
    PIP_CACHE_DIR=/app/cache

WORKDIR /app

# Crea directorios para caché y paquetes
RUN mkdir -p /app/cache /app/packages && chown -R appuser:appuser /app/cache /app/packages

# 4. Copia requirements ANTES del código (para mejor caching de capas)
COPY --chown=appuser:appuser app/requirements.txt .

COPY --chown=appuser:appuser app/ .
USER appuser

CMD ["/app/entrypoint.sh"]
#CMD ["tail", "-f", "/dev/null"]