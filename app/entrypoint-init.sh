#!/bin/bash
set -e

# Establecer PYTHONPATH para usar las dependencias instaladas
export PYTHONPATH=/app/packages:$PYTHONPATH

pip install --target /app/packages --upgrade-strategy only-if-needed -r requirements.txt

python /app/preload_models.py
# Ejecutar la aplicación
#exec python -m uvicorn whisper:app --host 0.0.0.0 --port 8000
