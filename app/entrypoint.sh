#!/bin/bash
set -e

# Establecer PYTHONPATH para usar las dependencias instaladas
#export PYTHONPATH=/app/packages:$PYTHONPATH

# Instalar dependencias en el directorio montado desde PVC
#pip install --target /app/packages --upgrade-strategy only-if-needed -r requirements.txt

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

# Ejecutar la aplicación
exec python3 -m uvicorn whisper:app --host 0.0.0.0 --port 8000
