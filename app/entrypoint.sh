#!/bin/bash

# Establecer PYTHONPATH para usar las dependencias instaladas
export PYTHONPATH=/app/packages:$PYTHONPATH
# Instalar dependencias en el directorio montado desde PVC
pip install --target /app/packages --upgrade-strategy only-if-needed -r requirements.txt

# Ejecutar la aplicación
exec python -m uvicorn whisper:app --host 0.0.0.0 --port 8000
