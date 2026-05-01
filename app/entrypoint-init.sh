#!/bin/bash

# Establecer PYTHONPATH para usar las dependencias instaladas
export PYTHONPATH=/app/packages:$PYTHONPATH

pip install --target /app/packages --upgrade-strategy only-if-needed -r requirements.txt

python3 -c "
import whisperx
whisperx.load_model('medium', device='cuda', compute_type='float16')
"
# Ejecutar la aplicación
#exec python -m uvicorn whisper:app --host 0.0.0.0 --port 8000