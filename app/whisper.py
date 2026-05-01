from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Form, File, Depends, APIRouter, Security
from fastapi.security import APIKeyHeader
from datetime import datetime, timezone
from fastapi.middleware.cors import CORSMiddleware
import os
import whisperx
import torch
import json
import uuid
import io
import time
import pytz
import tempfile
import gc

app = FastAPI()

jobs = {}
device = "cuda"  # o "cpu"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "pruebakey"  # podría venir de env vars o DB
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def validate_api_key(key: str = Security(api_key_header)):
    if not key or key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API Key"
        )
    return key

router = APIRouter(dependencies=[Depends(validate_api_key)])
app.include_router(router)
# 1. Carga del modelo WhisperX
model = whisperx.load_model(
    "medium", 
    device=device, 
    compute_type="float16"
)


def process_audio(job_id: str, audio: bytes, config: dict):
    #buffer = io.BytesIO(audio_bytes)
    #buffer.name = "audio.wav"
    start = time.perf_counter()
    created_at = jobs[job_id]["created_at"]

    # Guardar bytes en archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio)
        tmp_path = tmp.name
    try:
        result = whisper_transcribe(tmp_path, job_id, created_at,
            jobs[job_id]["data_name"],
            jobs[job_id]["config"]
        )
        end = time.perf_counter()
        jobs[job_id]["status"] = "done"
        jobs[job_id]["duration"] = int(round(end - start))
        jobs[job_id]["result"] = result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/v2/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    data_file: UploadFile = File(...),
    config: str = Form(...)
):
    try:
        config_dict = json.loads(config)
        audio = await data_file.read()
        job_id = str(uuid.uuid4().int >> (128 - 64))
        utc = pytz.UTC
        created = datetime.now(utc).isoformat()
        config_dict["transcription_config"]["operating_point"] = "enhanced"
        jobs[job_id] = {
            "id": job_id,
            "status": "Running",
            "created_at": created,
            "data_name": data_file.filename,
            "duration": 0,
            "config": config_dict,
            "type": "transcription"
        }
        background_tasks.add_task(process_audio, job_id, audio, config_dict)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON in config: {e}"}
    
    return {"id": job_id}
    
@app.get("/v2/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}, 404
    response = {
        "job": {
            "id": job["id"],
            "status": job["status"],
            "created_at": job["created_at"],
            "data_name": job["data_name"],
            "duration": job["duration"],
            "config": job["config"],
            "type": job["type"]
        }
    }
    return response

@app.get("/v2/jobs/{job_id}/transcript")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}, 404
    result = jobs[job_id]["result"]
    return result


def whisper_transcribe(audio_file, job_id, created_at, data_name, config_dict):

    # 2. Carga del audio
    audio = whisperx.load_audio(audio_file)

    # 3. Transcripción y obtención de segmentos
    result = model.transcribe(audio, batch_size=16)
    print("Segmentos iniciales:", result["segments"])

    # 4. Carga del modelo de alineación
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device
    )

    # 5. Alinear palabra a palabra
    result_aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio_file,
        device
    )

    # Añadir las palabras correspondientes
    word_results = []
    for word in result_aligned["word_segments"]:
        word_entry = {
        "alternatives": [
            {
                "confidence": word.get("score", 1.0),
                "content": word["word"],
                "language": result["language"],
                "speaker": "S1"  # WhisperX no incluye speaker, pero Speechmatics sí. Puedes ajustarlo más tarde si usas diarización.
            }
        ],
        "start_time": round(word["start"], 2),
        "end_time": round(word["end"], 2),
        "type": "word"
        }
        word_results.append(word_entry)
    return {
        "format": "2.1",
        "job": {
            "id": job_id,
            "created_at": created_at,
            "data_name": data_name,
            "duration": 0,  # se añade luego
            "status": "Done",
            "type": "transcription",
            "config": config_dict
        },
        "metadata": {
            "created_at": created_at,
            "language_pack_info": {
                "adapted": False,
                "itn": True,
                "language_description": result["language"],
                "word_delimiter": " ",
                "writing_direction": "left-to-right"
            },
            "transcription_config": {
                "operating_point": config_dict.get("transcription_config", {}).get("operating_point", "enhanced"),
                "language": result["language"]
            },
            "type": "transcription"
        },
        "results": word_results
    }
    
