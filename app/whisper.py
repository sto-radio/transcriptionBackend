import gc
import json
import os
import tempfile
import time
import uuid
from datetime import datetime

import pytz
import torch
import whisperx
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Security,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from diarization import (
    apply_diarization,
    normalize_speaker_segments,
    normalize_speakers,
)

app = FastAPI()

jobs = {}
device = os.getenv("WHISPERX_DEVICE", "cuda")  # o "cpu"

WHISPERX_MODEL = os.getenv("WHISPERX_MODEL", "medium")
WHISPERX_COMPUTE_TYPE = os.getenv("WHISPERX_COMPUTE_TYPE", "int8_float16")
WHISPERX_BATCH_SIZE = int(os.getenv("WHISPERX_BATCH_SIZE", "2"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY", "pruebakey")  # podría venir de env vars o DB
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
    WHISPERX_MODEL,
    device=device,
    compute_type=WHISPERX_COMPUTE_TYPE
)


def process_audio(job_id: str, audio: bytes, config: dict):
    #buffer = io.BytesIO(audio_bytes)
    #buffer.name = "audio.wav"
    start = time.perf_counter()
    created_at = jobs[job_id]["created_at"]

    # Guardar bytes en archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="/tmp/whisper") as tmp:
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
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)
        raise
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
        config_dict.setdefault("transcription_config", {})["operating_point"] = "enhanced"
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
    if "error" in job:
        response["job"]["error"] = job["error"]
    return response

@app.get("/v2/jobs/{job_id}/transcript")
def get_job_transcript(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}, 404
    if job.get("status") == "failed":
        raise HTTPException(status_code=500, detail=job.get("error", "Job failed"))
    if "result" not in job:
        raise HTTPException(status_code=202, detail="Job is still running")
    result = jobs[job_id]["result"]
    return result


def parse_optional_int(value):
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_transcription_batch_size(config_dict):
    transcription_config = config_dict.get("transcription_config", {})
    return parse_optional_int(transcription_config.get("batch_size")) or WHISPERX_BATCH_SIZE


def clear_cuda_cache():
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def whisper_transcribe(audio_file, job_id, created_at, data_name, config_dict):

    # 2. Carga del audio
    audio = whisperx.load_audio(audio_file)
    batch_size = get_transcription_batch_size(config_dict)

    # 3. Transcripción y obtención de segmentos
    result = model.transcribe(audio, batch_size=batch_size)
    print("Segmentos iniciales:", result["segments"])
    clear_cuda_cache()

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
    del align_model
    clear_cuda_cache()

    result_aligned, diarization_config, speaker_segments = apply_diarization(
        audio_file,
        result_aligned,
        config_dict,
        device
    )
    clear_cuda_cache()

    # Añadir las palabras correspondientes
    word_results = []
    speaker_map = normalize_speakers(result_aligned["word_segments"], speaker_segments)
    speaker_segments = normalize_speaker_segments(speaker_segments, speaker_map)
    speakers = sorted(set(speaker_map.values())) or ["S1"]
    for word in result_aligned["word_segments"]:
        if "start" not in word or "end" not in word:
            continue
        speaker = speaker_map.get(word.get("speaker"), "S1")
        word_entry = {
        "alternatives": [
            {
                "confidence": word.get("score", 1.0),
                "content": word["word"],
                "language": result["language"],
                "speaker": speaker
            }
        ],
        "start_time": round(float(word["start"]), 2),
        "end_time": round(float(word["end"]), 2),
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
                "language": result["language"],
                "diarization": "speaker" if diarization_config["enabled"] else "none"
            },
            "diarization": {
                "enabled": diarization_config["enabled"],
                "model": diarization_config["model"] if diarization_config["enabled"] else None,
                "speakers": speakers,
                "segments": speaker_segments
            },
            "type": "transcription"
        },
        "results": word_results
    }
    
