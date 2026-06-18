import gc
import json
import os
import queue
import tempfile
import threading
import time
import uuid
from datetime import datetime

import pytz
import torch
import whisperx
from audio_cleaning import (
    clean_audio_file,
    parse_audio_cleaning_config,
    parse_audio_cleaning_request_config,
)
from auphonic_compat import create_auphonic_router
from diarization import (
    apply_diarization,
    normalize_speaker_segments,
    normalize_speakers,
)
from fastapi import (
    APIRouter,
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
from gpu_monitor import gpu_has_capacity
from torch_compat import allow_trusted_pyannote_checkpoints

app = FastAPI()

jobs = {}
jobs_lock = threading.Lock()
device = os.getenv("WHISPERX_DEVICE", "cuda")  # o "cpu"

TEMP_AUDIO_DIR = os.getenv("WHISPERX_TEMP_AUDIO_DIR", "/tmp/whisper")
job_queue = queue.Queue()
active_gpu_jobs = 0
active_gpu_jobs_lock = threading.Lock()
dispatcher_started = False
dispatcher_started_lock = threading.Lock()
MAX_ACTIVE_GPU_JOBS = int(os.getenv("WHISPERX_MAX_ACTIVE_GPU_JOBS", "1"))
GPU_POLL_SECONDS = float(os.getenv("WHISPERX_GPU_POLL_SECONDS", "2"))

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
            detail="Missing or invalid API Key",
        )
    return key


router = APIRouter(dependencies=[Depends(validate_api_key)])
app.include_router(router)
# 1. Carga del modelo WhisperX
allow_trusted_pyannote_checkpoints()
model = whisperx.load_model(
    WHISPERX_MODEL, device=device, compute_type=WHISPERX_COMPUTE_TYPE
)


@app.on_event("startup")
def start_queue_dispatcher():
    global dispatcher_started

    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    with dispatcher_started_lock:
        if dispatcher_started:
            return

        threading.Thread(target=queue_dispatcher, daemon=True).start()
        dispatcher_started = True


def reserve_gpu_slot():
    global active_gpu_jobs

    with active_gpu_jobs_lock:
        if active_gpu_jobs >= MAX_ACTIVE_GPU_JOBS:
            return False

        if not gpu_has_capacity():
            return False

        active_gpu_jobs += 1
        return True


def release_gpu_slot():
    global active_gpu_jobs

    with active_gpu_jobs_lock:
        active_gpu_jobs = max(0, active_gpu_jobs - 1)


def queue_dispatcher():
    while True:
        job_id, audio_path = job_queue.get()
        slot_reserved = False

        try:
            while not reserve_gpu_slot():
                time.sleep(GPU_POLL_SECONDS)

            slot_reserved = True
            threading.Thread(
                target=run_queued_job,
                args=(job_id, audio_path),
                daemon=True,
            ).start()
        except Exception as exc:
            if slot_reserved:
                release_gpu_slot()

            mark_job_failed(job_id, exc)
            cleanup_audio_file(audio_path)
            job_queue.task_done()


def run_queued_job(job_id, audio_path):
    try:
        if get_job_type(job_id) == "audio_cleaning":
            process_audio_cleaning_file(job_id, audio_path)
        else:
            process_audio_file(job_id, audio_path)
    finally:
        release_gpu_slot()
        job_queue.task_done()


def get_job_type(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return None
        return job.get("type")


def mark_job_failed(job_id, exc):
    with jobs_lock:
        job = jobs.get(job_id)
        if job:
            job["status"] = "failed"
            job["error"] = str(exc)


def cleanup_audio_file(audio_path):
    try:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
    except OSError as exc:
        print(f"No se pudo eliminar el audio temporal {audio_path}: {exc}")


def get_upload_suffix(filename):
    _, suffix = os.path.splitext(filename or "")
    if not suffix or len(suffix) > 10:
        return ".wav"
    return suffix


def save_audio_to_temp(audio: bytes, filename: str):
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    suffix = get_upload_suffix(filename)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        dir=TEMP_AUDIO_DIR,
    ) as tmp:
        tmp.write(audio)
        return tmp.name


app.include_router(
    create_auphonic_router(
        api_key=API_KEY,
        jobs=jobs,
        jobs_lock=jobs_lock,
        job_queue=job_queue,
        save_audio_to_temp=save_audio_to_temp,
        cleanup_audio_file=cleanup_audio_file,
        temp_audio_dir=TEMP_AUDIO_DIR,
    )
)


def process_audio_file(job_id: str, audio_path: str):
    start = time.perf_counter()

    with jobs_lock:
        job = jobs[job_id]
        job["status"] = "running"
        created_at = job["created_at"]
        data_name = job["data_name"]
        config = job["config"]

    try:
        result = whisper_transcribe(
            audio_path,
            job_id,
            created_at,
            data_name,
            config,
        )
        end = time.perf_counter()
        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["duration"] = int(round(end - start))
            jobs[job_id]["result"] = result
    except Exception as exc:
        mark_job_failed(job_id, exc)
        raise
    finally:
        cleanup_audio_file(audio_path)


def process_audio_cleaning_file(job_id: str, audio_path: str):
    start = time.perf_counter()

    with jobs_lock:
        job = jobs[job_id]
        job["status"] = "running"
        config = job["config"]
        output_path = job.get("output_path")

    try:
        result = clean_audio_file(audio_path, output_path=output_path, config=config)
        end = time.perf_counter()
        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["duration"] = int(round(end - start))
            jobs[job_id]["result"] = result.to_dict()
    except Exception as exc:
        mark_job_failed(job_id, exc)
        raise
    finally:
        cleanup_audio_file(audio_path)


@app.post("/v2/jobs")
async def create_job(
    data_file: UploadFile = File(...),
    config: str = Form(...),
):
    tmp_path = None
    try:
        config_dict = json.loads(config)
        audio = await data_file.read()
        job_id = str(uuid.uuid4().int >> (128 - 64))
        utc = pytz.UTC
        created = datetime.now(utc).isoformat()
        config_dict.setdefault("transcription_config", {})["operating_point"] = (
            "enhanced"
        )
        tmp_path = save_audio_to_temp(audio, data_file.filename)

        with jobs_lock:
            jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "created_at": created,
                "data_name": data_file.filename,
                "duration": 0,
                "config": config_dict,
                "type": "transcription",
            }

        job_queue.put((job_id, tmp_path))
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON in config: {e}"}
    except Exception:
        cleanup_audio_file(tmp_path)
        raise

    return {"id": job_id}


@app.post("/v2/audio-cleaning/jobs")
async def create_audio_cleaning_job(
    data_file: UploadFile = File(...),
    config: str = Form(...),
):
    tmp_path = None
    try:
        config_dict = parse_audio_cleaning_request_config(config)
        cleaning_config = parse_audio_cleaning_config(config_dict)
        audio = await data_file.read()
        job_id = str(uuid.uuid4().int >> (128 - 64))
        utc = pytz.UTC
        created = datetime.now(utc).isoformat()
        tmp_path = save_audio_to_temp(audio, data_file.filename)

        with jobs_lock:
            jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "created_at": created,
                "data_name": data_file.filename,
                "duration": 0,
                "config": {
                    "audio_cleaning_config": cleaning_config.to_dict(),
                    "request": config_dict,
                },
                "type": "audio_cleaning",
            }

        job_queue.put((job_id, tmp_path))
    except ValueError as exc:
        cleanup_audio_file(tmp_path)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        cleanup_audio_file(tmp_path)
        raise

    return {"id": job_id}


@app.get("/v2/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job:
            response = {
                "job": {
                    "id": job["id"],
                    "status": job["status"],
                    "created_at": job["created_at"],
                    "data_name": job["data_name"],
                    "duration": job["duration"],
                    "config": job["config"],
                    "type": job["type"],
                }
            }
            if "error" in job:
                response["job"]["error"] = job["error"]

    if not job:
        return {"error": "Job not found"}, 404
    return response


@app.get("/v2/jobs/{job_id}/transcript")
def get_job_transcript(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job:
            status_value = job.get("status")
            error = job.get("error", "Job failed")
            result = job.get("result")

    if not job:
        return {"error": "Job not found"}, 404
    if status_value == "failed":
        raise HTTPException(status_code=500, detail=error)
    if result is None:
        raise HTTPException(status_code=202, detail="Job is still running")
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
    return (
        parse_optional_int(transcription_config.get("batch_size"))
        or WHISPERX_BATCH_SIZE
    )


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
        language_code=result["language"], device=device
    )

    # 5. Alinear palabra a palabra
    result_aligned = whisperx.align(
        result["segments"], align_model, metadata, audio_file, device
    )
    del align_model
    clear_cuda_cache()

    result_aligned, diarization_config, speaker_segments = apply_diarization(
        audio_file, result_aligned, config_dict, device
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
                    "speaker": speaker,
                }
            ],
            "start_time": round(float(word["start"]), 2),
            "end_time": round(float(word["end"]), 2),
            "type": "word",
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
            "config": config_dict,
        },
        "metadata": {
            "created_at": created_at,
            "language_pack_info": {
                "adapted": False,
                "itn": True,
                "language_description": result["language"],
                "word_delimiter": " ",
                "writing_direction": "left-to-right",
            },
            "transcription_config": {
                "operating_point": config_dict.get("transcription_config", {}).get(
                    "operating_point", "enhanced"
                ),
                "language": result["language"],
                "diarization": "speaker" if diarization_config["enabled"] else "none",
            },
            "diarization": {
                "enabled": diarization_config["enabled"],
                "model": diarization_config["model"]
                if diarization_config["enabled"]
                else None,
                "speakers": speakers,
                "segments": speaker_segments,
            },
            "type": "transcription",
        },
        "results": word_results,
    }
