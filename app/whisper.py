from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Form, File, Depends, APIRouter, Security, status
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
import threading

app = FastAPI()

jobs = {}
device = os.getenv("WHISPERX_DEVICE", "cuda")  # o "cpu"
diarization_model = None
diarization_model_name = None
diarization_model_lock = threading.Lock()

WHISPERX_MODEL = os.getenv("WHISPERX_MODEL", "medium")
WHISPERX_COMPUTE_TYPE = os.getenv("WHISPERX_COMPUTE_TYPE", "int8_float16")
WHISPERX_BATCH_SIZE = int(os.getenv("WHISPERX_BATCH_SIZE", "2"))
DIARIZATION_MODEL = os.getenv("WHISPERX_DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
DIARIZATION_ENABLED_BY_DEFAULT = os.getenv("WHISPERX_DIARIZE", "").lower() in {"1", "true", "yes", "on"}
HF_TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "PYANNOTE_AUTH_TOKEN",
)

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


def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "speaker", "speakers", "diarization"}:
            return True
        if normalized in {"0", "false", "no", "off", "none", "disabled", ""}:
            return False
    return default


def parse_optional_int(value):
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def get_hf_token():
    for env_name in HF_TOKEN_ENV_NAMES:
        token = os.getenv(env_name)
        if token:
            return token
    return None


def get_diarization_config(config_dict):
    transcription_config = config_dict.get("transcription_config", {})
    speaker_config = (
        config_dict.get("speaker_diarization_config")
        or config_dict.get("diarization_config")
        or transcription_config.get("speaker_diarization_config")
        or transcription_config.get("diarization_config")
        or {}
    )

    enabled_value = first_present(
        speaker_config.get("enabled"),
        transcription_config.get("diarization"),
        config_dict.get("diarization"),
        config_dict.get("diarize"),
    )
    enabled = parse_bool(enabled_value, DIARIZATION_ENABLED_BY_DEFAULT)

    num_speakers = parse_optional_int(first_present(
        speaker_config.get("num_speakers"),
        speaker_config.get("speakers"),
        speaker_config.get("speaker_count"),
    ))
    min_speakers = parse_optional_int(first_present(
        speaker_config.get("min_speakers"),
        transcription_config.get("min_speakers"),
    ))
    max_speakers = parse_optional_int(first_present(
        speaker_config.get("max_speakers"),
        transcription_config.get("max_speakers"),
    ))

    if num_speakers is not None:
        min_speakers = None
        max_speakers = None

    return {
        "enabled": enabled,
        "model": first_present(speaker_config.get("model"), config_dict.get("diarization_model"), DIARIZATION_MODEL),
        "num_speakers": num_speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "fill_nearest": parse_bool(speaker_config.get("fill_nearest"), True),
    }


def get_transcription_batch_size(config_dict):
    transcription_config = config_dict.get("transcription_config", {})
    return parse_optional_int(transcription_config.get("batch_size")) or WHISPERX_BATCH_SIZE


def clear_cuda_cache():
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_diarization_model(model_name):
    global diarization_model, diarization_model_name

    if diarization_model is not None and diarization_model_name == model_name:
        return diarization_model

    with diarization_model_lock:
        if diarization_model is not None and diarization_model_name == model_name:
            return diarization_model

        from whisperx.diarize import DiarizationPipeline

        hf_token = get_hf_token()
        try:
            diarization_model = DiarizationPipeline(
                model_name=model_name,
                token=hf_token,
                device=device
            )
        except Exception as exc:
            raise RuntimeError(
                "No se pudo cargar el modelo de diarización. "
                "Comprueba que el contenedor tenga acceso al modelo de Hugging Face "
                "y que hayas definido HF_TOKEN, HUGGINGFACE_TOKEN, "
                "HUGGING_FACE_HUB_TOKEN o PYANNOTE_AUTH_TOKEN si el modelo lo requiere."
            ) from exc
        diarization_model_name = model_name
        return diarization_model


def apply_diarization(audio_file, aligned_result, config_dict):
    diarization_config = get_diarization_config(config_dict)
    if not diarization_config["enabled"]:
        return aligned_result, diarization_config, []

    print("Realizando diarización con:", diarization_config["model"])
    diarize_model = get_diarization_model(diarization_config["model"])
    diarize_segments = diarize_model(
        audio_file,
        num_speakers=diarization_config["num_speakers"],
        min_speakers=diarization_config["min_speakers"],
        max_speakers=diarization_config["max_speakers"],
    )
    diarized_result = whisperx.assign_word_speakers(
        diarize_segments,
        aligned_result,
        fill_nearest=diarization_config["fill_nearest"],
    )
    speaker_segments = diarize_segments[["start", "end", "speaker"]].to_dict("records")
    return diarized_result, diarization_config, speaker_segments


def normalize_speakers(word_segments, speaker_segments):
    speaker_map = {}
    for segment in speaker_segments:
        raw_speaker = segment.get("speaker")
        if raw_speaker and raw_speaker not in speaker_map:
            speaker_map[raw_speaker] = f"S{len(speaker_map) + 1}"
    for word in word_segments:
        raw_speaker = word.get("speaker")
        if raw_speaker and raw_speaker not in speaker_map:
            speaker_map[raw_speaker] = f"S{len(speaker_map) + 1}"
    return speaker_map


def normalize_speaker_segments(speaker_segments, speaker_map):
    normalized_segments = []
    for segment in speaker_segments:
        normalized_segments.append({
            "start": round(float(segment["start"]), 2),
            "end": round(float(segment["end"]), 2),
            "speaker": speaker_map.get(segment.get("speaker"), "S1"),
        })
    return normalized_segments


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
        config_dict
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
    
