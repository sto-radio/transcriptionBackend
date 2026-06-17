import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from audio_cleaning import (
    parse_audio_cleaning_config,
    parse_audio_cleaning_request_config,
)
from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile as StarletteUploadFile


FILE_FIELD_NAMES = {"input_file", "data_file", "file", "audio", "audio_file"}
CONFIG_FIELD_NAMES = {
    "config",
    "api_config",
    "apiConfig",
    "apiconfig",
    "settings",
}
TEXT_AUDIO_FORMATS = {
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "wav": "audio/wav",
    "wave": "audio/wav",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}


def create_auphonic_router(
    *,
    api_key: str,
    jobs: dict,
    jobs_lock,
    job_queue,
    save_audio_to_temp,
    cleanup_audio_file,
    temp_audio_dir: str,
):
    router = APIRouter(prefix="/api", tags=["auphonic-compat"])

    def validate_auphonic_api_key(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-KEY"),
        bearer_token: str | None = Query(default=None),
    ):
        submitted_keys = [x_api_key, bearer_token]
        if authorization:
            scheme, _, value = authorization.partition(" ")
            submitted_keys.append(value if scheme.lower() == "bearer" else authorization)

        if api_key and api_key in submitted_keys:
            return api_key

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API Key",
        )

    @router.get("/user.json", dependencies=[validate_auphonic_api_key_dependency(validate_auphonic_api_key)])
    def get_user():
        return _success_response(
            {
                "username": "digasystem",
                "email": "",
                "credits": 999999,
                "credits_string": "999:59",
                "remaining_credits": 999999,
                "onetime_credits": 999999,
                "recurring_credits": 0,
            }
        )

    @router.get("/presets.json", dependencies=[validate_auphonic_api_key_dependency(validate_auphonic_api_key)])
    def get_presets():
        return _success_response(
            [
                {
                    "uuid": "deepfilternet3",
                    "preset_name": "DeepFilterNet3",
                    "display_name": "DeepFilterNet3 Audio Cleaning",
                }
            ]
        )

    @router.get("/productions.json", dependencies=[validate_auphonic_api_key_dependency(validate_auphonic_api_key)])
    def list_productions(request: Request):
        with jobs_lock:
            audio_jobs = [
                _production_data(job_id, job, request)
                for job_id, job in jobs.items()
                if job.get("type") in {"audio_cleaning", "audio_cleaning_pending"}
            ]
        return _success_response(audio_jobs)

    @router.post("/simple/productions.json", dependencies=[validate_auphonic_api_key_dependency(validate_auphonic_api_key)])
    async def create_simple_production(request: Request):
        upload, config_dict = await _read_request_payload(request)
        if upload is None:
            raise HTTPException(
                status_code=422,
                detail="input_file is required for simple productions.",
            )

        production_id = await _create_queued_audio_cleaning_job(
            upload=upload,
            config_dict=config_dict,
        )

        with jobs_lock:
            job = jobs[production_id]
            data = _production_data(production_id, job, request)

        return _success_response(data)

    @router.post("/productions.json", dependencies=[validate_auphonic_api_key_dependency(validate_auphonic_api_key)])
    async def create_production(request: Request):
        upload, config_dict = await _read_request_payload(request)
        if upload is not None:
            production_id = await _create_queued_audio_cleaning_job(
                upload=upload,
                config_dict=config_dict,
            )
        else:
            production_id = _create_pending_audio_cleaning_job(config_dict)

        with jobs_lock:
            job = jobs[production_id]
            data = _production_data(production_id, job, request)

        return _success_response(data)

    @router.post("/production/{production_id}/upload.json", dependencies=[validate_auphonic_api_key_dependency(validate_auphonic_api_key)])
    async def upload_production_audio(production_id: str, request: Request):
        upload, config_dict = await _read_request_payload(request)
        if upload is None:
            raise HTTPException(status_code=422, detail="input_file is required.")

        cleaning_config = parse_audio_cleaning_config(config_dict)
        audio = await upload.read()
        audio_path = save_audio_to_temp(audio, upload.filename)
        output_filename = _build_output_filename(upload.filename, cleaning_config.output_format)
        output_path = _build_result_path(temp_audio_dir, production_id, output_filename)

        with jobs_lock:
            job = jobs.get(production_id)
            if not job:
                cleanup_audio_file(audio_path)
                raise HTTPException(status_code=404, detail="Production not found")

            job["status"] = "uploaded"
            job["data_name"] = upload.filename
            job["duration"] = 0
            job["config"] = _stored_cleaning_config(config_dict, cleaning_config)
            job["input_audio_path"] = audio_path
            job["output_path"] = output_path
            job["type"] = "audio_cleaning_pending"
            job.setdefault("auphonic", {})
            job["auphonic"].update(
                {
                    "title": config_dict.get("title") or Path(upload.filename or "audio").stem,
                    "output_filename": output_filename,
                }
            )
            data = _production_data(production_id, job, request)

        return _success_response(data)

    @router.post("/production/{production_id}/start.json", dependencies=[validate_auphonic_api_key_dependency(validate_auphonic_api_key)])
    def start_production(production_id: str, request: Request):
        with jobs_lock:
            job = jobs.get(production_id)
            if not job:
                raise HTTPException(status_code=404, detail="Production not found")

            if job.get("status") in {"queued", "running", "done"}:
                return _success_response(_production_data(production_id, job, request))

            audio_path = job.get("input_audio_path")
            if not audio_path:
                raise HTTPException(status_code=422, detail="No uploaded audio found.")

            job["status"] = "queued"
            job["type"] = "audio_cleaning"

        job_queue.put((production_id, audio_path))

        with jobs_lock:
            data = _production_data(production_id, jobs[production_id], request)

        return _success_response(data)

    @router.get("/production/{production_id}/status.json", dependencies=[validate_auphonic_api_key_dependency(validate_auphonic_api_key)])
    def get_production_status(production_id: str):
        with jobs_lock:
            job = jobs.get(production_id)
            if not job:
                raise HTTPException(status_code=404, detail="Production not found")

            status_code, status_string = _map_job_status(job.get("status"))

        return _success_response(
            {
                "uuid": production_id,
                "status": status_code,
                "status_string": status_string,
            }
        )

    @router.get("/production/{production_id}.json", dependencies=[validate_auphonic_api_key_dependency(validate_auphonic_api_key)])
    def get_production(production_id: str, request: Request):
        with jobs_lock:
            job = jobs.get(production_id)
            if not job:
                raise HTTPException(status_code=404, detail="Production not found")
            data = _production_data(production_id, job, request)

        return _success_response(data)

    @router.get(
        "/download/audio-result/{production_id}/{filename}",
        name="auphonic_download_audio_result",
        dependencies=[validate_auphonic_api_key_dependency(validate_auphonic_api_key)],
    )
    def download_audio_result(production_id: str, filename: str):
        with jobs_lock:
            job = jobs.get(production_id)
            if not job:
                raise HTTPException(status_code=404, detail="Production not found")
            if job.get("status") != "done":
                raise HTTPException(status_code=409, detail="Production is not done")

            output_path = _get_job_output_path(job)

        if not output_path or not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="Audio result not found")

        return FileResponse(
            output_path,
            media_type=_media_type_for_filename(filename),
            filename=filename,
        )

    async def _create_queued_audio_cleaning_job(
        *,
        upload: StarletteUploadFile,
        config_dict: Mapping[str, Any],
    ):
        cleaning_config = parse_audio_cleaning_config(config_dict)
        audio = await upload.read()
        production_id = _new_production_id()
        audio_path = save_audio_to_temp(audio, upload.filename)
        output_filename = _build_output_filename(upload.filename, cleaning_config.output_format)
        output_path = _build_result_path(temp_audio_dir, production_id, output_filename)
        created = datetime.now(timezone.utc).isoformat()

        try:
            with jobs_lock:
                jobs[production_id] = {
                    "id": production_id,
                    "status": "queued",
                    "created_at": created,
                    "data_name": upload.filename,
                    "duration": 0,
                    "config": _stored_cleaning_config(config_dict, cleaning_config),
                    "type": "audio_cleaning",
                    "output_path": output_path,
                    "auphonic": {
                        "title": config_dict.get("title")
                        or Path(upload.filename or "audio").stem,
                        "output_filename": output_filename,
                    },
                }

            job_queue.put((production_id, audio_path))
            return production_id
        except Exception:
            cleanup_audio_file(audio_path)
            raise

    def _create_pending_audio_cleaning_job(config_dict: Mapping[str, Any]):
        cleaning_config = parse_audio_cleaning_config(config_dict)
        production_id = _new_production_id()
        output_filename = _build_output_filename(
            config_dict.get("output_basename") or config_dict.get("title") or "audio",
            cleaning_config.output_format,
        )
        created = datetime.now(timezone.utc).isoformat()

        with jobs_lock:
            jobs[production_id] = {
                "id": production_id,
                "status": "created",
                "created_at": created,
                "data_name": config_dict.get("title") or "audio",
                "duration": 0,
                "config": _stored_cleaning_config(config_dict, cleaning_config),
                "type": "audio_cleaning_pending",
                "output_path": _build_result_path(
                    temp_audio_dir,
                    production_id,
                    output_filename,
                ),
                "auphonic": {
                    "title": config_dict.get("title") or "Audio Cleaning",
                    "output_filename": output_filename,
                },
            }

        return production_id

    def _new_production_id():
        while True:
            production_id = uuid.uuid4().hex[:22]
            with jobs_lock:
                if production_id not in jobs:
                    return production_id

    return router


def validate_auphonic_api_key_dependency(validate_auphonic_api_key):
    from fastapi import Depends

    return Depends(validate_auphonic_api_key)


async def _read_request_payload(request: Request):
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        upload = None
        form_values = {}

        for key, value in form.multi_items():
            if isinstance(value, StarletteUploadFile):
                if key in FILE_FIELD_NAMES or upload is None:
                    upload = value
                continue
            form_values[key] = value

        return upload, _merge_config_values(form_values)

    if "application/json" in content_type:
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=422, detail="JSON body must be an object.")
        return None, parse_audio_cleaning_request_config(data)

    return None, {}


def _merge_config_values(values: Mapping[str, Any]):
    config = {}
    for key in CONFIG_FIELD_NAMES:
        raw_value = values.get(key)
        if raw_value:
            config.update(parse_audio_cleaning_request_config(str(raw_value)))

    for key, value in values.items():
        if key not in CONFIG_FIELD_NAMES:
            config[key] = value

    return config


def _stored_cleaning_config(request_config, cleaning_config):
    return {
        "audio_cleaning_config": cleaning_config.to_dict(),
        "request": dict(request_config),
    }


def _production_data(production_id: str, job: Mapping[str, Any], request: Request):
    status_code, status_string = _map_job_status(job.get("status"))
    auphonic_data = job.get("auphonic") or {}
    data = {
        "uuid": production_id,
        "status": status_code,
        "status_string": status_string,
        "created": job.get("created_at"),
        "title": auphonic_data.get("title") or job.get("data_name") or production_id,
        "metadata": {
            "title": auphonic_data.get("title") or job.get("data_name") or production_id,
        },
        "output_files": [],
    }

    if job.get("status") == "failed":
        data["error_message"] = job.get("error", "Production failed")

    if job.get("status") == "done":
        output_filename = _get_job_output_filename(job)
        data["output_files"] = [
            {
                "format": Path(output_filename).suffix.lstrip(".").lower(),
                "filename": output_filename,
                "download_url": str(
                    request.url_for(
                        "auphonic_download_audio_result",
                        production_id=production_id,
                        filename=output_filename,
                    )
                ),
            }
        ]

    return data


def _map_job_status(status_value):
    if status_value in {"created", "uploaded", "queued"}:
        return 1, "Waiting"
    if status_value == "running":
        return 2, "Processing"
    if status_value == "done":
        return 3, "Done"
    if status_value == "failed":
        return 4, "Error"
    return 0, "Unknown"


def _success_response(data):
    return {
        "status_code": 200,
        "form_errors": {},
        "error_code": None,
        "error_message": "",
        "data": data,
    }


def _build_output_filename(input_filename, output_format):
    source_name = Path(input_filename or "audio").stem or "audio"
    return f"{_safe_filename(source_name)}.{output_format.lower().lstrip('.')}"


def _build_result_path(temp_audio_dir, production_id, output_filename):
    return str(Path(temp_audio_dir) / f"{production_id}_{_safe_filename(output_filename)}")


def _safe_filename(value):
    basename = Path(str(value)).name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", basename).strip("._") or "audio"


def _get_job_output_filename(job: Mapping[str, Any]):
    auphonic_data = job.get("auphonic") or {}
    filename = auphonic_data.get("output_filename")
    if filename:
        return filename

    output_path = _get_job_output_path(job)
    if output_path:
        return Path(output_path).name

    return "audio.wav"


def _get_job_output_path(job: Mapping[str, Any]):
    result = job.get("result") or {}
    return result.get("output_path") or job.get("output_path")


def _media_type_for_filename(filename):
    suffix = Path(filename).suffix.lower().lstrip(".")
    return TEXT_AUDIO_FORMATS.get(suffix, "application/octet-stream")
