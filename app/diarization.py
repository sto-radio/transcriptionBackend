import os
import threading

import whisperx


DIARIZATION_MODEL = os.getenv(
    "WHISPERX_DIARIZATION_MODEL",
    "pyannote/speaker-diarization-3.1",
)
DIARIZATION_ENABLED_BY_DEFAULT = os.getenv("WHISPERX_DIARIZE", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HF_TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "PYANNOTE_AUTH_TOKEN",
)

diarization_model = None
diarization_model_name = None
diarization_model_device = None
diarization_model_lock = threading.Lock()


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
        "model": first_present(
            speaker_config.get("model"),
            config_dict.get("diarization_model"),
            DIARIZATION_MODEL,
        ),
        "num_speakers": num_speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "fill_nearest": parse_bool(speaker_config.get("fill_nearest"), True),
    }


def get_diarization_model(model_name, device):
    global diarization_model, diarization_model_name, diarization_model_device

    if (
        diarization_model is not None
        and diarization_model_name == model_name
        and diarization_model_device == device
    ):
        return diarization_model

    with diarization_model_lock:
        if (
            diarization_model is not None
            and diarization_model_name == model_name
            and diarization_model_device == device
        ):
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
        diarization_model_device = device
        return diarization_model


def apply_diarization(audio_file, aligned_result, config_dict, device):
    diarization_config = get_diarization_config(config_dict)
    if not diarization_config["enabled"]:
        return aligned_result, diarization_config, []

    print("Realizando diarización con:", diarization_config["model"])
    diarize_model = get_diarization_model(diarization_config["model"], device)
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
    speaker_segments = diarize_segments[["start", "end", "speaker"]].to_dict(
        "records"
    )
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
