import gc
import os

import nltk

import torch
import whisperx
from torch_compat import allow_trusted_pyannote_checkpoints


HF_TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "PYANNOTE_AUTH_TOKEN",
)


def get_hf_token():
    for env_name in HF_TOKEN_ENV_NAMES:
        token = os.getenv(env_name)
        if token:
            return token
    return None


def parse_bool(value, default=False):
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def clear_gpu_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_nltk_data():
    nltk_data_dir = os.getenv("NLTK_DATA", "/home/appuser/nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    for package in ("punkt_tab", "punkt"):
        print(f"Precargando datos NLTK {package} en {nltk_data_dir}...")
        nltk.download(package, download_dir=nltk_data_dir, quiet=False)


def preload_deepfilter_model():
    preload_deepfilter = parse_bool(
        os.getenv("AUDIO_CLEANING_PRELOAD_DEEPFILTER"),
        True,
    )
    if not preload_deepfilter:
        return

    model_name = os.getenv("AUDIO_CLEANING_DEEPFILTER_MODEL", "DeepFilterNet3")
    post_filter = parse_bool(os.getenv("AUDIO_CLEANING_POST_FILTER"), True)

    print(f"Precargando modelo DeepFilterNet '{model_name}'...")
    from df.enhance import init_df

    model, df_state, _, _ = init_df(
        model_name,
        post_filter=post_filter,
        log_level="ERROR",
        log_file=None,
    )
    del model
    del df_state
    clear_gpu_cache()


def preload_whisper_model(device, model_name, compute_type):
    preload_whisper = parse_bool(os.getenv("WHISPERX_PRELOAD_MODEL"), True)
    if not preload_whisper:
        print("No se precarga WhisperX porque WHISPERX_PRELOAD_MODEL=false.")
        return

    print(f"Precargando modelo WhisperX '{model_name}' en {device}...")
    model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
    del model
    clear_gpu_cache()


def main():
    allow_trusted_pyannote_checkpoints()

    device = os.getenv("WHISPERX_DEVICE", "cuda")
    model_name = os.getenv("WHISPERX_MODEL", "medium")
    compute_type = os.getenv("WHISPERX_COMPUTE_TYPE", "int8_float16")
    diarization_model = os.getenv(
        "WHISPERX_DIARIZATION_MODEL",
        "pyannote/speaker-diarization-3.1",
    )

    ensure_nltk_data()

    preload_whisper_model(device, model_name, compute_type)

    align_languages = os.getenv("WHISPERX_PRELOAD_ALIGN_LANGUAGES", "")
    for language in [lang.strip() for lang in align_languages.split(",") if lang.strip()]:
        print(f"Precargando modelo de alineacion para '{language}'...")
        align_model, _ = whisperx.load_align_model(language_code=language, device=device)
        del align_model
        clear_gpu_cache()

    preload_diarization = parse_bool(os.getenv("WHISPERX_PRELOAD_DIARIZATION"), True)
    hf_token = get_hf_token()
    if preload_diarization and hf_token:
        print(f"Precargando modelo de diarizacion '{diarization_model}'...")
        from whisperx.diarize import DiarizationPipeline

        diarize_model = DiarizationPipeline(
            model_name=diarization_model,
            token=hf_token,
            device=device,
        )
        del diarize_model
        clear_gpu_cache()
    elif preload_diarization:
        print("No se precarga diarizacion porque no hay HF_TOKEN configurado.")

    preload_deepfilter_model()

    print("Precarga de modelos completada.")


if __name__ == "__main__":
    main()
