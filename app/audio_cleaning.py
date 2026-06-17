import json
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_NOISE_ATTENUATION_PERCENT = int(
    os.getenv(
        "AUDIO_CLEANING_NOISE_ATTENUATION_PERCENT",
        os.getenv("AUDIO_CLEANING_PERCENT", "100"),
    )
)
DEFAULT_DEEPFILTER_MODEL = os.getenv(
    "AUDIO_CLEANING_DEEPFILTER_MODEL",
    "DeepFilterNet3",
)
DEFAULT_TARGET_LOUDNESS_LUFS = float(
    os.getenv("AUDIO_CLEANING_TARGET_LOUDNESS_LUFS", "-16")
)
DEFAULT_LOUDNESS_RANGE_LU = float(
    os.getenv("AUDIO_CLEANING_LOUDNESS_RANGE_LU", "11")
)
DEFAULT_TRUE_PEAK_DB = float(os.getenv("AUDIO_CLEANING_TRUE_PEAK_DB", "-1.5"))
DEFAULT_MAX_ATTENUATION_DB = float(
    os.getenv("AUDIO_CLEANING_MAX_ATTENUATION_DB", "30")
)

_deepfilter_models = {}
_deepfilter_lock = threading.Lock()


class AudioCleaningDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioCleaningConfig:
    noise_attenuation_percent: int = DEFAULT_NOISE_ATTENUATION_PERCENT
    deepfilter_model: str = DEFAULT_DEEPFILTER_MODEL
    normalize_loudness: bool = True
    target_loudness_lufs: float = DEFAULT_TARGET_LOUDNESS_LUFS
    loudness_range_lu: float = DEFAULT_LOUDNESS_RANGE_LU
    true_peak_db: float = DEFAULT_TRUE_PEAK_DB
    max_attenuation_db: float = DEFAULT_MAX_ATTENUATION_DB
    post_filter: bool = True
    compensate_delay: bool = True
    output_format: str = "wav"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class AudioCleaningResult:
    input_path: str
    output_path: str
    config: AudioCleaningConfig
    steps: list[str]

    def to_dict(self):
        data = asdict(self)
        data["config"] = self.config.to_dict()
        return data


def parse_audio_cleaning_request_config(config: Mapping[str, Any] | str | None):
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if not isinstance(config, str):
        raise ValueError("La configuracion de limpieza debe ser JSON o clave=valor.")

    raw_config = config.strip()
    if not raw_config:
        return {}

    try:
        parsed_config = json.loads(raw_config)
    except json.JSONDecodeError:
        return _parse_key_value_config(raw_config)

    if not isinstance(parsed_config, dict):
        raise ValueError("La configuracion JSON debe ser un objeto.")

    return parsed_config


def get_audio_cleaning_config(config_dict: Mapping[str, Any] | str | None = None):
    config_dict = parse_audio_cleaning_request_config(config_dict)
    audio_config = (
        config_dict.get("audio_cleaning_config")
        or config_dict.get("cleaning_config")
        or {}
    )

    return AudioCleaningConfig(
        noise_attenuation_percent=_clamp_percent(
            _first_present(
                _get_requested_noise_attenuation_percent(config_dict),
                DEFAULT_NOISE_ATTENUATION_PERCENT,
            )
        ),
        deepfilter_model=str(
            _first_present(
                audio_config.get("deepfilter_model"),
                audio_config.get("model"),
                config_dict.get("deepfilter_model"),
                DEFAULT_DEEPFILTER_MODEL,
            )
        ),
        normalize_loudness=_parse_bool(
            _first_present(
                audio_config.get("normalize_loudness"),
                audio_config.get("loudness_normalization"),
                audio_config.get("normloudness"),
                config_dict.get("normalize_loudness"),
                config_dict.get("normloudness"),
            ),
            default=True,
        ),
        target_loudness_lufs=_parse_float(
            _first_present(
                audio_config.get("target_loudness_lufs"),
                audio_config.get("target_lufs"),
                audio_config.get("loudness_target_level"),
                audio_config.get("loudnesstarget"),
                config_dict.get("target_loudness_lufs"),
                config_dict.get("target_lufs"),
                config_dict.get("loudness_target_level"),
                config_dict.get("loudnesstarget"),
                DEFAULT_TARGET_LOUDNESS_LUFS,
            ),
            default=DEFAULT_TARGET_LOUDNESS_LUFS,
        ),
        loudness_range_lu=_parse_float(
            _first_present(
                audio_config.get("loudness_range_lu"),
                audio_config.get("lra"),
                DEFAULT_LOUDNESS_RANGE_LU,
            ),
            default=DEFAULT_LOUDNESS_RANGE_LU,
        ),
        true_peak_db=_parse_float(
            _first_present(
                audio_config.get("true_peak_db"),
                audio_config.get("true_peak"),
                audio_config.get("loudness_peak_limit"),
                audio_config.get("maxpeak"),
                config_dict.get("true_peak_db"),
                config_dict.get("true_peak"),
                config_dict.get("loudness_peak_limit"),
                config_dict.get("maxpeak"),
                DEFAULT_TRUE_PEAK_DB,
            ),
            default=DEFAULT_TRUE_PEAK_DB,
        ),
        max_attenuation_db=_parse_float(
            _first_present(
                audio_config.get("max_attenuation_db"),
                DEFAULT_MAX_ATTENUATION_DB,
            ),
            default=DEFAULT_MAX_ATTENUATION_DB,
        ),
        post_filter=_parse_bool(audio_config.get("post_filter"), default=True),
        compensate_delay=_parse_bool(
            audio_config.get("compensate_delay"),
            default=True,
        ),
        output_format=str(
            _first_present(
                audio_config.get("output_format"),
                audio_config.get("transcode_kind"),
                audio_config.get("format"),
                config_dict.get("output_format"),
                config_dict.get("transcode_kind"),
                config_dict.get("format"),
                "wav",
            )
        ).lower().lstrip("."),
    )


def parse_audio_cleaning_config(config_dict: Mapping[str, Any] | str | None = None):
    config_dict = parse_audio_cleaning_request_config(config_dict)
    noise_attenuation_percent = _get_requested_noise_attenuation_percent(config_dict)

    if noise_attenuation_percent is not None:
        if (
            isinstance(noise_attenuation_percent, bool)
            or not isinstance(noise_attenuation_percent, int)
        ):
            raise ValueError(
                "noise_attenuation_percent debe ser un entero entre 0 y 100."
            )
        if not 0 <= noise_attenuation_percent <= 100:
            raise ValueError(
                "noise_attenuation_percent debe estar entre 0 y 100."
            )

    return get_audio_cleaning_config(config_dict)


def clean_audio_file(
    input_path: str,
    output_path: str | None = None,
    config: AudioCleaningConfig | Mapping[str, Any] | None = None,
):
    cleaning_config = _ensure_config(config)
    output_path = output_path or _build_output_path(input_path, cleaning_config)
    temp_paths = []
    steps = []

    try:
        current_path = input_path
        if cleaning_config.noise_attenuation_percent > 0:
            denoised_path = _make_temp_audio_path("wav")
            temp_paths.append(denoised_path)
            _run_deepfilternet(current_path, denoised_path, cleaning_config)
            current_path = denoised_path
            steps.append("deepfilternet")

        if cleaning_config.normalize_loudness:
            _normalize_loudness(current_path, output_path, cleaning_config)
            steps.append("loudness_normalization")
        elif current_path != output_path:
            shutil.copyfile(current_path, output_path)

        return AudioCleaningResult(
            input_path=input_path,
            output_path=output_path,
            config=cleaning_config,
            steps=steps,
        )
    finally:
        for temp_path in temp_paths:
            _remove_temp_file(temp_path)


def get_deepfilter_attenuation_limit_db(config: AudioCleaningConfig):
    if config.noise_attenuation_percent >= 100:
        return None
    if config.noise_attenuation_percent <= 0:
        return 0.0

    return round(
        config.max_attenuation_db * config.noise_attenuation_percent / 100,
        2,
    )


def _run_deepfilternet(
    input_path: str,
    output_path: str,
    config: AudioCleaningConfig,
):
    try:
        import torchaudio
        from df.enhance import enhance, init_df
        from df.io import load_audio, resample
    except ImportError as exc:
        raise AudioCleaningDependencyError(
            "Faltan dependencias para la limpieza de audio. "
            "Instala deepfilternet y torchaudio en la imagen antes de usar "
            "este endpoint."
        ) from exc

    model, df_state = _get_deepfilter_model(
        init_df,
        config.deepfilter_model,
        config.post_filter,
    )
    model_sr = df_state.sr()
    audio, metadata = load_audio(input_path, sr=model_sr)
    enhanced_audio = enhance(
        model,
        df_state,
        audio,
        pad=config.compensate_delay,
        atten_lim_db=get_deepfilter_attenuation_limit_db(config),
    )

    output_sr = getattr(metadata, "sample_rate", model_sr) or model_sr
    if output_sr != model_sr:
        enhanced_audio = resample(enhanced_audio.to("cpu"), model_sr, output_sr)

    torchaudio.save(output_path, enhanced_audio.to("cpu"), output_sr)


def _normalize_loudness(
    input_path: str,
    output_path: str,
    config: AudioCleaningConfig,
):
    output_parent = Path(output_path).parent
    if str(output_parent):
        output_parent.mkdir(parents=True, exist_ok=True)

    loudnorm_filter = (
        f"loudnorm=I={config.target_loudness_lufs}:"
        f"LRA={config.loudness_range_lu}:TP={config.true_peak_db}"
    )
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-af",
        loudnorm_filter,
        output_path,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "No se pudo normalizar el loudness del audio: "
            f"{completed.stderr.strip()}"
        )


def _get_deepfilter_model(init_df, model_name: str, post_filter: bool):
    cache_key = (model_name, post_filter)
    cached = _deepfilter_models.get(cache_key)
    if cached is not None:
        return cached

    with _deepfilter_lock:
        cached = _deepfilter_models.get(cache_key)
        if cached is not None:
            return cached

        model, df_state, _, _ = init_df(
            model_name,
            post_filter=post_filter,
            log_level="ERROR",
            log_file=None,
        )
        _deepfilter_models[cache_key] = (model, df_state)
        return model, df_state


def _ensure_config(config):
    if isinstance(config, AudioCleaningConfig):
        return config
    return get_audio_cleaning_config(config)


def _build_output_path(input_path: str, config: AudioCleaningConfig):
    input_file = Path(input_path)
    return str(input_file.with_name(f"{input_file.stem}_clean.{config.output_format}"))


def _make_temp_audio_path(output_format: str):
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{output_format}",
    ) as temp_audio:
        return temp_audio.name


def _remove_temp_file(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"No se pudo eliminar el audio temporal {path}: {exc}")


def _clamp_percent(value):
    try:
        percent = int(value)
    except (TypeError, ValueError):
        percent = DEFAULT_NOISE_ATTENUATION_PERCENT
    return max(0, min(100, percent))


def _get_requested_noise_attenuation_percent(config_dict):
    audio_config = (
        config_dict.get("audio_cleaning_config")
        or config_dict.get("cleaning_config")
        or {}
    )
    explicit_percent = _first_present(
        audio_config.get("noise_attenuation_percent"),
        audio_config.get("noise_attenuation_percentage"),
        audio_config.get("noise_attenuation"),
        audio_config.get("cleaning_percent"),
        audio_config.get("cleaning_percentage"),
        audio_config.get("cleaning"),
        audio_config.get("denoiseamount"),
        config_dict.get("noise_attenuation_percent"),
        config_dict.get("noise_attenuation_percentage"),
        config_dict.get("noise_attenuation"),
        config_dict.get("cleaning_percent"),
        config_dict.get("denoiseamount"),
    )
    if explicit_percent is not None:
        return explicit_percent

    return _normalize_enhancement_level(
        _first_present(
            audio_config.get("enhancement_level"),
            config_dict.get("enhancement_level"),
        )
    )


def _normalize_enhancement_level(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value

    try:
        level = float(value)
    except (TypeError, ValueError):
        return value

    if 0 <= level <= 1:
        return int(round(level * 100))
    return int(round(level))


def _parse_key_value_config(raw_config: str):
    parsed_config = {}
    entries = raw_config.replace("\n", ";").split(";")

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                "La configuracion debe usar el formato clave=valor separado por ';'."
            )

        key, value = entry.split("=", 1)
        parsed_config[key.strip()] = _parse_config_value(value.strip())

    return parsed_config


def _parse_config_value(value: str):
    lower_value = value.lower()
    if lower_value in {"true", "yes", "on"}:
        return True
    if lower_value in {"false", "no", "off"}:
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled", ""}:
            return False
    return default


def _parse_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None
