#!/usr/bin/env python3
from pathlib import Path
import sys

CONTENT = r"""from hashlib import sha256
from pathlib import Path
from typing import Optional, Tuple, Union
import os
import time

import librosa
import torch

from piper_train.vits.mel_processing import spectrogram_torch

from .trim import trim_silence
from .vad import SileroVoiceActivityDetector

_DIR = Path(__file__).parent
_LOCK_TIMEOUT_SEC = 300.0
_LOCK_POLL_SEC = 0.05


def make_silence_detector() -> SileroVoiceActivityDetector:
    silence_model = _DIR / "models" / "silero_vad.onnx"
    return SileroVoiceActivityDetector(silence_model)


def _lock_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.lock")


def _acquire_lock(lock_path: Path) -> None:
    deadline = time.monotonic() + _LOCK_TIMEOUT_SEC
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(_LOCK_POLL_SEC)


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _atomic_torch_save(obj, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _safe_torch_load(path: Path):
    try:
        return torch.load(path)
    except (EOFError, OSError, RuntimeError):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None


def _build_audio_norm_tensor(
    audio_path: Path,
    detector: SileroVoiceActivityDetector,
    sample_rate: int,
    silence_threshold: float,
    silence_samples_per_chunk: int,
    silence_keep_chunks_before: int,
    silence_keep_chunks_after: int,
) -> torch.FloatTensor:
    vad_sample_rate = 16000
    audio_16khz, _sr = librosa.load(path=audio_path, sr=vad_sample_rate)

    offset_sec, duration_sec = trim_silence(
        audio_16khz,
        detector,
        threshold=silence_threshold,
        samples_per_chunk=silence_samples_per_chunk,
        sample_rate=vad_sample_rate,
        keep_chunks_before=silence_keep_chunks_before,
        keep_chunks_after=silence_keep_chunks_after,
    )

    audio_norm_array, _sr = librosa.load(
        path=audio_path,
        sr=sample_rate,
        offset=offset_sec,
        duration=duration_sec,
    )
    return torch.FloatTensor(audio_norm_array).unsqueeze(0)


def _ensure_audio_norm_tensor(
    audio_path: Path,
    audio_norm_path: Path,
    detector: SileroVoiceActivityDetector,
    sample_rate: int,
    silence_threshold: float,
    silence_samples_per_chunk: int,
    silence_keep_chunks_before: int,
    silence_keep_chunks_after: int,
    ignore_cache: bool,
) -> torch.FloatTensor:
    lock_path = _lock_path(audio_norm_path)
    _acquire_lock(lock_path)
    try:
        if not ignore_cache and audio_norm_path.exists():
            cached = _safe_torch_load(audio_norm_path)
            if cached is not None:
                return cached

        audio_norm_tensor = _build_audio_norm_tensor(
            audio_path=audio_path,
            detector=detector,
            sample_rate=sample_rate,
            silence_threshold=silence_threshold,
            silence_samples_per_chunk=silence_samples_per_chunk,
            silence_keep_chunks_before=silence_keep_chunks_before,
            silence_keep_chunks_after=silence_keep_chunks_after,
        )
        _atomic_torch_save(audio_norm_tensor, audio_norm_path)
        return audio_norm_tensor
    finally:
        _release_lock(lock_path)


def _ensure_audio_spec(
    audio_norm_tensor: Optional[torch.FloatTensor],
    audio_spec_path: Path,
    audio_norm_path: Path,
    audio_path: Path,
    detector: SileroVoiceActivityDetector,
    sample_rate: int,
    silence_threshold: float,
    silence_samples_per_chunk: int,
    silence_keep_chunks_before: int,
    silence_keep_chunks_after: int,
    filter_length: int,
    window_length: int,
    hop_length: int,
    ignore_cache: bool,
) -> None:
    lock_path = _lock_path(audio_spec_path)
    _acquire_lock(lock_path)
    try:
        if not ignore_cache and audio_spec_path.exists():
            cached = _safe_torch_load(audio_spec_path)
            if cached is not None:
                return

        if audio_norm_tensor is None:
            audio_norm_tensor = _ensure_audio_norm_tensor(
                audio_path=audio_path,
                audio_norm_path=audio_norm_path,
                detector=detector,
                sample_rate=sample_rate,
                silence_threshold=silence_threshold,
                silence_samples_per_chunk=silence_samples_per_chunk,
                silence_keep_chunks_before=silence_keep_chunks_before,
                silence_keep_chunks_after=silence_keep_chunks_after,
                ignore_cache=ignore_cache,
            )

        audio_spec_tensor = spectrogram_torch(
            y=audio_norm_tensor,
            n_fft=filter_length,
            sampling_rate=sample_rate,
            hop_size=hop_length,
            win_size=window_length,
            center=False,
        ).squeeze(0)
        _atomic_torch_save(audio_spec_tensor, audio_spec_path)
    finally:
        _release_lock(lock_path)


def cache_norm_audio(
    audio_path: Union[str, Path],
    cache_dir: Union[str, Path],
    detector: SileroVoiceActivityDetector,
    sample_rate: int,
    silence_threshold: float = 0.2,
    silence_samples_per_chunk: int = 480,
    silence_keep_chunks_before: int = 2,
    silence_keep_chunks_after: int = 2,
    filter_length: int = 1024,
    window_length: int = 1024,
    hop_length: int = 256,
    ignore_cache: bool = False,
) -> Tuple[Path, Path]:
    audio_path = Path(audio_path).absolute()
    cache_dir = Path(cache_dir)

    audio_cache_id = sha256(str(audio_path).encode()).hexdigest()
    audio_norm_path = cache_dir / f"{audio_cache_id}.pt"
    audio_spec_path = cache_dir / f"{audio_cache_id}.spec.pt"

    audio_norm_tensor: Optional[torch.FloatTensor] = None
    if ignore_cache or (not audio_norm_path.exists()):
        audio_norm_tensor = _ensure_audio_norm_tensor(
            audio_path=audio_path,
            audio_norm_path=audio_norm_path,
            detector=detector,
            sample_rate=sample_rate,
            silence_threshold=silence_threshold,
            silence_samples_per_chunk=silence_samples_per_chunk,
            silence_keep_chunks_before=silence_keep_chunks_before,
            silence_keep_chunks_after=silence_keep_chunks_after,
            ignore_cache=ignore_cache,
        )

    if ignore_cache or (not audio_spec_path.exists()):
        _ensure_audio_spec(
            audio_norm_tensor=audio_norm_tensor,
            audio_spec_path=audio_spec_path,
            audio_norm_path=audio_norm_path,
            audio_path=audio_path,
            detector=detector,
            sample_rate=sample_rate,
            silence_threshold=silence_threshold,
            silence_samples_per_chunk=silence_samples_per_chunk,
            silence_keep_chunks_before=silence_keep_chunks_before,
            silence_keep_chunks_after=silence_keep_chunks_after,
            filter_length=filter_length,
            window_length=window_length,
            hop_length=hop_length,
            ignore_cache=ignore_cache,
        )

    return audio_norm_path, audio_spec_path
"""


def main() -> int:
    if len(sys.argv) != 2:
        print('Usage: patch_norm_audio_cache.py <path-to-piper_train/norm_audio/__init__.py>', file=sys.stderr)
        return 1
    path = Path(sys.argv[1]).expanduser().resolve()
    path.write_text(CONTENT, encoding='utf-8')
    print(f'Patched {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
