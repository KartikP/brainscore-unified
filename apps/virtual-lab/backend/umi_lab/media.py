"""Prepare explicit paired video/audio stimuli without loading a model."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import wave

MAX_AUDIO_BYTES = 128 * 1024 * 1024
_EXTRACTION_LOCK = threading.Lock()


def ffmpeg_executable() -> str:
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except (ImportError, RuntimeError) as error:
        raise ValueError('Audio extraction needs FFmpeg. Install brainscore-lab with its media dependencies, or install imageio-ffmpeg.') from error


def extract_audio(uploads: Path, asset: str) -> dict:
    with _EXTRACTION_LOCK:
        return _extract_audio(uploads, asset)


def _extract_audio(uploads: Path, asset: str) -> dict:
    if not isinstance(asset, str) or Path(asset).name != asset or Path(asset).suffix.lower() not in ('.mp4', '.mov', '.webm'):
        raise ValueError('Choose an uploaded video asset')
    source = (uploads / asset).resolve()
    if not source.is_relative_to(uploads.resolve()) or not source.is_file():
        raise ValueError('Unknown uploaded video')
    target = source.with_name(source.stem + '.audio-16k-v1.wav')
    metadata = target.with_suffix('.json')
    stat = source.stat()
    identity = {'source_video': asset, 'source_bytes': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns}
    if target.is_file() and metadata.is_file():
        try:
            saved = json.loads(metadata.read_text())
            if all(saved.get(key) == value for key, value in identity.items()):
                return saved
        except (ValueError, OSError):
            pass
    with tempfile.TemporaryDirectory(dir=uploads, prefix='audio-') as temporary:
        output = Path(temporary) / 'soundtrack.wav'
        command = [ffmpeg_executable(), '-hide_banner', '-loglevel', 'error', '-nostdin',
                   '-threads', '1', '-filter_threads', '1', '-protocol_whitelist', 'file,pipe',
                   '-copyts', '-start_at_zero', '-i', str(source), '-map', '0:a:0?', '-vn',
                   '-af', 'aresample=16000:async=1:first_pts=0', '-ac', '1', '-ar', '16000',
                   '-c:a', 'pcm_s16le', '-fs', str(MAX_AUDIO_BYTES), '-f', 'wav', str(output)]
        try:
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    timeout=120, check=False)
        except subprocess.TimeoutExpired as error:
            raise ValueError('Audio extraction exceeded 120 seconds; use a shorter clip or supply a WAV file') from error
        except OSError as error:
            raise ValueError(f'Cannot start FFmpeg: {error}') from error
        detail = result.stderr.decode(errors='replace')
        if result.returncode:
            if 'does not contain any stream' in detail:
                return {**identity, 'audio_asset': None, 'status': 'no_audio'}
            raise ValueError('Cannot extract this video soundtrack: ' + detail[-500:])
        if not output.is_file() or output.stat().st_size >= MAX_AUDIO_BYTES:
            raise ValueError('The extracted audio exceeds the 128 MiB limit; use a shorter clip')
        try:
            with wave.open(str(output), 'rb') as audio:
                if audio.getnframes() == 0:
                    raise ValueError('The video soundtrack contains no samples')
                duration = audio.getnframes() / audio.getframerate()
        except wave.Error as error:
            raise ValueError('FFmpeg did not produce a valid PCM WAV file') from error
        saved = {**identity, 'audio_asset': target.name, 'status': 'ready', 'sample_rate': 16000,
                 'channels': 1, 'sample_format': 'PCM16', 'duration_s': duration,
                 'preparation': 'first audio stream; mono PCM16 WAV; 16000 Hz; timestamps relative to container start, with leading gaps retained'}
        output.replace(target)
        metadata.write_text(json.dumps(saved, indent=2) + '\n')
        return saved
