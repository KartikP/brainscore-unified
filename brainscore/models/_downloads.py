"""Local pre-flight for the shipped multi-gigabyte model checkpoints.

No network requests or prompts. Like ``brainscore.data.local``, a missing asset
raises instructions naming the source, destination, and next step. Managed
CI/EC2 runs can opt out with ``BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK=1``.
"""

import json
import os
from pathlib import Path
import shutil


SKIP_ENV = 'BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK'


class ModelDownloadRequired(RuntimeError):
    """A large checkpoint needs downloading; the message explains how."""


def _present(filename) -> bool:
    return isinstance(filename, (str, Path)) and Path(filename).is_file() \
        and Path(filename).stat().st_size > 0


def _require_download(identifier, source, destination, approximate_gb, relocate):
    if os.environ.get(SKIP_ENV) == '1':
        return
    destination = Path(destination).expanduser().absolute()
    # The cache may not exist yet. Check its filesystem without creating it.
    existing = destination.parent
    while not existing.exists():
        existing = existing.parent
    required_gb = approximate_gb * 1.1 + 1  # staging/metadata headroom
    try:
        free_gb = shutil.disk_usage(existing).free / 1e9
        disk_status = f'{free_gb:.1f} GB free; budget about {required_gb:.1f} GB.'
        if free_gb < required_gb:
            disk_status += ' Insufficient free disk for this estimate.'
    except OSError as error:
        disk_status = f'Could not check free disk: {error}.'
    raise ModelDownloadRequired('\n'.join([
        f'{identifier} (model weights) requires a download to',
        f'    {destination}',
        '',
        'Why you have to fetch it: pretrained weights are not shipped with Brain-Score.',
        f'Get it from: {source}',
        f'What to download: approximately {approximate_gb:.1f} GB of checkpoint files.',
        f'Free disk: {disk_status}',
        '',
        f'Then either populate the cache yourself, or set {SKIP_ENV}=1 and retry',
        'to allow the download without this check (including in CI/EC2).',
        f'To change the destination: {relocate}',
        'This is a disk estimate, not a RAM or accelerator-memory fit check.',
    ]))


def hf_preflight(identifier: str, repo_id: str, approximate_gb: float) -> dict:
    """Check local weights and return kwargs for every checkpoint loader.

    Fully cached weights load with ``local_files_only=True``: a new upstream
    revision cannot trigger an unannounced download. Config alone or a partial
    shard set does not count as cached. Sizes budget a full checkpoint, even
    when some shards are present; they are approximate, not remaining bytes.
    The opt-out restores the loaders' normal online behaviour and skips disk
    checks. It does not override Hugging Face's own offline settings.
    """
    from huggingface_hub import constants, try_to_load_from_cache
    from transformers.utils import hub

    # Transformers 4 honours legacy overrides; 5 uses the Hub cache directly.
    cache = Path(getattr(hub, 'TRANSFORMERS_CACHE', constants.HF_HUB_CACHE))
    cache = cache.expanduser().absolute()
    kwargs = {'cache_dir': str(cache)}
    if os.environ.get(SKIP_ENV) == '1':
        return kwargs

    def cached(filename):
        return try_to_load_from_cache(repo_id, filename, cache_dir=cache)

    complete = False
    if _present(cached('config.json')):
        for weights in ('model.safetensors', 'pytorch_model.bin'):
            if _present(cached(weights)):
                complete = True
                break
            index = cached(weights + '.index.json')
            if not _present(index):
                continue
            try:
                shards = set(json.loads(Path(index).read_text())['weight_map'].values())
                complete = bool(shards) and all(_present(cached(s)) for s in shards)
            except (OSError, ValueError, KeyError, AttributeError, TypeError):
                complete = False
            if complete:
                break
    if complete:
        return {**kwargs, 'local_files_only': True}
    _require_download(
        identifier, f'https://huggingface.co/{repo_id}',
        cache / ('models--' + repo_id.replace('/', '--')), approximate_gb,
        'set HF_HUB_CACHE before starting Python (or TRANSFORMERS_CACHE if already set).',
    )
    return kwargs


def file_preflight(identifier: str, source: str, destination: Path,
                   approximate_gb: float) -> None:
    """Same guard for a checkpoint downloaded directly to a file."""
    if not _present(destination):
        _require_download(
            identifier, source, destination, approximate_gb,
            'set HF_HOME or XDG_CACHE_HOME before starting Python; '
            'V-JEPA v1 uses the vjepa_v1 subdirectory.',
        )
