"""Hooks on the vision blocks of Kimi K2.5 inside a vLLM server: record block outputs and inactivate chosen
channels while the model takes part in a task.

Start the server with this class mixed into each worker, then install the hooks once through vLLM's
development endpoint (``VLLM_SERVER_DEV_MODE=1``)::

    vllm serve <weights> --trust-remote-code -tp 8 --mm-encoder-tp-mode data \\
        --worker-extension-cls brainscore.models.kimi_k25_local.vllm_hooks.VisionHooks \\
        --no-enable-prefix-caching --mm-processor-cache-gb 0 \\
        --compilation-config '{"cudagraph_mm_encoder": false}'
    curl -X POST localhost:8000/collective_rpc -d '{"method": "kimi_hooks_install", "args": ["<settings json>"]}'

Settings: ``lesion_file`` (required), ``record_dir``, ``record_blocks`` (block indices whose per-picture mean
output is written at each encoder call) and ``full_blocks`` (block indices whose whole output is written: for
the equality check). The lesion file is read at every encoder call, so a new condition needs no restart; after
writing it, reset the encoder cache (``POST /reset_encoder_cache``), else a picture seen before the change is
not encoded again. Lesion file forms (JSON)::

    {"name": "intact"}
    {"name": "selected-K50", "block": 23, "channels": [12, 40, ...]}
    {"name": "L4", "block": 23, "direction": [1152 numbers]}
    {"name": "L1-mean", "block": 23, "channels": [...], "values": [one number per channel]}
    {"name": "L4-mean", "block": 23, "direction": [...], "value": 0.42}

One rule of inactivation: the output of block ``block`` is changed for every token of every picture: the
listed channels are set to 0 (or to the given ``values``: a mean ablation), or the component along the
(normalized) direction is removed (or set to the given ``value``). Each encoder call
writes one line to ``<record_dir>/calls-rank<r>.jsonl`` with the lesion name it ran with.

Limit: this lesion is a hook inside a serving program for one experiment; it is not the library's lesion
channel (an intervention on the ``lesion`` channel of a session), and the subject does not state it.
"""
import json
import os
import time
from functools import partial
from pathlib import Path
from typing import Optional

import torch


def read_lesion(path: str) -> dict:
    """The lesion of the file at ``path``; a missing or malformed file is an error, never 'intact'."""
    try:
        lesion = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(f'The lesion file {path} cannot be read ({error}). Next step: write it '
                           '(for no lesion: {"name": "intact"}) before the next request.') from error
    if 'name' not in lesion or ('block' in lesion) != ('channels' in lesion or 'direction' in lesion):
        raise RuntimeError(f'The lesion file {path} needs a "name", and a "block" with "channels" or '
                           f'"direction" (or neither for no lesion); it holds {sorted(lesion)}.')
    return lesion


def apply_lesion(output: torch.Tensor, lesion: dict) -> torch.Tensor:
    """``output`` (tokens x channels) with the lesion's channels set to 0 (or to ``values``), or its direction's
    component removed (or set to ``value``)."""
    if 'channels' in lesion:
        changed = output.clone()
        index = torch.as_tensor(lesion['channels'], dtype=torch.long, device=output.device)
        changed[..., index] = (0 if 'values' not in lesion else
                               torch.as_tensor(lesion['values'], dtype=output.dtype, device=output.device))
        return changed
    direction = torch.as_tensor(lesion['direction'], dtype=torch.float32, device=output.device)
    direction = direction / direction.norm()
    values = output.float()
    kept = float(lesion.get('value', 0.0))
    return (values - ((values @ direction)[..., None] - kept) * direction).to(output.dtype)


def picture_means(output: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """The mean over the tokens of each picture of a packed encoder batch (pictures x channels)."""
    bounds = cu_seqlens.tolist()
    return torch.stack([output[start:end].float().mean(0) for start, end in zip(bounds[:-1], bounds[1:])])


class VisionHooks:
    """Mixed into each vLLM worker; its methods are called through ``collective_rpc``."""

    def kimi_hooks_install(self, settings: str) -> dict:
        settings = json.loads(settings)
        blocks = self.model_runner.get_model().vision_tower.encoder.blocks
        for handle in getattr(self, '_kimi_handles', []):
            handle.remove()
        record_dir = settings.get('record_dir')
        if record_dir:
            Path(record_dir).mkdir(parents=True, exist_ok=True)
        self._kimi = {'lesion_file': settings['lesion_file'], 'record_dir': record_dir,
                      'record_blocks': set(settings.get('record_blocks', [])),
                      'full_blocks': set(settings.get('full_blocks', [])), 'calls': 0, 'lesion': None}
        read_lesion(self._kimi['lesion_file'])        # fail at install, not at the first picture
        self._kimi_handles = [block.register_forward_hook(partial(self._kimi_hook, index))
                              for index, block in enumerate(blocks)]
        return {'rank': self.rank, 'blocks': len(blocks), 'pid': os.getpid()}

    def kimi_hooks_status(self) -> dict:
        state = getattr(self, '_kimi', None)
        return None if state is None else {'rank': self.rank, 'calls': state['calls'],
                                           'lesion': (state['lesion'] or {}).get('name')}

    def _kimi_hook(self, index: int, module, args, output: torch.Tensor) -> Optional[torch.Tensor]:
        state = self._kimi
        if index == 0:                                # a new encoder call: read the condition once for it
            state['lesion'] = read_lesion(state['lesion_file'])
            state['calls'] += 1
            state['tokens'] = int(output.shape[0])
            self._kimi_log(args)
        lesion = state['lesion']
        if lesion.get('block') == index:
            output = apply_lesion(output, lesion)
        if state['record_dir'] and (index in state['record_blocks'] or index in state['full_blocks']):
            stem = Path(state['record_dir']) / f"rank{self.rank}-call{state['calls']:06d}-block{index:02d}"
            record = {'lesion': lesion['name'], 'cu_seqlens': args[1].cpu(),
                      'means': picture_means(output, args[1]).cpu()}
            if index in state['full_blocks']:
                record['output'] = output.detach().cpu()
            torch.save(record, f'{stem}.pt')
        return output if lesion.get('block') == index else None

    def _kimi_log(self, args) -> None:
        if not self._kimi['record_dir']:
            return
        line = {'call': self._kimi['calls'], 'rank': self.rank, 'time': time.time(),
                'lesion': self._kimi['lesion']['name'], 'tokens': self._kimi['tokens'],
                'pictures': len(args[1]) - 1}
        with open(Path(self._kimi['record_dir']) / f'calls-rank{self.rank}.jsonl', 'a') as log:
            log.write(json.dumps(line) + '\n')
