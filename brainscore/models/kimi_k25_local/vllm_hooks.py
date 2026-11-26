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
    {"name": "W1-K50", "lesions": [{"block": 9, "channels": [...]}, {"block": 23, "channels": [...]}]}

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
    parts = lesion.get('lesions', [lesion] if 'block' in lesion else [])
    loose = 'lesions' not in lesion and 'block' not in lesion and ('channels' in lesion or 'direction' in lesion)
    if 'name' not in lesion or loose or any('block' not in part or not ('channels' in part or 'direction' in part)
                                            for part in parts):
        raise RuntimeError(f'The lesion file {path} needs a "name", and a "block" with "channels" or '
                           f'"direction" (or a list "lesions" of such parts, or neither for no lesion); '
                           f'it holds {sorted(lesion)}.')
    blocks = [part['block'] for part in parts]
    if len(blocks) != len(set(blocks)):
        raise RuntimeError(f'The lesion file {path} names a block twice: {blocks}.')
    return {**lesion, 'by_block': {part['block']: part for part in parts}}


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
        part = lesion['by_block'].get(index)
        if part is not None:
            output = apply_lesion(output, part)
        if state['record_dir'] and (index in state['record_blocks'] or index in state['full_blocks']):
            stem = Path(state['record_dir']) / f"rank{self.rank}-call{state['calls']:06d}-block{index:02d}"
            record = {'lesion': lesion['name'], 'cu_seqlens': args[1].cpu(),
                      'means': picture_means(output, args[1]).cpu()}
            if index in state['full_blocks']:
                record['output'] = output.detach().cpu()
            torch.save(record, f'{stem}.pt')
        return output if part is not None else None

    def _kimi_log(self, args) -> None:
        if not self._kimi['record_dir']:
            return
        line = {'call': self._kimi['calls'], 'rank': self.rank, 'time': time.time(),
                'lesion': self._kimi['lesion']['name'], 'tokens': self._kimi['tokens'],
                'pictures': len(args[1]) - 1}
        with open(Path(self._kimi['record_dir']) / f'calls-rank{self.rank}.jsonl', 'a') as log:
            log.write(json.dumps(line) + '\n')


# The language part: the residual stream after each decoder layer, at the picture's token positions.
# vLLM keeps the stream as two tensors that a decoder layer returns, (hidden, residual); the stream after the layer
# is their sum (the next layer's norm adds them). Under tensor parallel serving both are whole on every rank after the
# all-reduce, so every rank applies a lesion and rank 0 records. The picture positions of a forward step come from the
# model's embedding merge (``embed_input_ids(..., is_multimodal=mask)``), which runs just before the decoder layers.
# Hooks on the decoder layers need the server in eager mode (``--enforce-eager``): a compiled or graph-captured
# forward does not call them.

def language_parts(lesion: dict) -> dict:
    """The language part of a lesion: layer -> part ({"layer", "dims", "values"} or {"layer", "direction", "value"});
    only mean ablations (a zero in the residual stream is not a lesion of a unit)."""
    parts = {}
    for part in lesion.get('language', []):
        if 'layer' not in part or not (('dims' in part and 'values' in part) or ('direction' in part and 'value' in part)):
            raise RuntimeError(f'A language lesion part needs "layer" and "dims" with "values", or "direction" with '
                               f'"value" (a mean ablation); it holds {sorted(part)}.')
        if part['layer'] in parts:
            raise RuntimeError(f"A language lesion names layer {part['layer']} twice.")
        parts[part['layer']] = part
    return parts


def set_stream(hidden: torch.Tensor, residual: torch.Tensor, mask: torch.Tensor, part: dict) -> torch.Tensor:
    """A new residual such that hidden + residual takes the part's values at the masked positions."""
    stream = hidden[mask].float() + residual[mask].float()
    if 'dims' in part:
        index = torch.as_tensor(part['dims'], dtype=torch.long, device=hidden.device)
        target = stream.clone()
        target[:, index] = torch.as_tensor(part['values'], dtype=torch.float32, device=hidden.device)
    else:
        u = torch.as_tensor(part['direction'], dtype=torch.float32, device=hidden.device)
        u = u / u.norm()
        target = stream - ((stream @ u)[:, None] - float(part['value'])) * u
    changed = residual.clone()
    changed[mask] = (target - hidden[mask].float()).to(residual.dtype)
    return changed


class LanguageHooks:
    """Mixed into each vLLM worker beside ``VisionHooks``; installed with ``collective_rpc('kimi_language_install')``."""

    def kimi_language_install(self, settings: str) -> dict:
        settings = json.loads(settings)
        model = self.model_runner.get_model()
        layers = model.language_model.model.layers
        for handle in getattr(self, '_kimi_lang_handles', []):
            handle.remove()
        state = self._kimi_lang = {'lesion_file': settings['lesion_file'], 'record_dir': settings.get('record_dir'),
                                   'mask': None, 'step': 0, 'parts': {}, 'name': None, 'rows': {}}
        if state['record_dir']:
            Path(state['record_dir']).mkdir(parents=True, exist_ok=True)
        if not hasattr(self, '_kimi_embed'):
            self._kimi_embed = model.embed_input_ids

            def embed(input_ids, *args, **kwargs):
                self._kimi_lang_step(input_ids, kwargs.get('is_multimodal'))
                return self._kimi_embed(input_ids, *args, **kwargs)
            model.embed_input_ids = embed
        self._kimi_lang_handles = [layer.register_forward_hook(partial(self._kimi_lang_hook, index))
                                   for index, layer in enumerate(layers)]
        return {'rank': self.rank, 'layers': len(layers)}

    def _kimi_lang_step(self, input_ids, mask) -> None:
        state = self._kimi_lang
        self._kimi_lang_flush()
        lesion = read_lesion(state['lesion_file'])
        state.update(step=state['step'] + 1, parts=language_parts(lesion), name=lesion['name'], rows={},
                     mask=None if mask is None or not bool(mask.any()) else mask.to(input_ids.device),
                     tokens=int(input_ids.shape[0]))

    def _kimi_lang_hook(self, index: int, module, args, output):
        state = self._kimi_lang
        mask = state['mask']
        if mask is None:
            return None
        hidden, residual = output
        if mask.shape[0] != hidden.shape[0]:          # a padded step: the picture positions are the first tokens
            mask = torch.nn.functional.pad(mask, (0, hidden.shape[0] - mask.shape[0]))
        part = state['parts'].get(index)
        if part is not None:
            residual = set_stream(hidden, residual, mask, part)
        if state['record_dir'] and self.rank == 0:
            stream = hidden.float() + residual.float()
            state['rows'][index] = (stream[mask].mean(0).half().cpu(), stream[state['tokens'] - 1].half().cpu())
        return (hidden, residual) if part is not None else None

    def _kimi_lang_flush(self) -> None:
        state = self._kimi_lang
        if state['rows'] and state['record_dir']:
            layers = sorted(state['rows'])
            torch.save({'lesion': state['name'], 'layers': layers,
                        'picture_mean': torch.stack([state['rows'][k][0] for k in layers]),
                        'last_position': torch.stack([state['rows'][k][1] for k in layers]),
                        'picture_tokens': int(state['mask'].sum())},
                       Path(state['record_dir']) / f"lang-step{state['step']:06d}.pt")
        state['rows'] = {}

    def kimi_language_flush(self) -> dict:
        self._kimi_lang_flush()
        return {'rank': self.rank, 'step': self._kimi_lang['step']}


class KimiHooks(VisionHooks, LanguageHooks):
    """Both sets of hooks in one worker extension (``--worker-extension-cls vllm_hooks.KimiHooks``)."""
