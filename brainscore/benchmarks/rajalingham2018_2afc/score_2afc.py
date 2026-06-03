"""Driver: run the faithful Rajalingham2018 2-AFC on a model and score it.

Every path is exposed through a ``process()``-bearing model so the run is
Witness-recordable (``with Witness(model): model.process(montage_stim_set)``):

  --path generation : a VLM reads the composed montage and answers LEFT/RIGHT
  --path similarity : nearest object token to the sample in feature space
  --path random     : coin flip (chance null)

Montages are pre-composed to disk so the model sees one inspectable image per
trial. The per-trial choice -> chosen object -> benchmark.score_choices (i1/i2n
against the human pool). Witness panels show exactly what the model saw + answered.

Run on EC2 (GPU) for generation/similarity; the random path runs anywhere.
    python score_2afc.py --path generation --model Qwen/Qwen2.5-VL-3B-Instruct --n_images 240 --out /tmp/raj2afc_qwen3b
"""
import argparse
import json
import os
from typing import Any, Callable, Dict, Set

import numpy as np
import pandas as pd

from brainscore_core.model_interface import (
    EnvironmentStep, StateChange, TaskContext, UnifiedModel)
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly

from . import benchmark as B
from .montage import compose_montage

INSTRUCTION = ("The image shows a SAMPLE object on top and two options below (LEFT and RIGHT). "
               "Exactly one option is the same object as the sample. "
               "Reason briefly, then end with a line: 'Answer: LEFT' or 'Answer: RIGHT'.")


# --------------------------------------------------------------------------- #
# Montage composition (one inspectable image per trial)
# --------------------------------------------------------------------------- #
def compose_montages(trials, n_images, out_dir, seed=0):
    """Compose montages for a subsample of test images. Returns a stim DataFrame
    with one row per trial: image_path (montage), plus the trial metadata."""
    from brainscore_vision import load_stimulus_set
    ss = load_stimulus_set('objectome.public')
    path_of = lambda iid: str(ss.get_stimulus(iid))
    os.makedirs(out_dir, exist_ok=True)

    images = sorted(trials['image_id'].unique())
    rng = np.random.RandomState(seed)
    if n_images and n_images < len(images):
        images = sorted(rng.choice(images, size=n_images, replace=False).tolist())
    sub = trials[trials['image_id'].isin(images)].reset_index(drop=True)

    rows = []
    for i, t in sub.iterrows():
        sample_tok = path_of(t['token_sample_id'])
        dist_tok = path_of(t['token_dist_id'])
        sample_left = bool(rng.rand() < 0.5)               # randomize side
        left_path, right_path = (sample_tok, dist_tok) if sample_left else (dist_tok, sample_tok)
        left_obj, right_obj = (t['sample_obj'], t['dist_obj']) if sample_left else (t['dist_obj'], t['sample_obj'])
        montage_path = os.path.join(out_dir, f'trial_{i:05d}.png')
        if not os.path.exists(montage_path):
            compose_montage(path_of(t['image_id']), left_path, right_path).save(montage_path)
        rows.append({
            'stimulus_id': f'trial_{i:05d}', 'image_path': montage_path,
            'image_id': t['image_id'], 'sample_obj': t['sample_obj'], 'dist_obj': t['dist_obj'],
            'left_obj': left_obj, 'right_obj': right_obj,
            'sample_path': path_of(t['image_id']), 'left_path': left_path, 'right_path': right_path,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# A process()-bearing 2-AFC model (so the run is Witness-recordable)
# --------------------------------------------------------------------------- #
class TwoAFCModel(UnifiedModel):
    """Wraps a per-trial ``choose(row) -> 'LEFT'|'RIGHT'`` into a UnifiedModel
    whose ``process(stim_set)`` returns the chosen objects as a BehavioralAssembly."""

    COLUMN_TO_MODALITY = {'image_path': 'vision'}   # so Witness renders the montage

    def __init__(self, identifier, choose: Callable[[Any], str], mode='generation'):
        self._identifier = identifier
        self._choose = choose
        self._witness_mode = mode      # how the Witness should label this run
        self._task_context = TaskContext(task_type='probabilities', label_set=['LEFT', 'RIGHT'],
                                          instruction=INSTRUCTION)

    @property
    def identifier(self): return self._identifier
    @property
    def region_layer_map(self): return {}
    @property
    def supported_modalities(self) -> Set[str]: return {'vision'}
    def start_task(self, *a, **k): pass
    def start_recording(self, *a, **k): pass
    def reset(self): pass

    def process(self, input_event) -> Any:
        if isinstance(input_event, (StateChange, EnvironmentStep)):
            raise NotImplementedError
        stim = input_event
        chosen = []
        for _, row in stim.iterrows():
            side = self._choose(row)                       # 'LEFT' or 'RIGHT'
            chosen.append(row['left_obj'] if side == 'LEFT' else row['right_obj'])
        coords = {
            'stimulus_id': ('presentation', stim['image_id'].values),
            'sample_obj': ('presentation', stim['sample_obj'].values),
            'dist_obj': ('presentation', stim['dist_obj'].values),
            'truth': ('presentation', stim['sample_obj'].values),
        }
        return BehavioralAssembly(np.array(chosen), coords=coords, dims=['presentation'])


# --------------------------------------------------------------------------- #
# Choosers per path
# --------------------------------------------------------------------------- #
def _parse_lr(text, rng):
    t = text.upper()
    # prefer the last explicit "ANSWER: X"; else last bare LEFT/RIGHT token
    for key in ('ANSWER: LEFT', 'ANSWER:LEFT'):
        if key in t and t.rfind(key) >= t.rfind('ANSWER: RIGHT'):
            return 'LEFT'
    iL, iR = t.rfind('LEFT'), t.rfind('RIGHT')
    if iL < 0 and iR < 0:
        return 'LEFT' if rng.rand() < 0.5 else 'RIGHT'     # unparseable -> coin flip (chance)
    return 'LEFT' if iL > iR else 'RIGHT'


def build_generation_chooser(model_id):
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLM
    except Exception:
        from transformers import AutoModelForVision2Seq as VLM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    proc = AutoProcessor.from_pretrained(model_id)
    model = VLM.from_pretrained(model_id, torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                                device_map=device).eval()
    rng = np.random.RandomState(0)
    stats = {'calls': 0, 'parse_miss': 0}

    def choose(row):
        stats['calls'] += 1
        img = Image.open(row['image_path']).convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': INSTRUCTION}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        ans = proc.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        if 'LEFT' not in ans.upper() and 'RIGHT' not in ans.upper():
            stats['parse_miss'] += 1
        return _parse_lr(ans, rng)

    return choose, stats


def build_similarity_chooser(model_id):
    """Vision-only 2-AFC: pick the token nearer the sample in feature space."""
    import torch
    from PIL import Image
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import open_clip  # CLIP family
    model, _, preprocess = open_clip.create_model_and_transforms(model_id)
    model = model.to(device).eval()

    def embed(path):
        img = preprocess(Image.open(path).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model.encode_image(img)
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().ravel()

    def choose(row):
        s = embed(row['sample_path']); l = embed(row['left_path']); r = embed(row['right_path'])
        return 'LEFT' if float(s @ l) >= float(s @ r) else 'RIGHT'

    return choose, {'calls': 0}


def build_random_chooser(seed=0):
    rng = np.random.RandomState(seed)
    return (lambda row: 'LEFT' if rng.rand() < 0.5 else 'RIGHT'), {'calls': 0}


CHOOSERS = {'generation': build_generation_chooser, 'similarity': build_similarity_chooser,
            'random': lambda *_: build_random_chooser()}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True, choices=list(CHOOSERS))
    ap.add_argument('--model', default='')
    ap.add_argument('--n_images', type=int, default=240)
    ap.add_argument('--out', required=True)
    ap.add_argument('--montage_dir', default='')
    ap.add_argument('--witness_panels', type=int, default=12)
    ap.add_argument('--no_score', action='store_true',
                    help='skip i1/i2n scoring (mechanics smoke; i2 needs full object coverage)')
    args = ap.parse_args()

    from brainscore.witness import Witness
    os.makedirs(args.out, exist_ok=True)
    montage_dir = args.montage_dir or os.path.join(args.out, 'montages')

    trials = B.load_trials()
    stim = compose_montages(trials, args.n_images, montage_dir)
    print(f'composed {len(stim)} montage trials', flush=True)

    choose, stats = (CHOOSERS[args.path](args.model) if args.path != 'random'
                     else CHOOSERS[args.path]())
    model = TwoAFCModel(f'{args.path}:{args.model or "null"}', choose, mode=args.path)

    with Witness(model, label=f'{args.model or args.path} · Rajalingham 2-AFC') as w:
        assembly = model.process(stim)

    choices = pd.DataFrame({
        'image_id': assembly['stimulus_id'].values, 'sample_obj': assembly['sample_obj'].values,
        'dist_obj': assembly['dist_obj'].values, 'choice': assembly.values})
    acc = float((choices['choice'] == choices['sample_obj']).mean())
    scores = None if args.no_score else B.score_all(choices, metrics=('i1', 'i2n'))

    witness_out = w.save(os.path.join(args.out, 'witness'), max_panels=args.witness_panels)
    result = {'path': args.path, 'model': args.model, 'n_trials': len(choices),
              'accuracy': acc, 'stats': stats, 'scores': scores, 'witness': witness_out['summary']}
    with open(os.path.join(args.out, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    summary = {'path': args.path, 'model': args.model, 'accuracy': round(acc, 3)}
    if scores is not None:
        summary['i2n'], summary['i1'] = scores['i2n'], scores['i1']
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
