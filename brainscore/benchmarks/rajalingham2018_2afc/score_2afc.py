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

# Two prompt modes, to separate perception from reasoning. Match-to-sample is a
# perceptual task: chain-of-thought may HURT a small model (it talks itself away
# from its first-glance percept), unlike the planning-heavy grid game where CoT helped.
INSTRUCTION_COT = ("The image shows a SAMPLE object on top and two options below (LEFT and RIGHT). "
                   "Exactly one option is the same object as the sample. "
                   "Reason briefly, then end with a line: 'Answer: LEFT' or 'Answer: RIGHT'.")
INSTRUCTION_DIRECT = ("The image shows a SAMPLE object on top and two options below (LEFT and RIGHT). "
                      "Which option is the SAME object as the sample? "
                      "Answer with exactly one word: LEFT or RIGHT.")
INSTRUCTION = INSTRUCTION_COT       # default; overridden per run via --prompt_mode


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
    stim = pd.DataFrame(rows)
    # persist a manifest so a standalone chooser (e.g. a model in a different
    # conda env, with no brainscore) can run on the identical trials + score later.
    stim.to_csv(os.path.join(out_dir, 'manifest.csv'), index=False)
    return stim


# --------------------------------------------------------------------------- #
# A process()-bearing 2-AFC model (so the run is Witness-recordable)
# --------------------------------------------------------------------------- #
class TwoAFCModel(UnifiedModel):
    """Wraps a per-trial ``choose(row) -> 'LEFT'|'RIGHT'`` into a UnifiedModel
    whose ``process(stim_set)`` returns the chosen objects as a BehavioralAssembly."""

    COLUMN_TO_MODALITY = {'image_path': 'vision'}   # so Witness renders the montage

    def __init__(self, identifier, choose: Callable[[Any], str], mode='generation', instruction=INSTRUCTION_COT):
        self._identifier = identifier
        self._choose = choose
        self._witness_mode = mode      # how the Witness should label this run
        self._task_context = TaskContext(task_type='probabilities', label_set=['LEFT', 'RIGHT'],
                                          instruction=instruction)

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


def select_demos(stim, k):
    """k balanced solved practice trials (correct side known) from the stim set,
    returned with the demo image_ids so they can be held out of the test set."""
    from PIL import Image
    left, right, seen = [], [], set()
    for _, r in stim.iterrows():
        if r['image_id'] in seen:
            continue
        correct = 'LEFT' if r['sample_obj'] == r['left_obj'] else 'RIGHT'
        bucket = left if correct == 'LEFT' else right
        if len(bucket) < k // 2:
            bucket.append({'img': Image.open(r['image_path']).convert('RGB'), 'answer': correct})
            seen.add(r['image_id'])
        if len(left) >= k // 2 and len(right) >= k // 2:
            break
    demos = [x for pair in zip(left, right) for x in pair]   # alternate sides
    return demos, seen


def build_generation_chooser(model_id, prompt_mode='cot', demos=None):
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
    demos = demos or []
    stats = {'calls': 0, 'parse_miss': 0, 'prompt_mode': prompt_mode, 'n_shots': len(demos)}
    instr = INSTRUCTION_DIRECT if prompt_mode == 'direct' else INSTRUCTION_COT
    max_new = 6 if prompt_mode == 'direct' else 200    # direct = first-glance percept; cot = room to reason

    def choose(row):
        stats['calls'] += 1
        img = Image.open(row['image_path']).convert('RGB')
        messages = []
        for d in demos:                                  # in-context practice trials
            messages.append({'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': instr}]})
            messages.append({'role': 'assistant', 'content': f'Answer: {d["answer"]}'})
        messages.append({'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': instr}]})
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [d['img'] for d in demos] + [img]
        inputs = proc(text=[text], images=images, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        ans = proc.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        if 'LEFT' not in ans.upper() and 'RIGHT' not in ans.upper():
            stats['parse_miss'] += 1
        return _parse_lr(ans, rng)

    return choose, stats, instr


def build_similarity_chooser(model_id):
    """Vision-only 2-AFC: pick the token whose image embedding is nearest the
    sample's, in a CLIP-family feature space (no trained classifier — pure
    zero-shot feature matching). Uses HuggingFace CLIP."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mid = model_id or 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(mid).to(device).eval()
    proc = CLIPProcessor.from_pretrained(mid)
    cache = {}

    def embed(path):
        if path in cache:
            return cache[path]
        img = Image.open(path).convert('RGB')
        inputs = proc(images=img, return_tensors='pt').to(device)
        with torch.no_grad():
            f = model.get_image_features(**inputs)
        f = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy().ravel()
        cache[path] = f
        return f

    def choose(row):
        s, l, r = embed(row['sample_path']), embed(row['left_path']), embed(row['right_path'])
        return 'LEFT' if float(s @ l) >= float(s @ r) else 'RIGHT'

    return choose, {'calls': 0}


def build_random_chooser(seed=0):
    rng = np.random.RandomState(seed)
    return (lambda row: 'LEFT' if rng.rand() < 0.5 else 'RIGHT'), {'calls': 0}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True, choices=['generation', 'similarity', 'random'])
    ap.add_argument('--model', default='')
    ap.add_argument('--prompt_mode', default='cot', choices=['cot', 'direct'],
                    help='generation only: cot = reason-then-answer; direct = one-word first-glance answer')
    ap.add_argument('--n_shots', type=int, default=0,
                    help='generation only: in-context practice trials prepended per prompt (0 = zero-shot)')
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

    demos = []
    if args.path == 'generation' and args.n_shots:
        demos, demo_imgs = select_demos(stim, args.n_shots)
        stim = stim[~stim['image_id'].isin(demo_imgs)].reset_index(drop=True)   # no leakage
        print(f'{len(demos)} practice demos held out ({[d["answer"] for d in demos]}); '
              f'{len(stim)} test trials', flush=True)

    if args.path == 'generation':
        choose, stats, instr = build_generation_chooser(args.model, args.prompt_mode, demos=demos)
    elif args.path == 'similarity':
        choose, stats = build_similarity_chooser(args.model); instr = INSTRUCTION_DIRECT
    else:
        choose, stats = build_random_chooser(); instr = INSTRUCTION_DIRECT
    tag = f'{args.path}:{args.model or "null"}' + (
        f':{args.prompt_mode}:{args.n_shots}shot' if args.path == 'generation' else '')
    model = TwoAFCModel(tag, choose, mode=args.path, instruction=instr)

    with Witness(model, label=f'{args.model or args.path} · Rajalingham 2-AFC') as w:
        assembly = model.process(stim)

    chosen = assembly.values
    choices = pd.DataFrame({
        'image_id': assembly['stimulus_id'].values, 'sample_obj': assembly['sample_obj'].values,
        'dist_obj': assembly['dist_obj'].values, 'choice': chosen,
        'left_obj': stim['left_obj'].values, 'right_obj': stim['right_obj'].values})
    choices['side'] = np.where(choices['choice'] == choices['left_obj'], 'LEFT', 'RIGHT')
    choices['correct'] = choices['choice'] == choices['sample_obj']
    choices.to_csv(os.path.join(args.out, 'choices.csv'), index=False)
    acc = float(choices['correct'].mean())
    frac_left = float((choices['side'] == 'LEFT').mean())          # side-bias diagnostic
    per_obj_acc = choices.groupby('sample_obj')['correct'].mean().round(3).to_dict()
    scores = None if args.no_score else B.score_all(choices[['image_id', 'sample_obj', 'dist_obj', 'choice']],
                                                     metrics=('i1', 'i2n'))

    witness_out = w.save(os.path.join(args.out, 'witness'), max_panels=args.witness_panels)
    result = {'path': args.path, 'model': args.model,
              'prompt_mode': args.prompt_mode if args.path == 'generation' else None,
              'n_shots': args.n_shots if args.path == 'generation' else 0,
              'n_trials': len(choices), 'accuracy': acc, 'frac_left': frac_left,
              'per_object_accuracy': per_obj_acc,
              'stats': stats, 'scores': scores, 'witness': witness_out['summary']}
    with open(os.path.join(args.out, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    summary = {'path': args.path, 'model': args.model, 'accuracy': round(acc, 3),
               'frac_left': round(frac_left, 3)}
    if scores is not None:
        summary['i2n'], summary['i1'] = scores['i2n'], scores['i1']
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
