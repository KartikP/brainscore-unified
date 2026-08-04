"""Sequential (faithful-MTS) Rajalingham2018 2-AFC.

The simultaneous version (score_2afc.py + compose_montage) shows the SAMPLE and
the two choice tokens on ONE screen at once. The real match-to-sample task is
temporal: the test image is flashed briefly, REMOVED, and only THEN do the two
choice tokens appear — sample and choices are never co-visible. This module is
that variant. It leaves the simultaneous code untouched (kept until we have
results to compare).

Two presentation steps per trial:
  step 1  render_sample(...)        — the sample shown alone
  step 2  compose_choice_array(...) — the two tokens, sample absent

Two faithfulness modes for how the sample is "removed" between steps:
  describe : turn 1 the model describes the sample in words; turn 2 it sees ONLY
             that description + the choice array — the sample pixels are gone.
             A true bottleneck: the decision rides on the model's own memory.
  recall   : sample shown in turn 1, choices in turn 2, but the sample image
             stays in the conversation context (still attendable). An UPPER
             BOUND, not a true removal — labelled as such.

The per-trial choice -> chosen object -> benchmark.score_all (i1/i2n), identical
to the simultaneous path, so the two are directly comparable. Run generation on
EC2 (GPU); the random null runs anywhere.

    python -m brainscore.benchmarks.rajalingham2018_2afc.sequential \
        --path generation --model Qwen/Qwen2.5-VL-7B-Instruct \
        --seq_mode describe --prompt_mode direct --n_images 120 --out /tmp/raj2afc_seq_qwen7b
"""
import argparse
import json
import os
from typing import Any, Callable, Set

import numpy as np
import pandas as pd

from brainscore_core.model_interface import (
    EnvironmentStep, StateChange, TaskContext, UnifiedModel)
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly

from . import benchmark as B
from .montage import render_sample, compose_choice_array
from .score_2afc import _parse_lr


# --------------------------------------------------------------------------- #
# Presentation: two images per trial (sample alone, then the choice array)
# --------------------------------------------------------------------------- #
def compose_sequential_trials(trials, n_images, out_dir, seed=0, balance_sides=False):
    """Per trial, write sample_<tag>.png (the brief test image, alone) and
    choices_<tag>.png (the two tokens, sample absent). Returns a stim DataFrame
    with both render paths + the same trial metadata the scorer expects."""
    from brainscore_vision import load_stimulus_set
    ss = load_stimulus_set('objectome.public')
    path_of = lambda iid: str(ss.get_stimulus(iid))
    os.makedirs(out_dir, exist_ok=True)

    images = sorted(trials['image_id'].unique())
    rng = np.random.RandomState(seed)
    if n_images and n_images < len(images):
        images = sorted(rng.choice(images, size=n_images, replace=False).tolist())
    sub = trials[trials['image_id'].isin(images)].reset_index(drop=True)
    sides = [True, False] if balance_sides else None

    rows = []
    for i, t in sub.iterrows():
        sample_tok = path_of(t['token_sample_id'])
        dist_tok = path_of(t['token_dist_id'])
        these = sides if balance_sides else [bool(rng.rand() < 0.5)]
        for s in these:
            tag = f'{i:05d}' + ('L' if (balance_sides and s) else 'R' if balance_sides else '')
            left_path, right_path = (sample_tok, dist_tok) if s else (dist_tok, sample_tok)
            left_obj, right_obj = (t['sample_obj'], t['dist_obj']) if s else (t['dist_obj'], t['sample_obj'])
            sample_render = os.path.join(out_dir, f'sample_{tag}.png')
            choices_render = os.path.join(out_dir, f'choices_{tag}.png')
            if not os.path.exists(sample_render):
                render_sample(path_of(t['image_id'])).save(sample_render)
            if not os.path.exists(choices_render):
                compose_choice_array(left_path, right_path).save(choices_render)
            rows.append({
                'stimulus_id': f'trial_{tag}', 'sample_render_path': sample_render,
                'choices_path': choices_render,
                'image_id': t['image_id'], 'sample_obj': t['sample_obj'], 'dist_obj': t['dist_obj'],
                'left_obj': left_obj, 'right_obj': right_obj,
                'sample_path': path_of(t['image_id']), 'left_path': left_path, 'right_path': right_path,
            })
    stim = pd.DataFrame(rows)
    stim.to_csv(os.path.join(out_dir, 'manifest.csv'), index=False)
    return stim


# --------------------------------------------------------------------------- #
# A process()-bearing sequential 2-AFC model (Witness-recordable, same output
# shape as the simultaneous TwoAFCModel so scoring is identical)
# --------------------------------------------------------------------------- #
class SequentialTwoAFCModel(UnifiedModel):
    """Wraps a per-trial ``choose_seq(row) -> 'LEFT'|'RIGHT'`` (which internally
    presents sample-then-choices) into a UnifiedModel."""

    COLUMN_TO_MODALITY = {'choices_path': 'vision'}   # Witness shows the decision screen

    def __init__(self, identifier, choose_seq: Callable[[Any], str], seq_mode='describe',
                 instruction='Which option is the same object as the sample you saw? LEFT or RIGHT.'):
        self._identifier = identifier
        self._choose_seq = choose_seq
        self._witness_mode = 'generation'
        self._seq_mode = seq_mode
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
            side = self._choose_seq(row)                   # 'LEFT' | 'RIGHT'
            chosen.append(row['left_obj'] if side == 'LEFT' else row['right_obj'])
        coords = {
            'stimulus_id': ('presentation', stim['image_id'].values),
            'sample_obj': ('presentation', stim['sample_obj'].values),
            'dist_obj': ('presentation', stim['dist_obj'].values),
            'truth': ('presentation', stim['sample_obj'].values),
        }
        return BehavioralAssembly(np.array(chosen), coords=coords, dims=['presentation'])


# --------------------------------------------------------------------------- #
# Choosers
# --------------------------------------------------------------------------- #
def build_sequential_generation_chooser(model_id, seq_mode='describe', prompt_mode='direct',
                                        max_desc_tokens=40):
    """Two-turn VLM chooser. seq_mode='describe' truly removes the sample (turn 2
    sees only the model's own description + the choices); seq_mode='recall' keeps
    the sample image in context (attendable upper bound)."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLM
    except Exception:
        # AutoModelForVision2Seq was removed in transformers 5; the
        # ImageTextToText auto-class is its replacement and exists in 4.57 too.
        from transformers import AutoModelForImageTextToText as VLM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    proc = AutoProcessor.from_pretrained(model_id)
    model = VLM.from_pretrained(model_id, torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                                device_map=device).eval()
    rng = np.random.RandomState(0)
    stats = {'calls': 0, 'parse_miss': 0, 'seq_mode': seq_mode, 'prompt_mode': prompt_mode}
    answer_instr = ('Answer with exactly one word: LEFT or RIGHT.' if prompt_mode == 'direct'
                    else "Reason briefly, then end with a line 'Answer: LEFT' or 'Answer: RIGHT'.")
    max_ans = 6 if prompt_mode == 'direct' else 200

    def _gen(messages, images, max_new):
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=images, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return proc.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    def choose_seq(row):
        stats['calls'] += 1
        sample_img = Image.open(row['sample_render_path']).convert('RGB')
        choices_img = Image.open(row['choices_path']).convert('RGB')
        if seq_mode == 'describe':
            m1 = [{'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': 'Briefly describe this object — its type, shape, and '
                                         'distinctive features — so you can recognize it later.'}]}]
            desc = _gen(m1, [sample_img], max_desc_tokens).strip().replace('\n', ' ')
            # sample is GONE: turn 2 gets only the description + the choice array
            m2 = [{'role': 'user', 'content': [
                {'type': 'text', 'text': f'You just saw an object and described it as: "{desc}". '
                                         f'Below are two options (LEFT and RIGHT). '
                                         f'Which is the SAME object you saw? ' + answer_instr},
                {'type': 'image'}]}]
            ans = _gen(m2, [choices_img], max_ans)
            stats['last_desc'] = desc
        else:  # recall — sample retained in context (upper bound, not a true removal)
            m = [{'role': 'user', 'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': 'Study this SAMPLE object; you will then pick it from two options.'}]},
                 {'role': 'assistant', 'content': 'I have studied the object.'},
                 {'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Here are two options (LEFT and RIGHT). '
                                             'Which is the SAME object as the sample? ' + answer_instr},
                    {'type': 'image'}]}]
            ans = _gen(m, [sample_img, choices_img], max_ans)
        if 'LEFT' not in ans.upper() and 'RIGHT' not in ans.upper():
            stats['parse_miss'] += 1
        return _parse_lr(ans, rng)

    return choose_seq, stats


def build_sequential_random_chooser(seed=0):
    """Chance null for the sequential variant (ignores the images)."""
    rng = np.random.RandomState(seed)
    return (lambda row: 'LEFT' if rng.rand() < 0.5 else 'RIGHT'), {'calls': 0, 'seq_mode': 'random'}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default='generation', choices=['generation', 'random'])
    ap.add_argument('--model', default='')
    ap.add_argument('--seq_mode', default='describe', choices=['describe', 'recall'])
    ap.add_argument('--prompt_mode', default='direct', choices=['cot', 'direct'])
    ap.add_argument('--balance_sides', action='store_true')
    ap.add_argument('--n_images', type=int, default=120)
    ap.add_argument('--out', required=True)
    ap.add_argument('--render_dir', default='')
    ap.add_argument('--witness_panels', type=int, default=12)
    ap.add_argument('--no_score', action='store_true')
    args = ap.parse_args()

    from brainscore.witness import Witness
    os.makedirs(args.out, exist_ok=True)
    render_dir = args.render_dir or os.path.join(args.out, 'renders')

    trials = B.load_trials()
    stim = compose_sequential_trials(trials, args.n_images, render_dir, balance_sides=args.balance_sides)
    print(f'composed {len(stim)} sequential trials (seq_mode={args.seq_mode}, '
          f'balance_sides={args.balance_sides})', flush=True)

    if args.path == 'generation':
        choose_seq, stats = build_sequential_generation_chooser(args.model, args.seq_mode, args.prompt_mode)
    else:
        choose_seq, stats = build_sequential_random_chooser()
    tag = f'sequential:{args.path}:{args.model or "null"}:{args.seq_mode}:{args.prompt_mode}'
    model = SequentialTwoAFCModel(tag, choose_seq, seq_mode=args.seq_mode)

    with Witness(model, label=f'{args.model or args.path} · Rajalingham 2-AFC (sequential/{args.seq_mode})') as w:
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
    frac_left = float((choices['side'] == 'LEFT').mean())
    scores = None if args.no_score else B.score_all(
        choices[['image_id', 'sample_obj', 'dist_obj', 'choice']], metrics=('i1', 'i2n'))

    witness_out = w.save(os.path.join(args.out, 'witness'), max_panels=args.witness_panels)
    result = {'variant': 'sequential', 'seq_mode': args.seq_mode, 'path': args.path,
              'model': args.model, 'prompt_mode': args.prompt_mode, 'balance_sides': args.balance_sides,
              'n_trials': len(choices), 'accuracy': acc, 'frac_left': frac_left,
              'stats': stats, 'scores': scores, 'witness': witness_out['summary']}
    with open(os.path.join(args.out, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    summary = {'variant': 'sequential', 'seq_mode': args.seq_mode, 'model': args.model,
               'accuracy': round(acc, 3), 'frac_left': round(frac_left, 3)}
    if scores is not None:
        summary['i2n'], summary['i1'] = scores['i2n'], scores['i1']
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
