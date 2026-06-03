"""Validate the closed-loop embodied interface across a MODEL LADDER and produce
a scaling curve: bad model -> good model on the same grid video game.

Two tracks, both driven one tick at a time through ``process(EnvironmentStep)``:

  * visual VLMs read the rendered frame (perception + spatial reasoning):
      Qwen2.5-VL-3B  ->  Qwen2.5-VL-7B
  * a thinking text model reads the ASCII board (perfect perception; isolates
    reasoning): Qwen3-8B with thinking enabled.

References on the same boards:
    oracle  : privileged-state upper bound (optimal on wall-free grids)
    random  : the null floor — every model must clear this to mean anything.

Models are loaded, run, and freed one at a time so the whole ladder fits a
single 24 GB GPU. Honest by construction: if a model is below the random floor,
we report it (the 3B VLM is — abstract-grid vision is hard for small VLMs).

Run on EC2 (GPU). Writes a scaling-curve-ready JSON to --out.
"""
import argparse
import gc
import json
import re
import sys
import traceback

import numpy as np
from PIL import Image

from brainscore_core.model_interface import BrainScoreModel
from brainscore.model_helpers.policy_wrapper import PolicyWrapper
from brainscore.harnesses.grid_game import (
    GridGameEnv, ACTIONS, greedy_oracle_policy, random_action_policy, play_game,
)

_WORD_TO_ACTION = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

# The ladder. (display_name, hf_id, mode, thinking). Ordered worse -> better
# within each track; the driver runs them in this order.
DEFAULT_LADDER = [
    ('Qwen2.5-VL-3B', 'Qwen/Qwen2.5-VL-3B-Instruct', 'visual', False),
    ('Qwen2.5-VL-7B', 'Qwen/Qwen2.5-VL-7B-Instruct', 'visual', False),
    ('Qwen3-8B-think', 'Qwen/Qwen3-8B', 'ascii', True),
]

_VISUAL_PROMPT = (
    "This is a grid game. A BLUE square is the player and a GREEN square is the "
    "goal. You move the BLUE square one cell at a time.\n"
    "Choose the single move that brings the BLUE square closer to the GREEN "
    "square:\n- 'up' toward the top\n- 'down' toward the bottom\n- 'left'\n- "
    "'right'\nAnswer with exactly one word: up, down, left, or right."
)

_ASCII_PROMPT = (
    "You are playing a grid game. The board uses: P = player, G = goal, "
    "# = wall, . = empty. Row 0 is the top. 'up' decreases the row, 'down' "
    "increases it, 'left' decreases the column, 'right' increases it.\n\n"
    "Board:\n{board}\n\n"
    "Think step by step about which single move brings P closer to G, then end "
    "your reply with a line exactly like:\nAction: <up|down|left|right>"
)


def _parse_action(text: str, prefer_last: bool = False) -> int:
    """Extract an action from model output. Prefer an explicit 'Action: word';
    else first (or last) direction word. Returns -1 if none found."""
    m = re.search(r'action\s*[:\-]\s*(up|down|left|right)', text, re.IGNORECASE)
    if m:
        return _WORD_TO_ACTION[m.group(1).lower()]
    words = re.findall(r'\b(up|down|left|right)\b', text, re.IGNORECASE)
    if words:
        return _WORD_TO_ACTION[(words[-1] if prefer_last else words[0]).lower()]
    return -1


def build_visual_policy(model_id):
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLM
    except Exception:
        from transformers import AutoModelForVision2Seq as VLM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(model_id)
    model = VLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map=device).eval()
    stats = {'parse_miss': 0, 'calls': 0}

    def policy(observation, history):
        stats['calls'] += 1
        img = Image.fromarray(observation['frame']).resize((320, 320), Image.NEAREST)
        messages = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': _VISUAL_PROMPT}]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        gen = out[0][inputs['input_ids'].shape[1]:]
        ans = processor.decode(gen, skip_special_tokens=True)
        a = _parse_action(ans)
        if a < 0:
            stats['parse_miss'] += 1
            return 3
        return a

    return policy, stats, (model, processor)


def build_thinking_policy(model_id):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map=device).eval()
    stats = {'parse_miss': 0, 'calls': 0}

    def policy(observation, history):
        stats['calls'] += 1
        prompt = _ASCII_PROMPT.format(board=observation['ascii'])
        messages = [{'role': 'user', 'content': prompt}]
        try:
            text = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=True)
        except TypeError:
            text = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
        inputs = tok([text], return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        gen = out[0][inputs['input_ids'].shape[1]:]
        ans = tok.decode(gen, skip_special_tokens=True)
        a = _parse_action(ans, prefer_last=True)
        if a < 0:
            stats['parse_miss'] += 1
            return 3
        return a

    return policy, stats, (model, tok)


def _purge_hf_cache(hf_id):
    """Remove a model's HuggingFace hub snapshot to bound disk use when running
    a ladder of large models on a small volume."""
    import os
    import shutil
    cache = os.path.expanduser('~/.cache/huggingface/hub')
    name = 'models--' + hf_id.replace('/', '--')
    path = os.path.join(cache, name)
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        print(f"  purged cache {path}", flush=True)


def make_model(policy):
    return BrainScoreModel(
        identifier='grid-player', model=None, region_layer_map={},
        preprocessors={}, activations_model=None,
        action_fn=PolicyWrapper(policy, max_history=4))


def run_policy(policy, n_episodes, size, max_steps, base_seed):
    solves, effs = [], []
    for i in range(n_episodes):
        env = GridGameEnv(size=size, seed=base_seed + i, max_steps=max_steps)
        res = play_game(make_model(policy), env)
        solves.append(1.0 if res['solved'] else 0.0)
        if res['solved']:
            effs.append(res['efficiency'])
    return {'success_rate': float(np.mean(solves)),
            'mean_efficiency': float(np.mean(effs)) if effs else 0.0,
            'n_episodes': n_episodes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=15)
    ap.add_argument('--size', type=int, default=5)
    ap.add_argument('--max-steps', type=int, default=20)
    ap.add_argument('--base-seed', type=int, default=500)
    ap.add_argument('--out', default='/tmp/vlm_game_scaling.json')
    ap.add_argument('--purge', action='store_true',
                    help='remove each large model from disk after running (bounds disk use)')
    args = ap.parse_args()
    common = dict(n_episodes=args.episodes, size=args.size,
                  max_steps=args.max_steps, base_seed=args.base_seed)

    results = {'config': vars(args), 'models': {}}

    print("[oracle]", flush=True)
    results['oracle'] = run_policy(greedy_oracle_policy, **common)
    print(f"  {results['oracle']}", flush=True)
    print("[random null]", flush=True)
    results['random_null'] = run_policy(random_action_policy(seed=0), **common)
    print(f"  {results['random_null']}", flush=True)

    for name, hf_id, mode, thinking in DEFAULT_LADDER:
        print(f"\n[{name}] mode={mode} thinking={thinking} loading ...", flush=True)
        try:
            if mode == 'visual':
                policy, stats, handles = build_visual_policy(hf_id)
            else:
                policy, stats, handles = build_thinking_policy(hf_id)
            res = run_policy(policy, **common)
            res['parse_miss'] = stats['parse_miss']
            res['calls'] = stats['calls']
            res['mode'] = mode
            res['thinking'] = thinking
            results['models'][name] = res
            print(f"  {name}: {res}", flush=True)
            # free GPU before the next model
            del policy, handles
        except Exception as e:
            results['models'][name] = {'error': str(e)}
            print(f"  {name} FAILED: {e}", flush=True)
            traceback.print_exc()
        finally:
            try:
                import torch
                gc.collect(); torch.cuda.empty_cache()
            except Exception:
                pass
            if args.purge and name != 'Qwen2.5-VL-3B':
                _purge_hf_cache(hf_id)

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print("\n=== SCALING (success rate) ===")
    print(f"random null : {results['random_null']['success_rate']:.2f}")
    for name, r in results['models'].items():
        sr = r.get('success_rate')
        print(f"{name:16s}: {sr if sr is None else f'{sr:.2f}'}"
              + (f"  ({r.get('mode')},think={r.get('thinking')})" if 'mode' in r else ''))
    print(f"oracle      : {results['oracle']['success_rate']:.2f}")
    print(f"written: {args.out}")


if __name__ == '__main__':
    sys.exit(main())
