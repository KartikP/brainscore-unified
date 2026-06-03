"""Validate the closed-loop embodied interface with a REAL small VLM.

A 3B vision-language model (Qwen2.5-VL-3B-Instruct) is the policy: it looks at
the rendered game frame and replies with one word (up/down/left/right). The same
``play_game`` driver that an oracle or a random null uses drives the VLM, one
``process(EnvironmentStep)`` per tick. We report all three on identical board
seeds so the comparison is apples-to-apples:

    oracle  : upper reference (privileged state, optimal on wall-free boards)
    VLM     : the thing under test — visual spatial reasoning, image only
    random  : the null floor (must be beaten for the VLM result to mean anything)

Honest by construction: if the VLM does not clear the random floor, we say so.
The point of this script is to prove the interface carries a real VLM end to
end and to measure where it lands between floor and ceiling — not to claim the
VLM is good at grids.

Run on EC2 (GPU). Writes results JSON to --out.
"""
import argparse
import json
import re
import sys

import numpy as np
from PIL import Image

# brainscore unified
from brainscore_core.model_interface import BrainScoreModel
from brainscore.model_helpers.policy_wrapper import PolicyWrapper
from brainscore.harnesses.grid_game import (
    GridGameEnv, ACTIONS, greedy_oracle_policy, random_action_policy, play_game,
)

_WORD_TO_ACTION = {'up': 0, 'down': 1, 'left': 2, 'right': 3}


def build_vlm_policy(model_id='Qwen/Qwen2.5-VL-3B-Instruct'):
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
        device_map=device)
    model.eval()

    stats = {'parse_miss': 0, 'calls': 0}

    prompt = (
        "This is a grid game. A BLUE square is the player and a GREEN square is "
        "the goal. You move the BLUE square one cell at a time.\n"
        "Choose the single move that brings the BLUE square closer to the GREEN "
        "square:\n"
        "- 'up' moves it one cell toward the top\n"
        "- 'down' moves it one cell toward the bottom\n"
        "- 'left' moves it one cell to the left\n"
        "- 'right' moves it one cell to the right\n"
        "Answer with exactly one word: up, down, left, or right."
    )

    def policy(observation, history):
        stats['calls'] += 1
        frame = observation['frame']
        # upscale for legibility
        img = Image.fromarray(frame).resize((320, 320), Image.NEAREST)
        messages = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        gen = out[0][inputs['input_ids'].shape[1]:]
        answer = processor.decode(gen, skip_special_tokens=True).strip().lower()
        m = re.search(r'\b(up|down|left|right)\b', answer)
        if m:
            return _WORD_TO_ACTION[m.group(1)]
        stats['parse_miss'] += 1
        return 3  # deterministic fallback so a parse miss isn't a free random move

    return policy, stats


def make_model(policy):
    return BrainScoreModel(
        identifier='grid-player', model=None, region_layer_map={},
        preprocessors={}, activations_model=None,
        action_fn=PolicyWrapper(policy, max_history=4))


def run_policy(policy, n_episodes, size, max_steps, base_seed):
    solves, effs, steps = [], [], []
    for i in range(n_episodes):
        env = GridGameEnv(size=size, seed=base_seed + i, max_steps=max_steps)
        res = play_game(make_model(policy), env)
        solves.append(1.0 if res['solved'] else 0.0)
        if res['solved']:
            effs.append(res['efficiency'])
            steps.append(res['steps'])
    return {
        'success_rate': float(np.mean(solves)),
        'mean_efficiency': float(np.mean(effs)) if effs else 0.0,
        'mean_steps_to_solve': float(np.mean(steps)) if steps else None,
        'n_episodes': n_episodes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=15)
    ap.add_argument('--size', type=int, default=5)
    ap.add_argument('--max-steps', type=int, default=20)
    ap.add_argument('--base-seed', type=int, default=500)
    ap.add_argument('--out', default='/tmp/vlm_game_results.json')
    ap.add_argument('--model-id', default='Qwen/Qwen2.5-VL-3B-Instruct')
    args = ap.parse_args()

    common = dict(n_episodes=args.episodes, size=args.size,
                  max_steps=args.max_steps, base_seed=args.base_seed)

    print(f"[oracle] running {args.episodes} episodes ...", flush=True)
    oracle = run_policy(greedy_oracle_policy, **common)
    print(f"  oracle: {oracle}", flush=True)

    print(f"[random null] running ...", flush=True)
    rnd = run_policy(random_action_policy(seed=0), **common)
    print(f"  random: {rnd}", flush=True)

    print(f"[VLM {args.model_id}] loading + running (slow) ...", flush=True)
    vlm_policy, stats = build_vlm_policy(args.model_id)
    vlm = run_policy(vlm_policy, **common)
    print(f"  vlm: {vlm}  (parse_miss={stats['parse_miss']}/{stats['calls']})",
          flush=True)

    results = {
        'config': vars(args),
        'oracle': oracle, 'random_null': rnd, 'vlm': vlm,
        'vlm_parse_stats': stats,
        'verdict': ('VLM beats random null' if vlm['success_rate'] > rnd['success_rate']
                    else 'VLM at/below random floor — interface works, model weak on grids'),
    }
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print("\n=== SUMMARY ===")
    print(f"oracle  success={oracle['success_rate']:.2f} eff={oracle['mean_efficiency']:.2f}")
    print(f"VLM     success={vlm['success_rate']:.2f} eff={vlm['mean_efficiency']:.2f}")
    print(f"random  success={rnd['success_rate']:.2f} eff={rnd['mean_efficiency']:.2f}")
    print(f"verdict: {results['verdict']}")
    print(f"written: {args.out}")


if __name__ == '__main__':
    sys.exit(main())
