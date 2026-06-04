"""Run Gemma-4-12B (4-bit, chain-of-thought) on the SAME toy GridGameEnv as the
embodied scaling curve — so it can be placed fairly next to Qwen-VL-7B (CoT)=0.53.

The curve's points (oracle, random, Qwen-VL-3B/7B CoT, DeepSeek-R1 ASCII) all come
from play_vlm_game.py on this toy GridGameEnv (size=5, max_steps=20, base_seed=500,
15 episodes, the step-by-step _VISUAL_PROMPT). Gemma's earlier 0.00 was on the
HARDER MiniGrid-DoorKey (minigrid_vlm.py) — a different game. This reruns Gemma on
the toy game with the identical visual CoT prompt, reusing every shared helper.

Run in the gemma4 conda env (4-bit needs its transformers/bitsandbytes pins).
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified/scripts/vlm_game')

from brainscore.harnesses.grid_game import ACTIONS
# reuse the toy-game CoT prompt + parser + episode driver from the curve's runner
from play_vlm_game import _VISUAL_PROMPT, _parse_action, run_policy


def build_gemma_visual_policy(model_id):
    """Gemma-4-12B 4-bit visual policy — same recipe as minigrid_vlm.build_gemma_policy,
    but on the toy game's rendered frame + the curve's step-by-step _VISUAL_PROMPT."""
    import torch
    from PIL import Image
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              BitsAndBytesConfig)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             llm_int8_skip_modules=['patch_dense', 'embedding_projection', 'lm_head'])
    rev = 'e18f459f54832f4ae2ab6686b935a2268668a9e9' if model_id == 'google/gemma-4-12B-it' else None
    proc = AutoProcessor.from_pretrained(model_id, revision=rev)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=rev, quantization_config=bnb, device_map='auto',
        dtype=torch.bfloat16).eval()
    dev = next(model.parameters()).device
    stats = {'parse_miss': 0, 'calls': 0}
    rng = np.random.RandomState(0)

    def policy(observation, history):
        stats['calls'] += 1
        img = Image.fromarray(observation['frame']).resize((320, 320), Image.NEAREST)
        msgs = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img},
            {'type': 'text', 'text': _VISUAL_PROMPT}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors='pt').to(dev)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=200, do_sample=False)
        ans = proc.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
        a = _parse_action(ans, prefer_last=True)
        if a < 0:
            stats['parse_miss'] += 1
            return int(rng.randint(0, len(ACTIONS)))
        return a

    return policy, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='google/gemma-4-12B-it')
    ap.add_argument('--episodes', type=int, default=15)
    ap.add_argument('--size', type=int, default=5)
    ap.add_argument('--max-steps', type=int, default=20)
    ap.add_argument('--base-seed', type=int, default=500)
    ap.add_argument('--out', default='/tmp/gemma_grid_cot.json')
    args = ap.parse_args()

    policy, stats = build_gemma_visual_policy(args.model)
    res = run_policy(policy, n_episodes=args.episodes, size=args.size,
                     max_steps=args.max_steps, base_seed=args.base_seed)
    res['parse_miss'] = stats['parse_miss']
    res['calls'] = stats['calls']
    res['parse_miss_rate'] = round(stats['parse_miss'] / max(1, stats['calls']), 3)
    res['instruction_following'] = res['parse_miss_rate'] < 0.5
    res['mode'] = 'visual'
    res['model'] = args.model
    json.dump(res, open(args.out, 'w'), indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == '__main__':
    main()
