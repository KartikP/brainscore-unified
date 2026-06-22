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

from brainscore.model_helpers.local_vlm_policy import (
    build_gemma_visual_policy, run_grid_episodes,
)

# gemma-4-12B-it needs a pinned snapshot for the 4-bit skip-module config.
_GEMMA_4_12B_REVISION = 'e18f459f54832f4ae2ab6686b935a2268668a9e9'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='google/gemma-4-12B-it')
    ap.add_argument('--episodes', type=int, default=15)
    ap.add_argument('--size', type=int, default=5)
    ap.add_argument('--max-steps', type=int, default=20)
    ap.add_argument('--base-seed', type=int, default=500)
    ap.add_argument('--out', default='/tmp/gemma_grid_cot.json')
    args = ap.parse_args()

    revision = (_GEMMA_4_12B_REVISION
                if args.model == 'google/gemma-4-12B-it' else None)
    policy, stats = build_gemma_visual_policy(args.model, revision=revision)
    res = run_grid_episodes(policy, n_episodes=args.episodes, size=args.size,
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
