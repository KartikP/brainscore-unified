"""Validate the closed-loop embodied interface across a MODEL LADDER and produce
a scaling curve: bad model -> good model on the same grid video game.

Two tracks, both driven one tick at a time through ``process(EnvironmentStep)``:

  * visual VLMs read the rendered frame (perception + spatial reasoning).
  * a thinking text model reads the ASCII board (perfect perception; isolates
    reasoning).

References on the same boards: oracle (privileged-state upper bound) and random
(the null floor every model must clear to mean anything).

Policy builders + the episode runner now live in
``brainscore.model_helpers.local_vlm_policy`` — this file is just the ladder
driver. Models are loaded, run, and freed one at a time so the ladder fits a
single 24 GB GPU. Run on EC2 (GPU). Writes a scaling-curve-ready JSON to --out.
"""
import argparse
import gc
import json
import sys
import traceback

from brainscore.harnesses.grid_game import (
    greedy_oracle_policy, random_action_policy,
)
from brainscore.model_helpers.local_vlm_policy import (
    build_visual_policy, build_thinking_policy, run_grid_episodes,
)

# The ladder. (display_name, hf_id, mode, thinking). Ordered worse -> better.
DEFAULT_LADDER = [
    ('DeepSeek-R1-7B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'ascii', True),
]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=15)
    ap.add_argument('--size', type=int, default=5)
    ap.add_argument('--max-steps', type=int, default=20)
    ap.add_argument('--base-seed', type=int, default=500)
    ap.add_argument('--out', default='/tmp/vlm_game_scaling.json')
    ap.add_argument('--purge', action='store_true',
                    help='remove each large model from disk after running')
    args = ap.parse_args()
    common = dict(n_episodes=args.episodes, size=args.size,
                  max_steps=args.max_steps, base_seed=args.base_seed)

    results = {'config': vars(args), 'models': {}}

    print("[oracle]", flush=True)
    results['oracle'] = run_grid_episodes(greedy_oracle_policy, **common)
    print(f"  {results['oracle']}", flush=True)
    print("[random null]", flush=True)
    results['random_null'] = run_grid_episodes(random_action_policy(seed=0), **common)
    print(f"  {results['random_null']}", flush=True)

    for name, hf_id, mode, thinking in DEFAULT_LADDER:
        print(f"\n[{name}] mode={mode} thinking={thinking} loading ...", flush=True)
        try:
            if mode == 'visual':
                policy, stats = build_visual_policy(hf_id)
            else:
                policy, stats = build_thinking_policy(hf_id)
            res = run_grid_episodes(policy, **common)
            res['parse_miss'] = stats['parse_miss']
            res['calls'] = stats['calls']
            miss_rate = stats['parse_miss'] / max(1, stats['calls'])
            # >50% unparseable -> not instruction-following; its success rate is
            # not a competence signal (moves are mostly random fallbacks).
            res['instruction_following'] = miss_rate < 0.5
            res['parse_miss_rate'] = round(miss_rate, 3)
            res['mode'] = mode
            res['thinking'] = thinking
            results['models'][name] = res
            print(f"  {name}: {res}", flush=True)
            del policy  # drops the only ref to the loaded model
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
