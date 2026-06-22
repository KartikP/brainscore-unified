"""Drive a complex Gymnasium game (MiniGrid) with a VLM policy, through the unified
``process(EnvironmentStep)`` interface — the same closed loop as the toy grid game,
now on a standard, harder environment (key → door → goal, orientation-aware).

The VLM looks at the rendered frame, reads the mission + the legal-action menu, reasons
(chain-of-thought helps here — MiniGrid is planning, not perception), and returns an
action index. Backends: Qwen2.5-VL (fp16, bsu env) or Gemma-4-12B (4-bit, gemma4 env).
Policy builders live in ``brainscore.model_helpers.local_vlm_policy``.

    python minigrid_vlm.py --backend qwen --model Qwen/Qwen2.5-VL-7B-Instruct \
        --env MiniGrid-DoorKey-6x6-v0 --episodes 10 --max_steps 60 --out /tmp/minigrid_qwen7b
"""
import argparse, json, os

import numpy as np

from brainscore_core.model_interface import BrainScoreModel
from brainscore.model_helpers.policy_wrapper import PolicyWrapper
from brainscore.harnesses.gymnasium_harness import play_gym_episode, random_gym_policy
from brainscore.model_helpers.local_vlm_policy import (
    build_minigrid_qwen_policy, build_minigrid_gemma_policy,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', required=True, choices=['qwen', 'gemma', 'random'])
    ap.add_argument('--model', default='')
    ap.add_argument('--env', default='MiniGrid-DoorKey-6x6-v0')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--max_steps', type=int, default=60)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.backend == 'qwen':
        policy, _ = build_minigrid_qwen_policy(args.model)
    elif args.backend == 'gemma':
        policy, _ = build_minigrid_gemma_policy(args.model)
    else:
        policy = random_gym_policy(0)

    def make_model():
        return BrainScoreModel(f'minigrid-{args.backend}', None, {}, {}, None,
                               action_fn=PolicyWrapper(policy, max_history=3))

    results = []
    for ep in range(args.episodes):
        res = play_gym_episode(make_model(), args.env, max_steps=args.max_steps, seed=ep)
        results.append(res)
        print(f"ep {ep}: solved={res['solved']} steps={res['steps']} reward={res['total_reward']}", flush=True)

    succ = np.mean([r['solved'] for r in results])
    rew = np.mean([r['total_reward'] for r in results])
    summary = {'backend': args.backend, 'model': args.model, 'env': args.env,
               'episodes': args.episodes, 'success_rate': round(float(succ), 3),
               'mean_reward': round(float(rew), 4), 'mission': results[0]['mission']}
    json.dump({'summary': summary, 'episodes': results}, open(os.path.join(args.out, 'result.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
