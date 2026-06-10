"""Play the MiniGrid embodied game with an API model (OpenRouter / OpenAI /
DeepSeek / Anthropic) — laptop-runnable, NO GPU, no model weights.

The model is driven one tick at a time through ``process(EnvironmentStep)``:
it sees the rendered frame + the legal-action menu and returns an action index.
Reuses the OpenAI-compatible API adapters + response cache from
``brainscore.model_helpers.api_behavioral``.

Examples:
    # Llama-3.3-70B via OpenRouter (set OPENROUTER_API_KEY)
    python play_api_game.py --provider openrouter \
        --model meta-llama/llama-3.3-70b-instruct --games 15

    # DeepSeek-R1 via OpenRouter
    python play_api_game.py --provider openrouter \
        --model deepseek/deepseek-r1 --games 15 --max_tokens 1024

    # GPT-4o direct (set OPENAI_API_KEY)
    python play_api_game.py --provider openai --model gpt-4o-2024-08-06 --games 15

Cost/latency note: this is a sequential closed loop — each tick is one API call,
so N games x up-to-max_steps ticks = hundreds of calls. The response cache makes
re-runs of the same seeds free + reproducible.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # repo root (…/unified) on path

from brainscore_core.model_interface import BrainScoreModel
from brainscore.model_helpers.api_behavioral import build_api_action_fn
from brainscore.harnesses.gymnasium_harness import play_gym_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', default='openrouter',
                    help='openrouter | openai | deepseek | anthropic')
    ap.add_argument('--model', required=True,
                    help="provider model id, e.g. meta-llama/llama-3.3-70b-instruct")
    ap.add_argument('--env', default='MiniGrid-DoorKey-6x6-v0')
    ap.add_argument('--games', type=int, default=15)
    ap.add_argument('--max_steps', type=int, default=60)
    ap.add_argument('--obs_mode', choices=('vision', 'ascii'), default='vision',
                    help="'vision' sends the rendered frame; 'ascii' sends the "
                         "text board (works for GridGameEnv; text models can play)")
    ap.add_argument('--max_tokens', type=int, default=512)
    ap.add_argument('--frame_resize', type=int, default=384)
    # Cache OFF by default: caching a closed agentic loop can freeze a stuck
    # state into a no-op loop. Pass --cache_dir only for reproducible re-runs.
    ap.add_argument('--cache_dir', default=None)
    ap.add_argument('--out', default='/tmp/api_game_result.json')
    args = ap.parse_args()

    cache_dir = (f"{args.cache_dir}/{args.provider}_{args.model.replace('/', '_')}"
                 if args.cache_dir else None)
    action_fn = build_api_action_fn(
        args.provider, args.model, obs_mode=args.obs_mode,
        max_tokens=args.max_tokens, cache_dir=cache_dir,
        frame_resize=args.frame_resize)
    model = BrainScoreModel(
        identifier=f'{args.provider}:{args.model}', model=None,
        region_layer_map={}, preprocessors={}, action_fn=action_fn)

    t0 = time.time()
    solved, episodes = 0, []
    for seed in range(args.games):
        res = play_gym_episode(model, args.env, max_steps=args.max_steps, seed=seed)
        solved += int(res['solved'])
        episodes.append(res)
        print(f"  game {seed:2d}: solved={res['solved']} steps={res['steps']} "
              f"reward={res['total_reward']}", flush=True)

    rate = solved / args.games if args.games else 0.0
    summary = {
        'provider': args.provider, 'model': args.model, 'env': args.env,
        'games': args.games, 'solved': solved, 'solve_rate': round(rate, 4),
        'wall_sec': round(time.time() - t0, 1), 'episodes': episodes,
    }
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSOLVE RATE: {solved}/{args.games} = {rate:.3f}  "
          f"({summary['wall_sec']}s)  -> {args.out}")


if __name__ == '__main__':
    main()
