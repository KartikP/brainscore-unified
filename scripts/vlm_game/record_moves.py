"""Record the 7B CoT policy's actual moves on the website's playable boards, so
the site can replay them (watch the model play). Runs Qwen2.5-VL-7B with the
chain-of-thought visual policy on GridGameEnv(size=6, seed in {11,500,501,502})
— the exact boards the browser game offers — and saves each move sequence.

Run on EC2 (GPU). Writes JS to --out (window.MODEL_MOVES = {...}) so the static
site can load it without fetch/CORS.
"""
import argparse
import json

from brainscore_core.model_interface import BrainScoreModel
from brainscore.model_helpers.policy_wrapper import PolicyWrapper
from brainscore.harnesses.grid_game import GridGameEnv, play_game
from play_vlm_game import build_visual_policy  # same dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='11,500,501,502')
    ap.add_argument('--size', type=int, default=6)
    ap.add_argument('--max_steps', type=int, default=30)
    ap.add_argument('--model_id', default='Qwen/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--out', default='/tmp/model_moves.js')
    args = ap.parse_args()

    policy, stats, handles = build_visual_policy(args.model_id)

    def make_model(p):
        return BrainScoreModel('grid-player', None, {}, {}, None,
                               action_fn=PolicyWrapper(p, max_history=4))

    out = {}
    for s in [int(x) for x in args.seeds.split(',')]:
        env = GridGameEnv(size=args.size, seed=s, max_steps=args.max_steps)
        player, goal = list(env.agent_pos), list(env.goal_pos)   # deterministic start
        res = play_game(make_model(policy), env, max_steps=args.max_steps)
        out[str(s)] = {'player': player, 'goal': goal, 'actions': res['actions'],
                       'solved': res['solved'], 'steps': res['steps'],
                       'optimal': res['optimal_steps']}
        print(f"seed {s}: player={player} goal={goal} solved={res['solved']} "
              f"steps={res['steps']} actions={res['actions']}", flush=True)

    with open(args.out, 'w') as f:
        f.write('window.MODEL_MOVES = ' + json.dumps(out, indent=2) + ';\n')
    print('written', args.out, flush=True)


if __name__ == '__main__':
    main()
