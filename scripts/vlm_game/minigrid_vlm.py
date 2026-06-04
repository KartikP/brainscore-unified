"""Drive a complex Gymnasium game (MiniGrid) with a VLM policy, through the unified
``process(EnvironmentStep)`` interface — the same closed loop as the toy grid game,
now on a standard, harder environment (key → door → goal, orientation-aware).

The VLM looks at the rendered frame, reads the mission + the legal-action menu, reasons
(chain-of-thought helps here — MiniGrid is planning, not perception), and returns an
action index. Backends: Qwen2.5-VL (fp16, bsu env) or Gemma-4-12B (4-bit, gemma4 env).

    python minigrid_vlm.py --backend qwen --model Qwen/Qwen2.5-VL-7B-Instruct \
        --env MiniGrid-DoorKey-6x6-v0 --episodes 10 --max_steps 60 --out /tmp/minigrid_qwen7b
"""
import argparse, json, os, re

import numpy as np

from brainscore_core.model_interface import BrainScoreModel
from brainscore.model_helpers.policy_wrapper import PolicyWrapper
from brainscore.harnesses.gymnasium_harness import play_gym_episode, MINIGRID_ACTIONS, random_gym_policy

PROMPT_TMPL = (
    "You control the red triangle agent in a grid world. The triangle POINTS in the "
    "direction the agent currently faces; 'move forward' goes that way. "
    "Mission: {mission}.\n"
    "Legend: yellow key = pick it up to unlock the door; a colored bar set into a wall "
    "= a door (locked until opened while carrying the key); green square = the goal.\n"
    "Actions: {menu}.\n"
    "Reason briefly about which way the agent faces and the next best step, then end with "
    "a line EXACTLY: 'Action: <number>'."
)


def _parse_action(text, n, rng):
    m = list(re.finditer(r'action\s*[:=]?\s*(\d+)', text.lower()))
    if not m:
        m = list(re.finditer(r'\b(\d)\b', text))
    if not m:
        return int(rng.randint(0, n))
    return int(m[-1].group(1)) % n


def build_qwen_policy(model_id):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration as VLM
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    proc = AutoProcessor.from_pretrained(model_id)
    model = VLM.from_pretrained(model_id, torch_dtype=torch.float16 if dev == 'cuda' else torch.float32,
                               device_map=dev).eval()
    rng = np.random.RandomState(0)

    def policy(obs, history):
        n = len(obs['legal_actions'])
        menu = '; '.join(f'{k}={v}' for k, v in obs['legal_actions'].items())
        prompt = PROMPT_TMPL.format(mission=obs['instruction'], menu=menu)
        img = Image.fromarray(obs['frame']).resize((336, 336), Image.NEAREST)
        msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[img], return_tensors='pt').to(dev)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=220, do_sample=False)
        ans = proc.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
        return _parse_action(ans, n, rng)
    return policy


def build_gemma_policy(model_id):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             llm_int8_skip_modules=['patch_dense', 'embedding_projection', 'lm_head'])
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=bnb, device_map='auto', dtype=torch.bfloat16).eval()
    dev = next(model.parameters()).device
    rng = np.random.RandomState(0)

    def policy(obs, history):
        n = len(obs['legal_actions'])
        menu = '; '.join(f'{k}={v}' for k, v in obs['legal_actions'].items())
        prompt = PROMPT_TMPL.format(mission=obs['instruction'], menu=menu)
        img = Image.fromarray(obs['frame']).resize((336, 336), Image.NEAREST)
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': prompt}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors='pt').to(dev)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, do_sample=False)
        ans = proc.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
        return _parse_action(ans, n, rng)
    return policy


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
        policy = build_qwen_policy(args.model)
    elif args.backend == 'gemma':
        policy = build_gemma_policy(args.model)
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
