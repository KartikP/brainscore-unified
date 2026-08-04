"""Local (HuggingFace-weights) VLM / LM game policies.

The local-weights twin of ``api_behavioral.build_api_action_fn``: turn a HF
vision-language or text model into a one-tick game policy
``policy(observation, history) -> action_index`` that ``PolicyWrapper``
drives through ``process(EnvironmentStep)``. These builders were consolidated
here from several duplicated, cross-importing copies.

Two action vocabularies, two parsers:
- grid-game (``harnesses.grid_game``): 4 directional actions; the model emits
  a direction word → ``parse_directional_action``.
- MiniGrid (``harnesses.gymnasium_harness``): N numbered actions; the model
  emits ``Action: <number>`` → ``parse_indexed_action``.

The builders load heavy weights and are not unit-tested here; the two parsers
are pure and carry the coverage. Builders return ``(policy, stats)`` where
``stats`` is ``{'calls', 'parse_miss'}``; ``del policy`` drops the only
reference to the loaded model so a ``gc.collect()`` frees the GPU between
runs of a model ladder.
"""
import re

import numpy as np
from brainscore_core.hf_compat import pin_image_processor

WORD_TO_ACTION = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

VISUAL_PROMPT = (
    "This is a grid game. The BLUE square is the player; the GREEN square is the "
    "goal. Row 0 is the top row; column 0 is the left column.\n"
    "Think step by step:\n"
    "1. State the (row, column) of the BLUE square.\n"
    "2. State the (row, column) of the GREEN square.\n"
    "3. Decide the single move that reduces the distance: 'up' decreases the "
    "row, 'down' increases the row, 'left' decreases the column, 'right' "
    "increases the column.\n"
    "End your reply with a line exactly: Action: <up|down|left|right>"
)

ASCII_PROMPT = (
    "You are playing a grid game. The board uses: P = player, G = goal, "
    "# = wall, . = empty. Row 0 is the top. 'up' decreases the row, 'down' "
    "increases it, 'left' decreases the column, 'right' increases it.\n\n"
    "Board:\n{board}\n\n"
    "Think step by step about which single move brings P closer to G, then end "
    "your reply with a line exactly like:\nAction: <up|down|left|right>"
)

MINIGRID_PROMPT = (
    "You control the red triangle agent in a grid world. The triangle POINTS in the "
    "direction the agent currently faces; 'move forward' goes that way. "
    "Mission: {mission}.\n"
    "Legend: yellow key = pick it up to unlock the door; a colored bar set into a wall "
    "= a door (locked until opened while carrying the key); green square = the goal.\n"
    "Actions: {menu}.\n"
    "Reason briefly about which way the agent faces and the next best step, then end with "
    "a line EXACTLY: 'Action: <number>'."
)


# ── Parsers (pure; the tested core) ────────────────────────────────

def parse_directional_action(text, prefer_last=False):
    """Extract a directional action (0..3) from model output. Prefer an
    explicit 'Action: <word>'; else the first (or last) direction word.
    Returns -1 if none found."""
    m = re.search(r'action\s*[:\-]\s*(up|down|left|right)', text, re.IGNORECASE)
    if m:
        return WORD_TO_ACTION[m.group(1).lower()]
    words = re.findall(r'\b(up|down|left|right)\b', text, re.IGNORECASE)
    if words:
        return WORD_TO_ACTION[(words[-1] if prefer_last else words[0]).lower()]
    return -1


def parse_indexed_action(text, n_actions, rng=None):
    """Extract a numbered action from model output. Prefer 'Action: <n>';
    else the last bare single digit; taken modulo n_actions. If nothing
    parses, return ``rng.randint(0, n_actions)`` when an RNG is supplied
    (so a non-following model degrades to the random floor), else -1."""
    m = list(re.finditer(r'action\s*[:=]?\s*(\d+)', text.lower()))
    if not m:
        m = list(re.finditer(r'\b(\d)\b', text))
    if not m:
        return int(rng.randint(0, n_actions)) if rng is not None else -1
    return int(m[-1].group(1)) % n_actions


# ── grid-game policy builders (4 directional actions) ──────────────

def build_visual_policy(model_id, *, prompt=VISUAL_PROMPT, max_new_tokens=200):
    """Qwen2.5-VL-style policy that reads the rendered RGB frame."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLM
    except Exception:
        # AutoModelForVision2Seq was removed in transformers 5; the
        # ImageTextToText auto-class is its replacement and exists in 4.57 too.
        from transformers import AutoModelForImageTextToText as VLM
    from brainscore.harnesses.grid_game import ACTIONS

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(model_id)
    processor = pin_image_processor(processor, model_id)
    model = VLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map=device).eval()
    stats = {'parse_miss': 0, 'calls': 0}
    rng = np.random.RandomState(0)

    def policy(observation, history):
        stats['calls'] += 1
        img = Image.fromarray(observation['frame']).resize((320, 320), Image.NEAREST)
        messages = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        ans = processor.decode(out[0][inputs['input_ids'].shape[1]:],
                               skip_special_tokens=True)
        a = parse_directional_action(ans, prefer_last=True)
        if a < 0:
            stats['parse_miss'] += 1
            return int(rng.randint(0, len(ACTIONS)))
        return a

    return policy, stats


def build_thinking_policy(model_id, *, max_new_tokens=1024):
    """Text reasoner that reads the ASCII board (perfect perception)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from brainscore.harnesses.grid_game import ACTIONS

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map=device).eval()
    stats = {'parse_miss': 0, 'calls': 0}
    rng = np.random.RandomState(0)

    def policy(observation, history):
        stats['calls'] += 1
        prompt = ASCII_PROMPT.format(board=observation['ascii'])
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
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        ans = tok.decode(out[0][inputs['input_ids'].shape[1]:],
                         skip_special_tokens=True)
        a = parse_directional_action(ans, prefer_last=True)
        if a < 0:
            stats['parse_miss'] += 1
            return int(rng.randint(0, len(ACTIONS)))
        return a

    return policy, stats


def build_gemma_visual_policy(model_id, *, prompt=VISUAL_PROMPT, max_new_tokens=200,
                              revision=None):
    """Gemma-style 4-bit image-text policy that reads the rendered frame."""
    import torch
    from PIL import Image
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              BitsAndBytesConfig)
    from brainscore.harnesses.grid_game import ACTIONS

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=['patch_dense', 'embedding_projection', 'lm_head'])
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    processor = pin_image_processor(processor, model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, quantization_config=bnb, device_map='auto',
        dtype=torch.bfloat16).eval()
    device = next(model.parameters()).device
    stats = {'parse_miss': 0, 'calls': 0}
    rng = np.random.RandomState(0)

    def policy(observation, history):
        stats['calls'] += 1
        img = Image.fromarray(observation['frame']).resize((320, 320), Image.NEAREST)
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img}, {'type': 'text', 'text': prompt}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        ans = processor.decode(out[0][inputs['input_ids'].shape[1]:],
                               skip_special_tokens=True)
        a = parse_directional_action(ans, prefer_last=True)
        if a < 0:
            stats['parse_miss'] += 1
            return int(rng.randint(0, len(ACTIONS)))
        return a

    return policy, stats


# ── MiniGrid policy builders (N numbered actions) ──────────────────

def build_minigrid_qwen_policy(model_id, *, max_new_tokens=220):
    """Qwen2.5-VL policy for the MiniGrid (gymnasium) harness."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration as VLM

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    proc = AutoProcessor.from_pretrained(model_id)
    proc = pin_image_processor(proc, model_id)
    model = VLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if dev == 'cuda' else torch.float32,
        device_map=dev).eval()
    stats = {'parse_miss': 0, 'calls': 0}
    rng = np.random.RandomState(0)

    def policy(obs, history):
        stats['calls'] += 1
        n = len(obs['legal_actions'])
        menu = '; '.join(f'{k}={v}' for k, v in obs['legal_actions'].items())
        prompt = MINIGRID_PROMPT.format(mission=obs['instruction'], menu=menu)
        img = Image.fromarray(obs['frame']).resize((336, 336), Image.NEAREST)
        msgs = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[img], return_tensors='pt').to(dev)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        ans = proc.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
        return parse_indexed_action(ans, n, rng)

    return policy, stats


def build_minigrid_gemma_policy(model_id, *, max_new_tokens=256, revision=None):
    """Gemma 4-bit image-text policy for the MiniGrid harness."""
    import torch
    from PIL import Image
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              BitsAndBytesConfig)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=['patch_dense', 'embedding_projection', 'lm_head'])
    proc = AutoProcessor.from_pretrained(model_id, revision=revision)
    proc = pin_image_processor(proc, model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, quantization_config=bnb,
        device_map='auto', dtype=torch.bfloat16).eval()
    dev = next(model.parameters()).device
    stats = {'parse_miss': 0, 'calls': 0}
    rng = np.random.RandomState(0)

    def policy(obs, history):
        stats['calls'] += 1
        n = len(obs['legal_actions'])
        menu = '; '.join(f'{k}={v}' for k, v in obs['legal_actions'].items())
        prompt = MINIGRID_PROMPT.format(mission=obs['instruction'], menu=menu)
        img = Image.fromarray(obs['frame']).resize((336, 336), Image.NEAREST)
        msgs = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img}, {'type': 'text', 'text': prompt}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors='pt').to(dev)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        ans = proc.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
        return parse_indexed_action(ans, n, rng)

    return policy, stats


# ── grid-game episode-runner glue (shared by the drivers) ──────────

def build_grid_player(policy, max_history=4):
    """Wrap a policy in a BrainScoreModel(action_fn=PolicyWrapper(...))."""
    from brainscore_core.model_interface import BrainScoreModel
    from brainscore.model_helpers.policy_wrapper import PolicyWrapper
    return BrainScoreModel(
        identifier='grid-player', model=None, region_layer_map={},
        preprocessors={}, activations_model=None,
        action_fn=PolicyWrapper(policy, max_history=max_history))


def run_grid_episodes(policy, n_episodes, size, max_steps, base_seed):
    """Run a policy over n_episodes of GridGameEnv; return success/efficiency."""
    from brainscore.harnesses.grid_game import GridGameEnv, play_game
    solves, effs = [], []
    for i in range(n_episodes):
        env = GridGameEnv(size=size, seed=base_seed + i, max_steps=max_steps)
        res = play_game(build_grid_player(policy), env)
        solves.append(1.0 if res['solved'] else 0.0)
        if res['solved']:
            effs.append(res['efficiency'])
    return {'success_rate': float(np.mean(solves)),
            'mean_efficiency': float(np.mean(effs)) if effs else 0.0,
            'n_episodes': n_episodes}
