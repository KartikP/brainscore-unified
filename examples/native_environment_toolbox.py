"""Qualify native-environment recording and intervention with a small trained model.

SmolVLM-256M reads the rendered GridGameEnvironment frame on CPU with greedy
decoding. The script compares four runs on the same seeds:

1. a direct loop over GridGameEnv, with no session, subject or tools;
2. Experiment(SessionProtocol) with RecordInputsOutputs;
3. the same plus Ablate on half of one MLP projection, with activity recorded
   before and after the intervention;
4. a restored run with no tools.

Runs 1, 2 and 4 must give identical actions and total rewards. In run 3 the
selected units must be zero and the others unchanged; changed actions are
reported, not required. This checks tool behavior, not task performance: the
model is small and the episodes are short.

The snapshot must already be staged locally; nothing here downloads:

    python examples/native_environment_toolbox.py \\
      --model-path /path/to/SmolVLM-256M-Instruct/snapshot --output-dir new-results

qualification.json is written on success and on failure (status, error, hashes).
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from brainscore.experiments import (
    Ablate, Experiment, RecordActivity, RecordInputsOutputs, SessionProtocol,
    TorchInstrumentation,
)
from brainscore.harnesses.grid_game import GridGameEnv, GridGameEnvironment
from brainscore.model_helpers.local_vlm_policy import VISUAL_PROMPT, parse_directional_action
from brainscore.model_helpers.policy_wrapper import PolicyWrapper
from brainscore_core.events import Selection
from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.streaming_helpers import EnvironmentSession

from coggym_toolbox import (
    check_ablation, environment, qualification_report, require, snapshot_digest,
)

MODEL_ID = 'HuggingFaceTB/SmolVLM-256M-Instruct'
MODEL_REVISION = '7e3e67edbbed1bf9888184d9df282b700a323964'
LAYER = 'model.text_model.layers.0.mlp.up_proj'


def load_smolvlm_policy(model_path: Path, max_new_tokens: int):
    """Return (policy, torch model) for a local SmolVLM snapshot.

    build_visual_policy is not reused: it loads the Qwen class whenever that
    class is importable and keeps the model private, so it cannot be hooked.
    The prompt and the action parser are the shared ones.
    """
    from PIL import Image
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32,
    ).eval()
    chat = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': VISUAL_PROMPT}]}]
    text = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

    def policy(observation, history):
        image = Image.fromarray(observation['frame']).resize((320, 320), Image.NEAREST)
        inputs = processor(text=text, images=[image], return_tensors='pt')
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        answer = processor.decode(output[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        action = parse_directional_action(answer, prefer_last=True)
        # ponytail: an unparsed answer maps to 'up' so every run stays deterministic.
        return 0 if action < 0 else action

    return policy, model


class _LoggedGrid(GridGameEnvironment):
    """Keep the actions the environment actually applied."""

    def __init__(self, env):
        super().__init__(env)
        self.applied = []

    def step(self, action):
        action = self.action_spec().validate(action)
        self.applied.append(int(action))
        return super().step(action)


def direct_episode(policy: Callable, seed: int, *, size: int, max_steps: int) -> dict:
    """Reference loop: no session, subject, recorder or hooks."""
    env = GridGameEnv(size=size, seed=seed, max_steps=max_steps)
    observation = env.reset(seed=seed)
    actions, total = [], 0.
    for _ in range(max_steps):
        action = int(policy(observation, ()))
        observation, reward, done, _ = env.step(action)
        actions.append(action)
        total += reward
        if done:
            break
    return {'actions': actions, 'total_reward': total}


def qualify(output_dir: Path, load: Callable[[], tuple], *, asset_dir: Path | None = None,
            layer: str = LAYER, seeds=(0, 1), size: int = 4, max_steps: int = 6,
            identifier: str = MODEL_ID, provenance: dict | None = None,
            max_record_bytes: int = 512 * 1024**2) -> dict[str, Any]:
    """Run the four conditions and write qualification.json, even on failure.

    load() returns (policy, torch model); it runs inside the report so a
    loading failure is reported too. asset_dir, if given, is hashed first.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        'purpose': 'trained-model CPU tool qualification on a native environment; '
                   'not a task-performance result',
        'settings': {'identifier': identifier, 'layer': layer, 'seeds': list(seeds),
                     'size': size, 'max_steps': max_steps, 'selection_seed': 7,
                     'max_record_bytes': max_record_bytes},
        'provenance': {**(provenance or {}), 'environment': environment()},
    }
    with qualification_report(output_dir / 'qualification.json', report):
        if asset_dir is not None:
            report['provenance']['model_sha256'] = snapshot_digest(asset_dir)
        policy, model = load()
        module = dict(model.named_modules())[layer]
        generator = torch.Generator().manual_seed(7)
        indices = sorted(torch.randperm(module.out_features, generator=generator)
                         [:module.out_features // 2].tolist())
        instrumentation = TorchInstrumentation(model)

        def run(name, tools=()):
            environments = {}

            @contextmanager
            def factory(trial):
                env = _LoggedGrid(GridGameEnv(size=size, seed=trial.identifier, max_steps=max_steps))
                environments[trial.identifier] = env
                with EnvironmentSession(environment=env, seed=trial.identifier, require_specs=True) as session:
                    yield session

            result = Experiment(
                subject=BrainScoreModel(identifier=identifier,
                                        policy=PolicyWrapper(policy, max_history=0)),
                protocol=SessionProtocol(
                    identifier='grid-game', session_factory=factory, trials=seeds,
                    input_channels={'observation'}, output_channels={'motor'},
                    collect_outputs=False,
                ),
                tools=tools,
                instrumentation=instrumentation if tools else None,
                output_dir=output_dir / name,
                metadata={'purpose': report['purpose'], 'condition': name},
            ).run()
            require(result.manifest['status'] == 'complete', f'{name} run completed')
            require(not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()),
                    f'no hooks remain after {name}')
            return result, {seed: {'actions': env.applied, 'total_reward': env.total_reward}
                            for seed, env in environments.items()}

        direct = {seed: direct_episode(policy, seed, size=size, max_steps=max_steps)
                  for seed in seeds}
        report['direct'] = {str(seed): value for seed, value in direct.items()}

        _, recorded = run('recorded', [RecordInputsOutputs(max_bytes=max_record_bytes)])
        require(recorded == direct, 'recorded actions and rewards match the direct loop')
        report['recording_preserves_actions_rewards'] = True

        ablated_run, ablated = run('ablated', [
            RecordInputsOutputs(max_bytes=max_record_bytes),
            RecordActivity([layer], when='before', name='before'),
            Ablate([Selection(layer=layer, indices=indices)]),
            RecordActivity([layer]),
        ])
        check_ablation(ablated_run.record, indices, module.out_features)
        report['ablation'] = {
            'layer': layer, 'units': len(indices), 'total_units': module.out_features,
            'selected_outputs_zero': True, 'unselected_outputs_unchanged': True,
            # Reported, not required: a change in behavior is the experiment's result.
            'changed_episodes': sum(ablated[s]['actions'] != direct[s]['actions'] for s in seeds),
            'actions': {str(seed): value['actions'] for seed, value in ablated.items()},
        }

        _, restored = run('restored')
        require(restored == direct, 'restored actions and rewards match the direct loop')
        report['restored_actions_match'] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--max-steps', type=int, default=6)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_new_tokens < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use positive limits and distinct seeds')
    report = qualify(
        args.output_dir,
        lambda: load_smolvlm_policy(args.model_path, args.max_new_tokens),
        asset_dir=args.model_path,
        seeds=tuple(args.seeds),
        max_steps=args.max_steps,
        provenance={'model_id': MODEL_ID, 'expected_revision': MODEL_REVISION,
                    'model_path': str(args.model_path.resolve()),
                    'dtype': 'float32', 'device': 'cpu', 'decoding': 'greedy',
                    'max_new_tokens': args.max_new_tokens, 'unparsed_action': 'up'},
    )
    print(f"{report['status']}: {args.output_dir / 'qualification.json'}")


if __name__ == '__main__':
    main()
