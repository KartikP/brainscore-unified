"""Qualify CogGym recording and intervention with a cached, small trained model.

This CPU smoke run uses Hu2023Fine/exp1; it is not a leaderboard reproduction.
See docs/coggym.md for the reference checkout and environment setup.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

from brainscore.experiments import (
    Ablate, Experiment, RecordActivity, RecordInputsOutputs, RecordReasoning,
    TorchInstrumentation, compare_outputs, replay_calls,
)
from brainscore.harnesses.coggym import CogGymRunner
from brainscore.model_helpers.response_trace import build_trace_subject
from brainscore_core.events import Selection


class LocalTextProvider:
    """CogGym's provider interface, using a local SmolVLM model in text mode."""

    def __init__(self, model_path):
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, dtype=torch.float32,
        ).eval()
        self.requests = []

    def reset(self, repetition):
        # Greedy generation has no conversation state or sampling RNG to reset.
        self.requests = []

    def complete_with_metadata(self, system, messages, model, temperature, max_tokens):
        if temperature != 0:
            raise ValueError('This CPU parity example requires greedy decoding: temperature=0')
        if any(not isinstance(message['content'], str) for message in messages):
            raise ValueError('This example supports text trials only')
        self.requests.append(deepcopy(dict(
            system=system, messages=messages, model=model,
            temperature=temperature, max_tokens=max_tokens,
        )))
        # Only format the reference prompt for this model; add no task instructions.
        chat = [{'role': 'system', 'content': [{'type': 'text', 'text': system}]}]
        chat += [{'role': message['role'], 'content': [
            {'type': 'text', 'text': message['content']},
        ]} for message in messages]
        text = self.processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, return_tensors='pt')
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        tokens = output[:, inputs['input_ids'].shape[-1]:]
        return {
            'text': self.processor.batch_decode(tokens, skip_special_tokens=True)[0],
            'reasoning': None,
            'token_usage': {'input_tokens': inputs['input_ids'].shape[-1],
                            'output_tokens': tokens.shape[-1]},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=3)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    model_id = 'HuggingFaceTB/SmolVLM-256M-Instruct'
    prepared = CogGymRunner(args.checkout, model=model_id, temperature=0, max_tokens=32)
    if args.trials > len(prepared.trial_ids):
        parser.error('--trials exceeds the canonical selection')
    runner = CogGymRunner(
        args.checkout, model=model_id, temperature=0, max_tokens=32,
        trial_ids=prepared.trial_ids[:args.trials],
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    provider = LocalTextProvider(args.model_path)
    subject = build_trace_subject(
        model_id,
        provider=lambda request: provider.complete_with_metadata(**request),
        parse=str,  # CogGym's own parser interprets answers later.
        provenance={'model_path': str(args.model_path.resolve()), 'dtype': 'float32',
                    'device': 'cpu', 'decoding': 'greedy', 'max_tokens': 32},
    )

    # Choose half the intermediate MLP units in ONE layer, not half the model.
    layer = 'model.text_model.layers.0.mlp.up_proj'
    module = dict(provider.model.named_modules())[layer]
    generator = torch.Generator().manual_seed(7)
    indices = sorted(torch.randperm(module.out_features, generator=generator)
                     [:module.out_features // 2].tolist())
    selected = Selection(layer, indices=indices)
    instrumentation = TorchInstrumentation(provider.model)

    def execute(name, ablate=False):
        tools = [RecordInputsOutputs(), RecordReasoning()]
        if ablate:
            tools.extend([
                RecordActivity([layer], when='before', name='before'),
                Ablate([selected]),
            ])
        tools.append(RecordActivity([layer]))
        result = Experiment(
            subject=subject, protocol=runner.protocol(reset=provider.reset),
            tools=tools, instrumentation=instrumentation,
            output_dir=args.output_dir / name,
            metadata={'purpose': 'trained-model smoke qualification',
                      'ablation_layer': layer, 'ablation_indices': indices if ablate else [],
                      'selection_seed': 7, 'temperature': 0},
        ).run()
        assert not any(m._forward_hooks for m in provider.model.modules())
        return result

    direct = runner.run(provider, output_dir=args.output_dir / 'direct', reset=provider.reset)
    direct_requests = deepcopy(provider.requests)
    recorded = execute('recorded')
    assert provider.requests == direct_requests
    assert direct == recorded.value

    def trials(directory):
        return json.loads((directory / 'run-001.json').read_text())['trials']

    original_trials = trials(args.output_dir / 'direct')
    assert original_trials == trials(recorded.directory / 'coggym')
    intervened = execute('ablated', ablate=True)
    events = list(intervened.record.events())
    before = [e['payload']['value']['array'] for e in events
              if e['kind'] == 'activity' and e['payload']['when'] == 'before']
    after = [e['payload']['value']['array'] for e in events
             if e['kind'] == 'activity' and e['payload']['when'] == 'after']
    assert before and len(before) == len(after)
    assert any((array[..., indices] != 0).any() for array in before)
    assert all((array[..., indices] == 0).all() for array in after)
    unselected = sorted(set(range(module.out_features)) - set(indices))
    assert all((a[..., unselected] == b[..., unselected]).all() for a, b in zip(before, after))
    restored = execute('restored')
    assert original_trials == trials(restored.directory / 'coggym')

    # Replay sends recorded model calls again; it does not rerun a task environment.
    replay = Experiment(
        subject=subject,
        protocol=replay_calls(recorded.directory, methods=['process'],
                              reset=lambda subject: provider.reset(1)),
        tools=[RecordInputsOutputs()], output_dir=args.output_dir / 'replay',
    ).run()
    comparison = compare_outputs(recorded.directory, replay.directory)
    assert comparison['equal']
    altered_trials = trials(intervened.directory / 'coggym')
    report = {
        'status': 'passed', 'scope': runner.describe(),
        'recording_preserves_requests_outputs_parsing_scores': True,
        'restored_outputs_match': True, 'replay': comparison,
        'ablation': {'layer': layer, 'units': len(indices), 'total_units': module.out_features,
                     'selected_outputs_zero': True, 'unselected_outputs_unchanged': True,
                     'changed_responses': sum(a['response'] != b['response']
                                              for a, b in zip(original_trials, altered_trials)),
                     'changed_parsed_answers': sum(a['parsed'] != b['parsed']
                                                   for a, b in zip(original_trials, altered_trials))},
        'leaderboard_reproduction': False,
    }
    (args.output_dir / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
