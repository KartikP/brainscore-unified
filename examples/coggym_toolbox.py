"""Qualify CogGym recording and intervention with a cached, small trained model.

This CPU smoke run uses Hu2023Fine/exp1; it is not a leaderboard reproduction.
It needs a local CogGym checkout and model snapshot and never downloads.
qualification.json is written on success and on failure (status, error, hashes).
For the shorter registered-benchmark walkthrough, see docs/coggym.md.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

from brainscore.experiments import (
    Ablate,
    Experiment,
    RecordActivity,
    RecordInputsOutputs,
    RecordReasoning,
    TorchInstrumentation,
    compare_outputs,
    replay_calls,
)
from brainscore.harnesses.coggym import CogGymRunner
from brainscore.model_helpers.response_trace import build_trace_subject
from brainscore_core.events import Selection

from example_qualification import (
    SMOLVLM_MANIFEST,
    check_ablation,
    environment,
    qualification_report,
    require,
    snapshot_digest,
    source_revision,
    verify_model_identity,
)


class LocalTextProvider:
    """Call a cached SmolVLM model using CogGym's provider arguments."""

    def __init__(self, model_path: Path) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.float32,
        ).eval()
        self.requests = []

    def reset(self, repetition: int) -> None:
        # Greedy generation has no conversation state or sampling RNG to reset.
        self.requests = []

    def complete_with_metadata(
        self,
        system: str,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        if temperature != 0:
            raise ValueError('This CPU parity example requires greedy decoding: temperature=0')
        if any(not isinstance(message['content'], str) for message in messages):
            raise ValueError('This example supports text trials only')
        request = {
            'system': system,
            'messages': messages,
            'model': model,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        self.requests.append(deepcopy(request))

        # Format CogGym's prompt for this model without adding task instructions.
        chat = [{'role': 'system', 'content': [{'type': 'text', 'text': system}]}]
        for message in messages:
            chat.append({
                'role': message['role'],
                'content': [{'type': 'text', 'text': message['content']}],
            })
        text = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(text=text, return_tensors='pt')
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        tokens = output[:, inputs['input_ids'].shape[-1]:]
        return {
            'text': self.processor.batch_decode(tokens, skip_special_tokens=True)[0],
            'reasoning': None,
            'token_usage': {
                'input_tokens': inputs['input_ids'].shape[-1],
                'output_tokens': tokens.shape[-1],
            },
        }


def _read_trials(directory: Path) -> list[dict[str, Any]]:
    return json.loads((directory / 'run-001.json').read_text())['trials']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=3)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')

    # Create the output first so any later failure leaves a failed report.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model_id = SMOLVLM_MANIFEST['repository']
    report = {
        'purpose': 'trained-model CPU tool qualification; not a leaderboard reproduction',
        'leaderboard_reproduction': False,
        'provenance': {
            'model_id': model_id,
            'model_path': str(args.model_path.resolve()),
            'checkout': str(args.checkout.resolve()),
            'environment': environment(),
        },
    }
    with qualification_report(args.output_dir / 'qualification.json', report):
        qualify(args, model_id, report)
    print(json.dumps(report, indent=2, default=str))


def qualify(args, model_id: str, report: dict[str, Any]) -> None:
    # Require pre-staged assets; nothing here downloads.
    provenance = report['provenance']
    if not args.checkout.is_dir():
        raise FileNotFoundError(f'Stage the pinned CogGym checkout first: {args.checkout}')
    # The model is identified by content before anything loads it.
    provenance['model_identity'] = verify_model_identity(args.model_path, SMOLVLM_MANIFEST)
    provenance['model_sha256'] = snapshot_digest(args.model_path)
    provenance['coggym_source'] = source_revision(args.checkout)

    # Prepare a small question subset before loading the model.
    prepared = CogGymRunner(
        args.checkout,
        model=model_id,
        temperature=0,
        max_tokens=32,
    )
    if args.trials > len(prepared.trial_ids):
        raise ValueError('--trials exceeds the canonical selection')
    runner = CogGymRunner(
        args.checkout,
        model=model_id,
        temperature=0,
        max_tokens=32,
        trial_ids=prepared.trial_ids[:args.trials],
    )
    report['scope'] = runner.describe()

    # Connect the provider to the same subject interface used by other UMI tools.
    provider = LocalTextProvider(args.model_path)
    subject = build_trace_subject(
        identifier=model_id,
        provider=lambda request: provider.complete_with_metadata(**request),
        parse=str,  # CogGym's own parser interprets answers later.
        provenance={
            'model_path': str(args.model_path.resolve()),
            'model_sha256': provenance['model_sha256'],
            'dtype': 'float32',
            'device': 'cpu',
            'decoding': 'greedy',
            'max_tokens': 32,
        },
    )

    # Choose half the intermediate MLP units in one layer, not half the model.
    layer = 'model.text_model.layers.0.mlp.up_proj'
    module = dict(provider.model.named_modules())[layer]
    generator = torch.Generator().manual_seed(7)
    shuffled = torch.randperm(module.out_features, generator=generator)
    indices = sorted(shuffled[:module.out_features // 2].tolist())
    selected = Selection(layer=layer, indices=indices)
    instrumentation = TorchInstrumentation(provider.model)  # Connect layer hooks.

    def execute(name, ablate=False):
        tools = [RecordInputsOutputs(), RecordReasoning()]
        if ablate:
            tools.extend([
                RecordActivity([layer], when='before', name='before'),
                Ablate([selected]),  # Zero the selected outputs during this run.
            ])
        tools.append(RecordActivity([layer]))  # Record after any intervention.
        result = Experiment(
            subject=subject,
            protocol=runner.protocol(reset=provider.reset),
            tools=tools,
            instrumentation=instrumentation,
            output_dir=args.output_dir / name,
            metadata={
                'purpose': 'trained-model smoke qualification',
                'ablation_layer': layer,
                'ablation_indices': indices if ablate else [],
                'selection_seed': 7,
                'temperature': 0,
            },
        ).run()
        require(not any(layer._forward_hooks for layer in provider.model.modules()),
                f'no forward hooks remain after {name}')
        return result

    # Verify that recording preserves requests, answers, parsing, and scoring.
    direct = runner.run(
        provider,
        output_dir=args.output_dir / 'direct',
        reset=provider.reset,
    )
    direct_requests = deepcopy(provider.requests)
    recorded = execute('recorded')
    require(provider.requests == direct_requests, 'recorded requests match direct requests')
    require(direct == recorded.value, 'recorded score matches direct score')
    original_trials = _read_trials(args.output_dir / 'direct')
    require(original_trials == _read_trials(recorded.directory / 'coggym'),
            'recorded trials match direct trials')
    report['recording_preserves_requests_outputs_parsing_scores'] = True

    # Use separate runs: CogGym question IDs are metadata, not UMI trial filters.
    intervened = execute('ablated', ablate=True)
    check_ablation(intervened.record, indices, module.out_features)
    restored = execute('restored')
    require(original_trials == _read_trials(restored.directory / 'coggym'),
            'restored trials match direct trials')
    report['restored_outputs_match'] = True

    # Replay sends the saved model calls again; it does not rerun CogGym.
    replay = Experiment(
        subject=subject,
        protocol=replay_calls(
            recorded.directory,
            methods=['process'],
            reset=lambda subject: provider.reset(1),
        ),
        tools=[RecordInputsOutputs()],
        output_dir=args.output_dir / 'replay',
    ).run()
    comparison = compare_outputs(recorded.directory, replay.directory)
    report['replay'] = comparison
    require(comparison['equal'], 'replayed outputs match recorded outputs')

    # Save the verification results separately from the scientific trial records.
    altered_trials = _read_trials(intervened.directory / 'coggym')
    report['ablation'] = {
        'layer': layer,
        'units': len(indices),
        'total_units': module.out_features,
        'selected_outputs_zero': True,
        'unselected_outputs_unchanged': True,
        'changed_responses': sum(
            original['response'] != changed['response']
            for original, changed in zip(original_trials, altered_trials)
        ),
        'changed_parsed_answers': sum(
            original['parsed'] != changed['parsed']
            for original, changed in zip(original_trials, altered_trials)
        ),
    }


if __name__ == '__main__':
    main()
