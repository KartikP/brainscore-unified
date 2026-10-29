"""Qualify CogGym recording and intervention with a cached, small trained model.

This CPU smoke run uses Hu2023Fine/exp1; it is not a leaderboard reproduction.
It needs a local CogGym checkout and model snapshot and never downloads.
qualification.json is written on success and on failure (status, error, hashes).
For the shorter registered-benchmark walkthrough, see docs/coggym.md.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
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


def require(condition: bool, message: str) -> None:
    """Check one qualification gate. Unlike assert, this also runs under python -O."""
    if not condition:
        raise RuntimeError(f'Qualification check failed: {message}')


def snapshot_digest(directory: Path) -> str:
    """Hash file names and contents of a local model snapshot (follows HF cache links)."""
    if not directory.is_dir():
        raise FileNotFoundError(f'Stage the model snapshot locally first: {directory}')
    digest = hashlib.sha256()
    count = 0
    for path in sorted(directory.rglob('*')):
        if not path.is_file():
            continue
        with path.open('rb') as stream:
            content = hashlib.file_digest(stream, 'sha256').digest()
        digest.update(path.relative_to(directory).as_posix().encode() + b'\0')
        digest.update(content)
        count += 1
    if count == 0:
        raise ValueError(f'Model snapshot is empty: {directory}')
    return digest.hexdigest()


def source_revision(directory: Path) -> dict[str, Any]:
    """Return the Git commit and tracked-edit state, or None for a non-Git directory."""
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(directory), *arguments],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    try:
        return {'commit': git('rev-parse', 'HEAD'),
                'tracked_edits': bool(git('status', '--porcelain', '--untracked-files=no'))}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'tracked_edits': None}


def environment() -> dict[str, Any]:
    """Versions that can change a result; missing packages are reported as None."""
    versions = {}
    for name in ('brainscore', 'brainscore-core', 'torch', 'transformers', 'numpy',
                 'gymnasium', 'minigrid'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': versions,
            'unified_source': source_revision(Path(__file__).resolve().parent)}


@contextmanager
def qualification_report(path: Path, report: dict[str, Any]):
    """Write the report whether the checks pass or fail; never claim a pass early."""
    report['status'] = 'failed'
    try:
        yield report
        report['status'] = 'passed'
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, default=str) + '\n')


def _read_trials(directory: Path) -> list[dict[str, Any]]:
    return json.loads((directory / 'run-001.json').read_text())['trials']


def check_ablation(record, indices: list[int], total_units: int) -> None:
    """Check that only the selected unit outputs were zeroed."""
    events = list(record.events())
    before = [
        event['payload']['value']['array'] for event in events
        if event['kind'] == 'activity' and event['payload']['when'] == 'before'
    ]
    after = [
        event['payload']['value']['array'] for event in events
        if event['kind'] == 'activity' and event['payload']['when'] == 'after'
    ]
    require(before and len(before) == len(after), 'paired before/after activity records')
    require(any((array[..., indices] != 0).any() for array in before),
            'selected units were active before the intervention')
    require(all((array[..., indices] == 0).all() for array in after),
            'selected units are zero after the intervention')
    unselected = sorted(set(range(total_units)) - set(indices))
    require(all(
        (original[..., unselected] == changed[..., unselected]).all()
        for original, changed in zip(before, after)
    ), 'unselected units are unchanged')


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
    model_id = 'HuggingFaceTB/SmolVLM-256M-Instruct'
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
    provenance['model_sha256'] = snapshot_digest(args.model_path)
    if not args.checkout.is_dir():
        raise FileNotFoundError(f'Stage the pinned CogGym checkout first: {args.checkout}')
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
