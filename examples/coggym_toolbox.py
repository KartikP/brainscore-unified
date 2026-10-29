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


# Hub listing of the loader's files at the pinned commit (metadata only). sha256 is the
# LFS hash of the file; git_sha1 is the Git blob hash Git uses for small files.
SMOLVLM_MANIFEST = {
    'repository': 'HuggingFaceTB/SmolVLM-256M-Instruct',
    'revision': '7e3e67edbbed1bf9888184d9df282b700a323964',
    'files': {
        'added_tokens.json': {'size': 4739, 'git_sha1': '71dc811700b6aa9274d3697ee7d4fb2d5375e766'},
        'chat_template.json': {'size': 429, 'git_sha1': '214a30563adf9598b26cdd3ad3bbafca9d7af45d'},
        'config.json': {'size': 7353, 'git_sha1': '1c09d62975f946e7e07f62ec4db3f16c9c701144'},
        'generation_config.json': {'size': 136, 'git_sha1': '14051e31eab5938a09976bf8c3e4769491f53a80'},
        'merges.txt': {'size': 466391, 'git_sha1': '69503b13f727ba3812b6803e97442a6de05ef5eb'},
        'model.safetensors': {
            'size': 513028808,
            'sha256': '74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e'},
        'preprocessor_config.json': {'size': 486, 'git_sha1': '893236748032d4bf033823d6a4a0d62b78b6bc0e'},
        'processor_config.json': {'size': 68, 'git_sha1': 'b3b1deadd0e6e18c2070fdb0c29074892ea58ffe'},
        'special_tokens_map.json': {'size': 1069, 'git_sha1': '647565708c361dc42350b5c7b79c46a4ff153b18'},
        'tokenizer.json': {'size': 3548256, 'git_sha1': 'a4005d1cf3170a31600a5c96f95768166cbc2b28'},
        'tokenizer_config.json': {'size': 28249, 'git_sha1': 'c85c07d72e6ffca6b6ff138a8e8f09ae65740bfb'},
        'vocab.json': {'size': 800662, 'git_sha1': '0ad5ecc2035b7031b88afb544ee95e2d49baa484'},
    },
    # Present in the repository but never read by the loader; not verified.
    'ignored': ['.gitattributes', 'README.md', 'onnx', '.cache'],
}


def file_hash(path: Path, kind: str) -> str:
    """sha256 of the bytes, or the Git blob SHA-1 ('blob <size>\\0' + bytes)."""
    with path.open('rb') as stream:
        if kind == 'sha256':
            return hashlib.file_digest(stream, 'sha256').hexdigest()
        digest = hashlib.sha1(b'blob %d\0' % path.stat().st_size)
        return hashlib.file_digest(stream, lambda: digest).hexdigest()


def verify_model_identity(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check a staged snapshot against a pinned Hub listing before anything loads it.

    Accepts a Hugging Face cache snapshot (.../snapshots/<commit>, files may be
    links into blobs/) or a huggingface-cli --local-dir download. Every file the
    loader reads must exist with the listed size and hash, computed from the bytes
    on disk. Any other top-level entry must be on the manifest's ignore list, so an
    extra file cannot change what is loaded. A cache snapshot name or a local-dir
    download record naming another commit fails.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f'Stage the model snapshot locally first: {directory}')
    revision = manifest['revision']
    if directory.parent.name == 'snapshots':
        require(directory.name == revision,
                f'cache snapshot {directory.name} is not the pinned revision {revision}')
    files, ignored = manifest['files'], set(manifest['ignored'])
    unexpected = sorted(p.name for p in directory.iterdir()
                        if p.name not in files and p.name not in ignored)
    require(not unexpected, f'unexpected files in the model snapshot: {unexpected}')
    records = directory / '.cache' / 'huggingface' / 'download'
    for name, expected in sorted(files.items()):
        path = directory / name
        require(path.is_file(), f'model file {name} is missing')
        require(path.stat().st_size == expected['size'], f'model file {name} has the wrong size')
        kind = 'sha256' if 'sha256' in expected else 'git_sha1'
        require(file_hash(path, kind) == expected[kind],
                f'model file {name} does not match {manifest["repository"]}@{revision}')
        record = records / f'{name}.metadata'
        if record.is_file():
            # huggingface-cli writes the commit on the record's first line.
            require(record.read_text().splitlines()[:1] == [revision],
                    f'download record for {name} names another revision')
    return {'repository': manifest['repository'], 'revision': revision,
            'verified_files': sorted(files),
            'method': 'size and content hash of each loader file against the Hub listing'}


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
