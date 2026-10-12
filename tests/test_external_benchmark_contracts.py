"""Reject incomparable settings and unsupported requests before model inference."""
import json
from pathlib import Path

import pandas as pd
import pytest

import brainscore
from brainscore_core.compatibility import CompatibilityError, check_channel_compatibility
from brainscore_core.streaming import StreamEvent, InMemorySession
from brainscore.data.preparation import prepare_dataset
from brainscore.experiments import Experiment, RecordInputsOutputs
from brainscore.model_helpers.response_trace import build_trace_subject
from brainscore.harnesses.coggym import CogGymRunner

from .test_coggym import reference, make_subject, provider
from .test_coggym_benchmark import scored_reference
from .test_ewok import native, prepared, subject, NativeModel, EWoKProvider

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('experiment,content,channel', [
    ('Aboody2025Inferring/exp1', ['image'], 'image_generation_request'),
    ('Fu2025Hierarchical/exp1', ['image', 'video'], 'video_generation_request'),
])
def test_media_requires_explicit_support_and_preserves_payload(
    reference, tmp_path, experiment, content, channel,
):
    root, cli = reference
    (root / 'evaluation/public_manifest.json').write_text(json.dumps({
        'experiments': [{'path': experiment}],
    }))
    messages = [{'role': 'user', 'content': [
        {'type': kind, 'data': f'fixture-{kind}'} for kind in content
    ]}]
    cli.build_messages = lambda *args: ('instructions', messages)
    benchmark = brainscore.load_benchmark(f'CogGym.{experiment.replace("/", ".")}', checkout=root)
    calls = []
    def capture(request):
        calls.append(request)
        return {'text': '50', 'reasoning': None, 'token_usage': None}
    text_subject = make_subject(capture)
    for check in (check_channel_compatibility, lambda model, bench: bench(model)):
        with pytest.raises(CompatibilityError):
            check(text_subject, benchmark)
    assert calls == []
    candidate = build_trace_subject(
        'fixture', provider=capture, parse=str, provenance={},
        input_channels=[channel],
    )
    check_channel_compatibility(candidate, benchmark)
    runner = CogGymRunner(
        root, experiment=experiment, model='fixture', modalities=['text', *content],
    )
    Experiment(
        subject=candidate,
        protocol=runner.protocol(reset=lambda repetition: None),
        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'media',
    ).run()
    assert len(calls) == 3
    assert all(call['messages'] == messages for call in calls)
    with pytest.raises(CompatibilityError):
        Experiment(
            subject=text_subject, protocol=runner.protocol(reset=lambda repetition: None),
            output_dir=tmp_path / 'unsupported',
        ).run()
    assert len(calls) == 3


def test_coggym_reference_profile_reset_and_leaderboard_policy(scored_reference, tmp_path):
    root, _ = scored_reference
    calls, resets = [], []
    candidate = build_trace_subject(
        'fixture', provider=lambda request: calls.append(request) or provider(request),
        parse=str, provenance={}, reset=resets.append,
    )
    benchmark = brainscore.load_benchmark(
        'CogGym.Hu2023Fine.exp1', checkout=root, output_dir=tmp_path / 'default',
    )
    score = benchmark(candidate)
    assert benchmark.identifier == 'CogGym.Hu2023Fine.exp1'
    assert resets == [1, 2, 3, 4, 5]
    assert len(calls) == 15 and all(call['max_tokens'] == 512 for call in calls)
    assert benchmark.leaderboard_eligible is score.attrs['leaderboard_eligible'] is False
    assert score.attrs['reference_profile'] is True
    assert score.attrs['records_retained'] is True
    assert (tmp_path / 'default/coggym/run-001.json').exists()


@pytest.mark.parametrize('outcome', ['success', 'undefined_score', 'interrupted'])
def test_coggym_implicit_records_always_deleted(scored_reference, tmp_path, monkeypatch, outcome):
    import brainscore.benchmarks.coggym.benchmark as module
    import tempfile
    root, analysis = scored_reference
    original = tempfile.TemporaryDirectory
    created = []

    def directory(**kwargs):
        handle = original(dir=tmp_path, **kwargs)
        created.append(Path(handle.name))
        return handle

    def respond(request):
        if outcome == 'interrupted':
            raise KeyboardInterrupt('fixture interrupt')
        return provider(request)

    monkeypatch.setattr(module.tempfile, 'TemporaryDirectory', directory)
    if outcome == 'undefined_score':
        analysis.analyze = lambda *args, **kwargs: ([{
            'model': 'fixture', 'experiment': 'Hu2023Fine/exp1', 'r2': None,
        }], [], [])
    benchmark = brainscore.load_benchmark(
        'CogGym.Hu2023Fine.exp1', checkout=root, reset=lambda repetition: None,
    )
    candidate = make_subject(respond)
    if outcome == 'success':
        score = benchmark(candidate)
        assert float(score) == 0.375
        assert score.attrs['run_directory'] is None
        assert score.attrs['records_retained'] is False
    else:
        with pytest.raises(KeyboardInterrupt if outcome == 'interrupted' else ValueError):
            benchmark(candidate)
    assert len(created) == 1 and not created[0].exists()
    assert not list(tmp_path.glob('brainscore-coggym-*'))


@pytest.mark.parametrize('settings', [
    {'max_tokens': 8192}, {'repetitions': 1}, {'temperature': 0.0},
    {'temperature': 1.00000001},
])
def test_nonreference_coggym_settings_have_distinct_id(scored_reference, settings):
    root, _ = scored_reference
    default = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', checkout=root)
    custom = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', checkout=root, **settings)
    assert custom.identifier != default.identifier


@pytest.mark.parametrize('mode', ['logprobs', 'choice'])
def test_ewok_generation_only_provider_rejected_without_calls(prepared, mode):
    calls = []
    candidate = build_trace_subject(
        'text-only', provider=lambda request: calls.append(request), parse=str, provenance={},
    )
    benchmark = brainscore.load_benchmark(f'EWoK-core-1.0-{mode}', root=prepared)
    with pytest.raises(CompatibilityError):
        benchmark(candidate)
    assert calls == []


def test_ewok_domains_and_source_have_distinct_stable_ids(native, tmp_path):
    frame = pd.read_csv(native)
    frame.loc[0, 'Domain'] = 'another-domain'
    frame.to_csv(native, index=False)
    root = prepare_dataset('EWoK-core-1.0', source=native, output=tmp_path / 'data')
    def load(**kwargs):
        return brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=root, **kwargs)
    full = load()
    first = load(domains=['another-domain'])
    second = load(domains=['synthetic-domain'])
    assert len({full.identifier, first.identifier, second.identifier}) == 3
    assert first.identifier == load(domains=['another-domain']).identifier
    assert load(domains=['another-domain', 'synthetic-domain']).identifier == load(
        domains=['synthetic-domain', 'another-domain'],
    ).identifier
    frame.loc[0, 'Context1'] = 'a changed synthetic context'
    frame.to_csv(native, index=False)
    other = prepare_dataset('EWoK-core-1.0', source=native, output=tmp_path / 'changed')
    assert full.identifier != brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=other).identifier


@pytest.mark.parametrize('fail', [False, True])
def test_ewok_implicit_records_deleted_on_success_and_failure(prepared, tmp_path, monkeypatch, fail):
    import brainscore.benchmarks.ewok.benchmark as module
    import tempfile
    original = tempfile.TemporaryDirectory
    created = []
    def directory(**kwargs):
        handle = original(dir=tmp_path, **kwargs)
        created.append(Path(handle.name))
        return handle
    monkeypatch.setattr(module.tempfile, 'TemporaryDirectory', directory)
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=prepared)
    model = subject((lambda request: {'text': '[]'}) if fail else EWoKProvider(NativeModel()))
    if fail:
        with pytest.raises(ValueError):
            benchmark(model)
    else:
        result = benchmark(model)
        assert result.attrs['run_directory'] is None
        assert result.attrs['records_retained'] is False
        assert result.attrs['leaderboard_eligible'] is benchmark.leaderboard_eligible is False
    assert len(created) == 1 and not created[0].exists()


def test_ewok_explicit_records_private_and_score_uses_resolved_id(prepared, tmp_path, monkeypatch):
    monkeypatch.setenv('BRAINSCORE_EWOK_ROOT', str(prepared))
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path / 'cache'))
    model = subject(EWoKProvider(NativeModel()))
    score = brainscore.score(model, 'EWoK-core-1.0-logprobs', check_mem=False)
    assert '-data-' in score.attrs['benchmark_identifier']
    benchmark = brainscore.load_benchmark(
        'EWoK-core-1.0-logprobs', root=prepared, output_dir=tmp_path / 'retained',
    )
    result = benchmark(model)
    assert result.attrs['records_retained'] is True
    assert (tmp_path / 'retained').stat().st_mode & 0o777 == 0o700
    assert (tmp_path / 'retained/inputs_outputs').exists()


def test_multichannel_trace_subject_keeps_dispatch_and_session_contract():
    channels = ['generation_request', 'image_generation_request', 'conditional_logprob_request']
    candidate = build_trace_subject(
        'multi', provider=lambda request: {'text': request['value']},
        parse=str, provenance={}, input_channels=channels,
    )
    assert candidate.in_channels == set(channels)
    events = [StreamEvent(channel, {'value': channel}, None) for channel in channels]
    session = InMemorySession(events)
    session.requested_output_channels = {'response_trace'}
    candidate.interact(session)
    assert [event.payload['answer'] for event in session.emitted] == channels
    # Creating another subject must not change the first subject's declarations.
    other = make_subject(provider)
    assert other.in_channels == {'generation_request'}
    assert candidate.in_channels == set(channels)
