"""Offline adapter checks. Reference-evaluator qualification is a separate run."""
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from brainscore.experiments import (
    Ablate, Experiment, RecordActivity, RecordInputsOutputs, TorchInstrumentation,
    compare_outputs, replay_sessions,
)
from brainscore.harnesses import coggym
from brainscore.model_helpers.response_trace import build_trace_subject
from brainscore_core.contract import Subject

pytestmark = pytest.mark.unit


@pytest.fixture
def reference(tmp_path, monkeypatch):
    """A provider-call seam, not a substitute for CogGym's scientific evaluator."""
    folder = tmp_path / 'evaluation'
    folder.mkdir()
    (folder / 'public_manifest.json').write_text(json.dumps({
        'experiments': [{'path': 'Hu2023Fine/exp1'}],
    }))
    trials = [SimpleNamespace(id=f'trial-{index}', queries=[]) for index in range(3)]
    experiment = SimpleNamespace(human_data_mean={'secret': 'never pass to provider'})

    def prompt(trial, experiment):
        return 'system', [{'role': 'user', 'content': trial.id}]

    def execute(experiment, selected, provider, model, temperature, max_tokens):
        results = []
        for trial in selected:
            try:
                system, messages = cli.build_messages(trial, experiment)
                raw = provider.complete_with_metadata(
                    system, messages, model, temperature, max_tokens,
                )
                results.append({'trial_id': trial.id, 'response': raw['text'],
                                'scores': {'answer': {'scorable': True}}})
            except Exception as error:
                results.append({'trial_id': trial.id, 'error': str(error)})
        return results

    cli = SimpleNamespace(
        load_experiment=lambda path: experiment,
        _load_selection_map=lambda path: {},
        _selected_trials=lambda exp, selection, use_all: trials,
        build_messages=prompt, _run_experiment=execute,
        aggregate_experiment=lambda rows: {'n_trials': len(rows)},
    )
    analysis = SimpleNamespace(analyze=lambda *args, **kwargs: ([], [], []))
    monkeypatch.setattr(coggym, '_load_reference', lambda checkout: (cli, analysis))
    monkeypatch.setattr(coggym, '_check_checkout', lambda checkout: None)
    return tmp_path, cli


def make_subject(provider):
    return build_trace_subject(
        'fixture', provider=provider, parse=str,
        provenance={'purpose': 'adapter contract test'},
    )


def provider(request):
    return {'text': request['messages'][0]['content'], 'reasoning': None, 'token_usage': None}


def test_requests_preserve_reference_arguments_without_scoring_targets(reference, tmp_path):
    root, _ = reference
    calls, resets = [], []

    def capture(request):
        calls.append(request)
        return provider(request)

    runner = coggym.CogGymRunner(root, model='fixture-model', repetitions=2)
    result = Experiment(
        subject=make_subject(capture), protocol=runner.protocol(reset=resets.append),
        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'run',
    ).run()
    assert resets == [1, 2]
    assert len(calls) == 6
    assert all(set(call) == {'system', 'messages', 'model', 'temperature', 'max_tokens'}
               for call in calls)
    assert calls[0] == calls[3] == {
        'system': 'system', 'messages': [{'role': 'user', 'content': 'trial-0'}],
        'model': 'fixture-model', 'temperature': 1.0, 'max_tokens': 512,
    }
    assert all(len(call['messages']) == 1 for call in calls)
    assert result.manifest['status'] == 'complete'
    # Each provider call is recorded as a session input event, not a method call.
    requests = [event['payload'] for event in result.record.events()
                if event['kind'] == 'input']
    assert requests[0].t_ms is None
    assert requests[0].meta['matching_trial_ids'] == ['trial-0']
    assert requests[3].meta['repetition'] == 2
    artifacts = result.manifest['artifacts']
    assert sum(a['producer'] == 'CogGym evaluator via UMI adapter' for a in artifacts) == 3


def test_subset_preserves_canonical_order_and_is_labelled(reference):
    root, _ = reference
    runner = coggym.CogGymRunner(root, model='fixture', trial_ids=['trial-2', 'trial-0'])
    assert runner.trial_ids == ('trial-0', 'trial-2')
    assert runner.describe()['scope'] == 'smoke_subset'
    assert runner.describe()['canonical_trial_count'] == 3


@pytest.mark.parametrize('kwargs', [
    {'temperature': float('nan')}, {'temperature': -1}, {'max_tokens': 0},
    {'repetitions': True}, {'trial_ids': []}, {'trial_ids': ['missing']},
    {'trial_ids': ['trial-0', 'trial-0']}, {'experiment': '../private'},
    {'modalities': []}, {'modalities': ['audio']}, {'model': ''},
])
def test_invalid_setup_is_rejected(reference, kwargs):
    root, _ = reference
    with pytest.raises(ValueError):
        coggym.CogGymRunner(root, **{'model': 'fixture', **kwargs})


def test_media_requires_explicit_provider_support(reference):
    root, cli = reference
    cli.build_messages = lambda *args: ('system', [{
        'role': 'user', 'content': [{'type': 'image', 'data': 'base64'}],
    }])
    with pytest.raises(ValueError, match='modalities'):
        coggym.CogGymRunner(root, model='fixture')
    assert coggym.CogGymRunner(root, model='fixture', modalities=['text', 'image'])


def test_upstream_caught_error_is_saved_and_fails_the_experiment(reference, tmp_path):
    root, _ = reference

    def fail(request):
        raise RuntimeError('provider failed')

    runner = coggym.CogGymRunner(root, model='fixture')
    output = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='incomplete or failed'):
        Experiment(
            subject=make_subject(fail), protocol=runner.protocol(reset=lambda repetition: None),
            tools=[RecordInputsOutputs()], output_dir=output,
        ).run()
    assert json.loads((output / 'experiment.json').read_text())['status'] == 'failed'
    raw = json.loads((output / 'coggym/run-001.json').read_text())
    assert raw['umi']['coverage']['error_trials'] == 3
    assert raw['trials'][0]['error'] == 'provider failed'
    assert raw['umi']['status'] == 'failed'


def test_missing_results_cannot_pass(reference, tmp_path):
    root, cli = reference
    cli._run_experiment = lambda *args: []
    runner = coggym.CogGymRunner(root, model='fixture')
    with pytest.raises(RuntimeError, match='incomplete or failed'):
        runner.run(SimpleNamespace(complete_with_metadata=lambda *a: None),
                   output_dir=tmp_path / 'missing', reset=lambda repetition: None)


@pytest.mark.parametrize('stage', ['summary', 'analysis'])
def test_scoring_failure_preserves_trials_and_registered_hashes(reference, tmp_path, stage):
    root, cli = reference
    runner = coggym.CogGymRunner(root, model='fixture')

    def overflow(*args, **kwargs):
        raise OverflowError('reference arithmetic overflow')

    if stage == 'summary':
        cli.aggregate_experiment = overflow
    else:
        runner._analysis.analyze = overflow
    output = tmp_path / 'scoring-failed'
    with pytest.raises(OverflowError, match='reference arithmetic overflow'):
        Experiment(
            subject=make_subject(provider), protocol=runner.protocol(reset=lambda repetition: None),
            tools=[RecordInputsOutputs()], output_dir=output,
        ).run()
    manifest = json.loads((output / 'experiment.json').read_text())
    assert manifest['status'] == 'failed'
    saved = json.loads((output / 'coggym/run-001.json').read_text())
    assert [t['response'] for t in saved['trials']] == ['trial-0', 'trial-1', 'trial-2']
    assert saved['umi']['coverage']['returned_trials'] == 3
    if stage == 'summary':
        assert saved['summary'] is None
        assert saved['umi']['summary_error']['type'] == 'OverflowError'
    else:
        assert saved['umi']['summary_status'] == 'complete'
        assert json.loads((output / 'coggym/analysis-error.json').read_text())['type'] == 'OverflowError'
    for artifact in manifest['artifacts']:
        assert hashlib.sha256((output / artifact['path']).read_bytes()).hexdigest() == artifact['sha256']


def test_missing_reasoning_metadata_is_not_fabricated(reference, tmp_path):
    root, _ = reference
    runner = coggym.CogGymRunner(root, model='fixture')
    with pytest.raises(RuntimeError, match='incomplete or failed'):
        Experiment(
            subject=make_subject(lambda request: {'text': 'answer'}),
            protocol=runner.protocol(reset=lambda repetition: None),
            output_dir=tmp_path / 'invalid',
        ).run()


def test_real_hooks_record_ablate_restore_and_replay(reference, tmp_path):
    root, _ = reference
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))

    def generate(request):
        with torch.inference_mode():
            value = model(torch.tensor([1., 2.])).tolist()
        return {'text': str(value), 'reasoning': None, 'token_usage': None}

    subject = make_subject(generate)
    runner = coggym.CogGymRunner(root, model='fixture')

    def run(name, intervention=False):
        tools = [RecordInputsOutputs()]
        if intervention:
            tools.append(Ablate(['0']))
        tools.append(RecordActivity(['0']))
        result = Experiment(
            subject=subject, protocol=runner.protocol(reset=lambda repetition: None),
            tools=tools, instrumentation=TorchInstrumentation(model),
            output_dir=tmp_path / name,
        ).run()
        assert not model[0]._forward_hooks
        return result

    baseline = run('baseline')
    changed = run('ablated', intervention=True)
    restored = run('restored')
    base_rows = json.loads((baseline.directory / 'coggym/run-001.json').read_text())['trials']
    changed_rows = json.loads((changed.directory / 'coggym/run-001.json').read_text())['trials']
    restored_rows = json.loads((restored.directory / 'coggym/run-001.json').read_text())['trials']
    assert base_rows == restored_rows
    assert base_rows[0]['response'] == '[1.0, 2.0]'
    assert changed_rows[0]['response'] == '[0.0, 0.0]'
    events = list(changed.record.events())
    activity = [event for event in events if event['kind'] == 'activity']
    assert len(activity) == 3
    assert all((event['payload']['value']['array'] == 0).all() for event in activity)
    replay = Experiment(
        subject=subject,
        protocol=replay_sessions(baseline.directory),
        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'replay',
    ).run()
    assert compare_outputs(baseline.directory, replay.directory)['equal']


def test_hook_cleanup_after_caught_provider_failure(reference, tmp_path):
    root, _ = reference
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))

    def fail_after_forward(request):
        model(torch.ones(2))
        raise ValueError('failure after model execution')

    runner = coggym.CogGymRunner(root, model='fixture')
    with pytest.raises(RuntimeError):
        Experiment(
            subject=make_subject(fail_after_forward),
            protocol=runner.protocol(reset=lambda repetition: None),
            tools=[RecordInputsOutputs(), Ablate(['0']), RecordActivity(['0'])],
            instrumentation=TorchInstrumentation(model), output_dir=tmp_path / 'failed',
        ).run()
    assert not model[0]._forward_hooks


@pytest.mark.parametrize('head,edits', [('wrong-revision', ''), (coggym.REFERENCE_REVISION, ' M evaluation/cli.py')])
def test_changed_reference_checkout_is_rejected(monkeypatch, tmp_path, head, edits):
    def git(command, **kwargs):
        return head if 'rev-parse' in command else edits
    monkeypatch.setattr(coggym.subprocess, 'check_output', git)
    with pytest.raises(ValueError):
        coggym._check_checkout(tmp_path)


# ── Old path vs participate on the SAME subject (Contract Rule 10.1) ──────

def _old_complete_with_metadata(self, system, messages, model, temperature, max_tokens):
    """What the provider did before it ran through participate: subject.process."""
    from brainscore_core.streaming import StreamEvent
    request = dict(system=system, messages=messages, model=model,
                   temperature=temperature, max_tokens=max_tokens)
    trial_ids = self.prompt_trials[coggym._prompt_key(system, messages)]
    event = StreamEvent(
        self.input_channel, request, t_ms=None,
        meta={'experiment': self.experiment, 'repetition': self.repetition,
              'provider_call': self.call_index, 'matching_trial_ids': trial_ids},
    )
    self.call_index += 1
    response = self.subject.process(event)
    if not isinstance(response, StreamEvent) or response.channel != 'response_trace':
        raise TypeError('CogGym needs a response_trace subject; use build_trace_subject')
    raw = response.payload['raw']
    if not isinstance(raw.get('text'), str):
        raise TypeError('CogGym provider output must contain text: str')
    if 'reasoning' not in raw or 'token_usage' not in raw:
        raise ValueError('Provider output needs reasoning and token_usage (None is allowed)')
    return raw


def _counting_provider(streaming):
    """A stateful provider: each answer carries how many calls it has served.

    Identical answers on both paths show the same calls in the same order and
    no subject reset that the old path did not do.
    """
    calls = []

    def answer(request):
        calls.append(request)
        return f"{request['messages'][0]['content']}#{len(calls)}"

    def plain(request):
        return {'text': answer(request), 'reasoning': None, 'token_usage': None}

    def stream(request):
        text = answer(request)
        yield {'text': '', 'final': False, 'reasoning': None, 'token_usage': None}
        yield {'text': text, 'final': True, 'reasoning': None, 'token_usage': None}

    return (stream if streaming else plain), calls


@pytest.mark.parametrize('streaming', [False, True])
def test_same_subject_old_process_path_and_participate_give_identical_trials(
        reference, tmp_path, monkeypatch, streaming):
    root, _ = reference
    resets = []
    provider_fn, calls = _counting_provider(streaming)
    subject = build_trace_subject('fixture', provider=provider_fn, parse=str,
                                  provenance={'purpose': 'old vs new'}, streaming=streaming)
    runner = coggym.CogGymRunner(root, model='fixture', repetitions=2)

    def run(name):
        result = Experiment(subject=subject, protocol=runner.protocol(reset=resets.append),
                            tools=[RecordInputsOutputs()], output_dir=tmp_path / name).run()
        return [json.loads((result.directory / f'coggym/run-{r:03d}.json').read_text())['trials']
                for r in (1, 2)]

    new = run('participate')
    with monkeypatch.context() as patch:
        patch.setattr(coggym._SubjectProvider, 'complete_with_metadata',
                      _old_complete_with_metadata)
        old = run('process')
    assert [trial['response'] for rep in new for trial in rep] == [
        f'trial-{i}#{n}' for n, i in enumerate([0, 1, 2, 0, 1, 2], start=1)]
    assert [trial['response'] for rep in old for trial in rep] == [
        f'trial-{i}#{n}' for n, i in enumerate([0, 1, 2, 0, 1, 2], start=7)]
    # Same answers apart from the running call count of the shared provider.
    strip = lambda runs: [[{**t, 'response': t['response'].split('#')[0]} for t in rep]
                          for rep in runs]
    assert strip(new) == strip(old)
    assert resets == [1, 2, 1, 2]


class _SessionCounter(Subject):
    """Answers with how many requests it has seen in the current session."""

    identifier = "session-counter"
    in_channels = frozenset({"generation_request"})
    out_channels = frozenset({"response_trace"})

    def participate(self, session):
        from brainscore_core.streaming import StreamEvent
        seen = 0
        while (event := session.next_input()) is not None:
            seen += 1
            raw = {'text': f'{seen}', 'reasoning': None, 'token_usage': None}
            session.respond(StreamEvent('response_trace', {
                'raw': raw, 'answer': raw['text'], 'valid': True, 'parse_error': None,
                'provenance': {}}, event.t_ms, dict(event.meta)))

    def reset(self):
        pass


def test_record_keeps_one_session_per_call_and_replay_rebuilds_them(reference, tmp_path):
    from brainscore.run_record import RunRecord
    root, _ = reference
    runner = coggym.CogGymRunner(root, model='fixture', repetitions=2)
    original = Experiment(subject=_SessionCounter(),
                          protocol=runner.protocol(reset=lambda repetition: None),
                          tools=[RecordInputsOutputs()], output_dir=tmp_path / 'original').run()
    events = list(RunRecord(original.directory).events())
    starts = [e['payload']['id'] for e in events if e['kind'] == 'repetition_start']
    assert starts == ['1.0', '1.1', '1.2', '2.0', '2.1', '2.2']
    assert sum(e['kind'] == 'input' for e in events) == 6
    replay = Experiment(subject=_SessionCounter(), protocol=replay_sessions(original.directory),
                        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'replay').run()
    # Merged sessions would answer 1, 2, 3, ...; one session per call answers 1 each time.
    assert compare_outputs(original.directory, replay.directory)['equal']
