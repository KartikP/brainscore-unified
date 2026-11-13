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
from brainscore import BrainScoreModel
from brainscore.harnesses import coggym
from brainscore_core.contract import Subject
from brainscore_core.responses import Text

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


def make_subject(reply, in_channels=('text',)):
    """A normal model with a ``reply`` part: the common channels, no CogGym channel."""
    return BrainScoreModel('fixture', in_channels=set(in_channels), reply=reply)


def reply_with_trial(prompt):
    """Answer with the trial's own text (the last part, after the instruction)."""
    return prompt.parts[-1][1]


def text_inputs(record):
    """The recorded input events that carry a trial's text."""
    return [event for event in record.events()
            if event['kind'] == 'input' and event['payload'].channel == 'text']


def test_requests_preserve_reference_arguments_without_scoring_targets(reference, tmp_path):
    root, _ = reference
    prompts, resets = [], []

    def capture(prompt):
        prompts.append(prompt)
        return reply_with_trial(prompt)

    runner = coggym.CogGymRunner(root, model='fixture-model', repetitions=2)
    result = Experiment(
        subject=make_subject(capture), protocol=runner.protocol(reset=resets.append),
        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'run',
    ).run()
    assert resets == [1, 2]
    assert len(prompts) == 6
    # The system text is the instruction; the user message is the stimulus part;
    # CogGym's temperature and length are the settings of the Text response.
    assert prompts[0] == prompts[3]
    assert prompts[0].parts == (('text', 'system'), ('text', 'trial-0'))
    assert prompts[0].names == ('instruction', 'text')
    assert prompts[0].response == Text(max_tokens=512, temperature=1.0)
    assert prompts[0].history == ()
    assert not any('never pass' in prompt.text() for prompt in prompts)
    assert result.manifest['status'] == 'complete'
    # Each provider call is recorded as a session input event, not a method call.
    requests = text_inputs(result.record)
    assert [event['payload'].payload for event in requests[:3]] == ['trial-0', 'trial-1', 'trial-2']
    assert requests[0]['payload'].meta['stimulus_id'] == 'trial-0'
    assert [event['repetition_id'] for event in requests] == ['1.0', '1.1', '1.2', '2.0', '2.1', '2.2']
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


PIXEL = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNgAAAAAgABSK+kcQAAAABJRU5ErkJggg=='


def image_part(data=PIXEL):
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': data}}


def test_media_requires_explicit_provider_support(reference):
    root, cli = reference
    cli.build_messages = lambda *args: ('system', [{
        'role': 'user', 'content': [image_part()],
    }])
    with pytest.raises(ValueError, match='modalities'):
        coggym.CogGymRunner(root, model='fixture')
    assert coggym.CogGymRunner(root, model='fixture', modalities=['text', 'image'])


def test_upstream_caught_error_is_saved_and_fails_the_experiment(reference, tmp_path):
    root, _ = reference

    def fail(prompt):
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
            subject=make_subject(reply_with_trial), protocol=runner.protocol(reset=lambda repetition: None),
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


class _ReasoningReplier(Subject):
    """Answers on behavior:text and puts reasoning and token counts in the answer's meta."""

    identifier = 'reasoning-replier'
    in_channels = frozenset({'instruction', 'text'})
    out_channels = frozenset({'behavior:text'})

    def participate(self, session):
        from brainscore_core.streaming import StreamEvent
        while (event := session.next_input()) is not None:
            session.respond(StreamEvent('behavior:text', event.payload, event.t_ms, {
                'reasoning': f'thought about {event.payload}', 'token_usage': {'output': 3}}))

    def reset(self):
        pass


@pytest.mark.parametrize('subject,reasoning,usage', [
    (make_subject(lambda prompt: 'answer'), None, None),
    (_ReasoningReplier(), 'thought about trial-0', {'output': 3}),
])
def test_reasoning_and_token_usage_come_from_the_answer_meta_only(
        reference, tmp_path, subject, reasoning, usage):
    """A subject without them gives None; they are never made up (rule 3.9)."""
    root, cli = reference
    saved = []

    def execute(experiment, selected, provider, model, temperature, max_tokens):
        rows = []
        for trial in selected:
            raw = provider.complete_with_metadata(*cli.build_messages(trial, experiment),
                                                  model, temperature, max_tokens)
            saved.append(raw)
            rows.append({'trial_id': trial.id, 'response': raw['text'],
                         'scores': {'answer': {'scorable': True}}})
        return rows

    cli._run_experiment = execute
    runner = coggym.CogGymRunner(root, model='fixture')
    Experiment(subject=subject, protocol=runner.protocol(reset=lambda repetition: None),
               output_dir=tmp_path / 'run').run()
    assert set(saved[0]) == {'text', 'reasoning', 'token_usage'}
    assert saved[0]['reasoning'] == reasoning
    assert saved[0]['token_usage'] == usage


def test_real_hooks_record_ablate_restore_and_replay(reference, tmp_path):
    root, _ = reference
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))

    def generate(prompt):
        with torch.inference_mode():
            value = model(torch.tensor([1., 2.])).tolist()
        return str(value)

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

    def fail_after_forward(prompt):
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


# ── One answer per call, in call order (Contract Rule 10.1) ──────────────

def _counting_reply():
    """A stateful reply: each answer carries how many calls it has served.

    The answers show the same calls in the same order, and no subject reset
    between calls inside a CogGym repetition.
    """
    calls = []

    def reply(prompt):
        calls.append(prompt)
        return f'{reply_with_trial(prompt)}#{len(calls)}'

    return reply, calls


def test_stateful_reply_sees_every_call_once_in_order(reference, tmp_path):
    root, _ = reference
    resets = []
    reply, calls = _counting_reply()
    runner = coggym.CogGymRunner(root, model='fixture', repetitions=2)
    result = Experiment(subject=make_subject(reply), protocol=runner.protocol(reset=resets.append),
                        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'run').run()
    runs = [json.loads((result.directory / f'coggym/run-{r:03d}.json').read_text())['trials']
            for r in (1, 2)]
    assert [trial['response'] for rep in runs for trial in rep] == [
        f'trial-{i}#{n}' for n, i in enumerate([0, 1, 2, 0, 1, 2], start=1)]
    assert resets == [1, 2]


class _TwoAnswers(Subject):
    """Sends a partial answer and then the full one on behavior:text; the session refuses it."""

    identifier = 'two-answers'
    in_channels = frozenset({'instruction', 'text'})
    out_channels = frozenset({'behavior:text'})

    def participate(self, session):
        from brainscore_core.streaming import StreamEvent
        while (event := session.next_input()) is not None:
            for text in ('', event.payload):
                session.respond(StreamEvent('behavior:text', text, event.t_ms, {}))

    def reset(self):
        pass


def test_more_than_one_answer_per_call_is_refused_not_scored(reference, tmp_path):
    root, _ = reference
    runner = coggym.CogGymRunner(root, model='fixture')
    with pytest.raises(RuntimeError, match='incomplete or failed'):
        Experiment(subject=_TwoAnswers(), protocol=runner.protocol(reset=lambda repetition: None),
                   output_dir=tmp_path / 'run').run()
    saved = json.loads((tmp_path / 'run/coggym/run-001.json').read_text())
    assert 'answered twice' in saved['trials'][0]['error']


class _SessionCounter(Subject):
    """Answers with how many trial texts it has seen in the current session."""

    identifier = "session-counter"
    in_channels = frozenset({"instruction", "text"})
    out_channels = frozenset({"behavior:text"})

    def participate(self, session):
        from brainscore_core.streaming import StreamEvent
        seen = 0
        while (event := session.next_input()) is not None:
            seen += 1
            session.respond(StreamEvent('behavior:text', f'{seen}', event.t_ms, dict(event.meta)))

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
    assert sum(e['kind'] == 'input' and e['payload'].channel == 'text' for e in events) == 6
    replay = Experiment(subject=_SessionCounter(), protocol=replay_sessions(original.directory),
                        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'replay').run()
    # Merged sessions would answer 1, 2, 3, ...; one session per call answers 1 each time.
    assert compare_outputs(original.directory, replay.directory)['equal']
