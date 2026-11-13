"""Reject incomparable settings and unsupported requests before model inference."""
import json
from pathlib import Path

import base64

import pandas as pd
import pytest

import brainscore
from brainscore_core.compatibility import CompatibilityError, check_channel_compatibility
from brainscore_core.contract import Subject
from brainscore_core.score_status import score_or_na
from brainscore_core.streaming import Session, StreamEvent
from brainscore.data.preparation import prepare_dataset
from brainscore.experiments import Experiment, RecordInputsOutputs
from brainscore.harnesses.coggym import CogGymRunner

from .test_coggym import PIXEL, image_part, reference, make_subject, reply_with_trial
from .test_coggym_benchmark import scored_reference
from .test_ewok import native, prepared, likelihood_subject, NativeModel, EWoKProvider

pytestmark = pytest.mark.unit


_MEDIA = [
    ('Aboody2025Inferring/exp1', ['image'], ('text', 'vision'), {'vision:image'}),
    ('Fu2025Hierarchical/exp1', ['image', 'video'], ('text', 'vision', 'video'),
     {'vision:image', 'vision:video'}),
]


def _media_benchmark(reference, tmp_path, experiment, content):
    root, cli = reference
    (root / 'evaluation/public_manifest.json').write_text(json.dumps({
        'experiments': [{'path': experiment}],
    }))
    clip = tmp_path / 'clip.mp4'
    clip.write_bytes(b'not decoded here')
    parts = {'image': image_part(), 'video': {'type': 'video', 'path': str(clip), 'mime_type': 'video/mp4'}}
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'look'},
                                             *[parts[kind] for kind in content]]}]
    cli.build_messages = lambda *args: ('instructions', messages)
    benchmark = brainscore.load_benchmark(f'CogGym.{experiment.replace("/", ".")}', checkout=root,
                                          reset=lambda repetition: None)
    return root, clip, benchmark


@pytest.mark.parametrize('experiment,content,modalities,media', _MEDIA)
def test_subject_taking_every_input_has_a_full_match(
    reference, tmp_path, experiment, content, modalities, media,
):
    """Every part reaches the reply, in CogGym's order; the match is full."""
    root, clip, benchmark = _media_benchmark(reference, tmp_path, experiment, content)
    prompts = []
    subject = make_subject(lambda prompt: prompts.append(prompt) or '50', modalities)
    match = check_channel_compatibility(subject, benchmark)
    assert match.kind == 'full' and not match.dropped_inputs
    runner = CogGymRunner(
        root, experiment=experiment, model='fixture', modalities=['text', *content],
    )
    Experiment(
        subject=subject,
        protocol=runner.protocol(reset=lambda repetition: None),
        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'media',
    ).run()
    assert len(prompts) == 3
    assert [kind for kind, _ in prompts[0].parts] == ['text', 'text', *content]
    shown = dict(zip(prompts[0].names, (value for _, value in prompts[0].parts)))
    assert Path(shown['image_path']).read_bytes() == base64.b64decode(PIXEL)
    if 'video' in content:
        assert shown['video_path'] == str(clip)


@pytest.mark.parametrize('experiment,content,modalities,media', _MEDIA)
def test_text_only_subject_runs_with_the_media_dropped_and_is_stamped(
    reference, scored_reference, tmp_path, experiment, content, modalities, media,
):
    """A subject without the media form gets no event on it and a prompt without it."""
    root, clip, benchmark = _media_benchmark(reference, tmp_path, experiment, content)
    prompts = []
    text_subject = make_subject(lambda prompt: prompts.append(prompt) or '50')
    match = check_channel_compatibility(text_subject, benchmark)
    assert match.kind == 'partial' and match.dropped_inputs == media
    score = benchmark(text_subject)
    assert len(prompts) == 15
    for prompt in prompts:
        assert [kind for kind, _ in prompt.parts] == ['text', 'text']
        assert prompt.names == ('instruction', 'text')
    assert score.attrs['compatibility']['match'] == 'partial'
    assert score.attrs['compatibility']['dropped_inputs'] == sorted(media)
    assert 'instruction' in score.attrs['compatibility']['consumed_inputs']
    # A protocol run through Experiment carries the same label.
    protocol = benchmark.protocol(model='fixture', reset=lambda repetition: None)
    result = Experiment(subject=text_subject, protocol=protocol,
                        output_dir=tmp_path / 'partial').run()
    assert result.value.attrs['compatibility']['match'] == 'partial'


def test_text_only_subject_receives_no_picture_event(reference, scored_reference, tmp_path):
    root, clip, benchmark = _media_benchmark(
        reference, tmp_path, 'Aboody2025Inferring/exp1', ['image'])
    events = []
    score = benchmark(_Recorder(events))
    assert events and 'vision:image' not in events
    assert {'instruction', 'text'} >= set(events)
    # The label rides on the score of a direct run and of a protocol run.
    for labelled in (score, Experiment(
            subject=_Recorder([]), output_dir=tmp_path / 'protocol',
            protocol=benchmark.protocol(model='recorder', reset=lambda repetition: None)).run().value):
        assert labelled.attrs['compatibility']['match'] == 'partial'
        assert labelled.attrs['compatibility']['dropped_inputs'] == ['vision:image']


class _Recorder(Subject):
    """A contract subject that records the channel of every input event."""

    identifier = 'recorder'
    in_channels = frozenset({'instruction', 'text'})
    out_channels = frozenset({'behavior:text'})

    def __init__(self, channels, in_channels=None):
        self.channels = channels
        if in_channels is not None:
            self.in_channels = in_channels

    def participate(self, session):
        # Every shipped subject keeps only the streams it declares (rule 5.5).
        session.declare_inputs(self.in_channels)
        while (event := session.next_input()) is not None:
            self.channels.append(event.channel)
            session.respond(StreamEvent('behavior:text', '50', event.t_ms, dict(event.meta)))

    def reset(self):
        pass


@pytest.mark.parametrize('experiment,content,modalities,media', _MEDIA)
def test_subject_with_no_shared_input_or_no_text_answer_is_still_na(
    reference, tmp_path, experiment, content, modalities, media,
):
    """Only an incompatible pair is N/A, before any model call."""
    root, clip, benchmark = _media_benchmark(reference, tmp_path, experiment, content)
    prompts = []
    capture = lambda prompt: prompts.append(prompt) or '50'
    # Takes none of the inputs the experiment sends (instruction, text, pictures).
    no_input = _Recorder([], in_channels=frozenset({'audio'}))
    with pytest.raises(CompatibilityError):
        benchmark(no_input)
    assert score_or_na(benchmark, no_input).attrs['status'] == 'N/A'
    # Cannot answer on behavior:text.
    no_text = brainscore.load_model('chance-baseline')
    with pytest.raises(CompatibilityError):
        benchmark(no_text)
    assert prompts == [] and no_input.channels == []


def test_coggym_parts_keep_their_order_within_one_trial(reference, tmp_path):
    """Several texts and pictures in one trial reach the reply in CogGym's order."""
    root, cli = reference
    (root / 'evaluation/public_manifest.json').write_text(json.dumps({
        'experiments': [{'path': 'Aboody2025Inferring/exp1'}],
    }))
    other = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC'
    content = [{'type': 'text', 'text': 'first'}, image_part(), {'type': 'text', 'text': 'second'},
               image_part(other), image_part(), {'type': 'text', 'text': 'third'}]
    cli.build_messages = lambda *args: ('instructions', [{'role': 'user', 'content': content}])
    prompts = []
    subject = make_subject(lambda prompt: prompts.append(prompt) or 'ok', ('text', 'vision'))
    runner = CogGymRunner(root, experiment='Aboody2025Inferring/exp1', model='fixture',
                          modalities=['text', 'image'], trial_ids=['trial-0'])
    Experiment(subject=subject, protocol=runner.protocol(reset=lambda repetition: None),
               output_dir=tmp_path / 'run').run()
    (prompt,) = prompts
    assert prompt.names == ('instruction', 'text', 'image_path', 'text_2', 'image_path_2',
                            'image_path_3', 'text_3')
    assert [kind for kind, _ in prompt.parts] == ['text', 'text', 'image', 'text', 'image',
                                                  'image', 'text']
    assert [value for kind, value in prompt.parts if kind == 'text'] == [
        'instructions', 'first', 'second', 'third']
    images = [Path(value).read_bytes() for kind, value in prompt.parts if kind == 'image']
    assert images == [base64.b64decode(PIXEL), base64.b64decode(other), base64.b64decode(PIXEL)]


def test_coggym_reference_profile_reset_and_leaderboard_policy(scored_reference, tmp_path):
    root, _ = scored_reference
    calls, resets = [], []
    subject = make_subject(lambda prompt: calls.append(prompt) or reply_with_trial(prompt))
    # The caller gives the provider's reset to the benchmark; it is never read
    # from the subject (Contract Rule 2.2).
    benchmark = brainscore.load_benchmark(
        'CogGym.Hu2023Fine.exp1', checkout=root, output_dir=tmp_path / 'default',
        reset=resets.append,
    )
    score = benchmark(subject)
    assert benchmark.identifier == 'CogGym.Hu2023Fine.exp1'
    assert resets == [1, 2, 3, 4, 5]
    assert len(calls) == 15 and all(call.response.max_tokens == 512 for call in calls)
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

    def respond(prompt):
        if outcome == 'interrupted':
            raise KeyboardInterrupt('fixture interrupt')
        return reply_with_trial(prompt)

    monkeypatch.setattr(module.tempfile, 'TemporaryDirectory', directory)
    if outcome == 'undefined_score':
        analysis.analyze = lambda *args, **kwargs: ([{
            'model': 'fixture', 'experiment': 'Hu2023Fine/exp1', 'r2': None,
        }], [], [])
    benchmark = brainscore.load_benchmark(
        'CogGym.Hu2023Fine.exp1', checkout=root, reset=lambda repetition: None,
    )
    subject = make_subject(respond)
    if outcome == 'success':
        score = benchmark(subject)
        assert float(score) == 0.375
        assert score.attrs['run_directory'] is None
        assert score.attrs['records_retained'] is False
    else:
        with pytest.raises(KeyboardInterrupt if outcome == 'interrupted' else ValueError):
            benchmark(subject)
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
def test_ewok_subject_without_the_text_input_rejected_without_calls(prepared, mode):
    """A subject that cannot take the text is N/A before any part runs."""
    from brainscore import BrainScoreModel
    calls = []
    subject = BrainScoreModel('pictures-only', supported_modalities={'vision'},
                                reply=lambda prompt: calls.append(prompt) or '1',
                                likelihood=lambda prompt, options: calls.append(prompt) or {})
    benchmark = brainscore.load_benchmark(f'EWoK-core-1.0-{mode}', root=prepared)
    with pytest.raises(CompatibilityError):
        benchmark(subject)
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
    subject = likelihood_subject((lambda prompt, candidates: {c: float('nan') for c in candidates})
                                 if fail else EWoKProvider(NativeModel()))
    if fail:
        with pytest.raises(ValueError):
            benchmark(subject)
    else:
        result = benchmark(subject)
        assert result.attrs['run_directory'] is None
        assert result.attrs['records_retained'] is False
        assert result.attrs['leaderboard_eligible'] is benchmark.leaderboard_eligible is False
    assert len(created) == 1 and not created[0].exists()


def test_ewok_explicit_records_private_and_score_uses_resolved_id(prepared, tmp_path, monkeypatch):
    monkeypatch.setenv('BRAINSCORE_EWOK_ROOT', str(prepared))
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path / 'cache'))
    model = likelihood_subject(EWoKProvider(NativeModel()))
    score = brainscore.score(model, 'EWoK-core-1.0-logprobs', check_mem=False)
    assert '-data-' in score.attrs['benchmark_identifier']
    benchmark = brainscore.load_benchmark(
        'EWoK-core-1.0-logprobs', root=prepared, output_dir=tmp_path / 'retained',
    )
    result = benchmark(model)
    assert result.attrs['records_retained'] is True
    assert (tmp_path / 'retained').stat().st_mode & 0o777 == 0o700
    assert (tmp_path / 'retained/inputs_outputs').exists()


def test_the_library_channels_are_gone():
    """External libraries run on the common channels only (Contract Rules 3.8, 3.9)."""
    import importlib.util
    from brainscore_core import io_catalog
    removed = ('generation_request', 'image_generation_request', 'video_generation_request',
               'conditional_logprob_request', 'context_choice_request', 'response_trace')
    assert not any(io_catalog.has(name) for name in removed)
    assert importlib.util.find_spec('brainscore.model_helpers.response_trace') is None
