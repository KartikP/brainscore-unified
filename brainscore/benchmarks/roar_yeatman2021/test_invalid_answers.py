"""ROAR scores an invalid generated answer as wrong, not as the first label."""
import pytest

from brainscore.benchmarks.roar_yeatman2021.benchmark import Yeatman2021LexicalDecision
from brainscore.brainscore_model import BrainScoreModel
from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from tests.benchmark_stand_in import without_init


def _benchmark():
    # Skip __init__: it downloads the ROAR data. __call__ needs only these fields.
    return without_init(
        Yeatman2021LexicalDecision, identifier='Yeatman2021-lexical_decision-text', ceiling=Score(1.0),
        version=1, _train_stimuli=None, stimulus_set=StimulusSet({
            'stimulus_id': ['w1', 'w2', 'p1', 'p2'], 'sentence': ['cat', 'dog', 'florp', 'blick'],
            'image_label': ['real', 'real', 'pseudo', 'pseudo']}))


def test_invalid_answers_count_as_wrong():
    # Always answering 'real' when unsure used to earn credit on every real word.
    replies = {'w1': 'real', 'w2': 'unsure', 'p1': 'pseudo', 'p2': 'unsure'}
    model = BrainScoreModel(identifier='answers', in_channels={'text'},
                            answer=lambda stimulus_row, instruction, label_set:
                            replies[stimulus_row.stimulus_id])
    benchmark, sessions = _benchmark(), []
    participate = model.participate
    model.participate = lambda session: sessions.append(session) or participate(session)
    score = benchmark(model)
    assert float(score.attrs['raw']) == pytest.approx(.5)
    # The count of invalid answers is in the score; each trial keeps its state and reason.
    assert (score.attrs['n_invalid'], score.attrs['n_missing']) == (2, 0)
    assert score.attrs['answer_policy'] == 'invalid: wrong, counted; missing: wrong, counted'
    answer = sessions[0].collect('behavior:choice')
    assert list(answer['response_state'].values) == ['valid', 'invalid', 'valid', 'invalid']
    assert list(answer['raw_answer'].values) == ['real', 'unsure', 'pseudo', 'unsure']
    assert score.attrs['accuracy_real'] == pytest.approx(.5)


def test_session_reading_matches_the_benchmark_score():
    """ROAR reads through behavioral_response; on a BrainScoreModel the score is
    the accuracy of the readout's answers in a session on the same task."""
    import numpy as np
    from brainscore_core.contract import TaskContext
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    words = {'w1': 'cat', 'w2': 'dog', 'w3': 'sun', 'p1': 'florp', 'p2': 'blick', 'p3': 'zund'}

    def extractor(stimuli, layers):
        rng = np.random.RandomState(0)
        table = {sid: rng.randn(4) + (2. if sid.startswith('w') else -2.) for sid in words}
        rows = np.stack([table[sid] for sid in stimuli['stimulus_id']])
        return NeuroidAssembly(rows, dims=['presentation', 'neuroid'], coords={
            **{k: ('presentation', list(stimuli[k])) for k in stimuli.columns},
            'layer': ('neuroid', [layers[0]] * 4), 'neuroid_id': ('neuroid', list(range(4)))})

    def stimuli(ids):
        table = StimulusSet({'stimulus_id': ids, 'sentence': [words[i] for i in ids],
                             'image_label': ['real' if i.startswith('w') else 'pseudo' for i in ids]})
        table.identifier = f'roar-equivalence-{len(ids)}'
        return table

    def model():
        # As the registered readout-only models (CLIP, random ViT, GPT-2) state it.
        return BrainScoreModel(identifier='readout', extractor={'text': extractor},
                               region_layer_map={}, behavioral_readout_layer='L',
                               answer_owners={
                                   'behavior:choice': ('readout', 'argmax')})

    benchmark = _benchmark()
    benchmark._train_stimuli = stimuli(['w1', 'w2', 'p1', 'p2'])
    benchmark.stimulus_set = stimuli(['w3', 'p3', 'w1', 'p1'])

    from brainscore_core.responses import Choice
    task = TaskContext(response=Choice(['real', 'pseudo']), fitting_stimuli=benchmark._train_stimuli,
                       instruction='Is this a real word or a pseudo word?')
    from brainscore_core.streaming_helpers import stimulus_session
    session = stimulus_session(benchmark.stimulus_set, ['behavior:probabilities'], task)
    model().participate(session)
    answered = session.collect('behavior:probabilities')
    assert list(answered['stimulus_id'].values) == list(benchmark.stimulus_set['stimulus_id'])
    choices = list(answered['choice'].values)
    expected_correct = [choices[int(i)] == label for i, label in
                        zip(answered.values.argmax(axis=1), benchmark.stimulus_set['image_label'])]

    score = benchmark(model())
    assert float(score.attrs['raw']) == np.mean(expected_correct)
    assert score.attrs['n_trials'] == len(answered)


def test_a_missing_answer_counts_as_wrong_and_is_counted_apart():
    """A trial with no answer is a missing row (core); ROAR counts it wrong and as missing."""
    from brainscore_core import Subject
    from brainscore_core.streaming import StreamEvent
    replies = {'w1': 'real', 'w2': 'real', 'p1': 'pseudo'}            # p2 is never answered

    class _Skips(Subject):
        identifier = 'skips-one'
        in_channels = {'text'}
        out_channels = {'behavior:choice'}
        required_channels = set()

        def participate(self, session):
            while (event := session.next_input()) is not None:
                stimulus_id = event.meta.get('stimulus_id')
                if stimulus_id in replies:
                    session.respond(StreamEvent('behavior:choice', replies[stimulus_id], event.t_ms,
                                                {'stimulus_id': stimulus_id}))

        def reset(self):
            pass

    score = _benchmark()(_Skips())
    assert float(score.attrs['raw']) == pytest.approx(.75)
    assert (score.attrs['n_trials'], score.attrs['n_valid'], score.attrs['n_invalid'],
            score.attrs['n_missing']) == (4, 3, 0, 1)
    assert score.attrs['result_validity'] == 'no limit stated'


def test_stated_change_a_finite_row_marked_invalid_loses_credit():
    """M9-03, stated change: a custom subject's one-hot row with answer_valid False counted
    as answered before this step (1.0 here); its state is invalid, so it is wrong now (0.75).
    No registered producer sends such a row: theirs are all-NaN when not valid."""
    import numpy as np
    from brainscore_core import Subject
    from brainscore_core.response_formats import response_record
    from brainscore_core.streaming import StreamEvent
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    truth = {'w1': 'real', 'w2': 'real', 'p1': 'pseudo', 'p2': 'pseudo'}

    class _FlagsFirst(Subject):
        identifier = 'flags-first'
        in_channels = {'text'}
        out_channels = {'behavior:choice'}
        required_channels = set()

        def participate(self, session):
            while session.next_input() is not None:
                pass
            ids = list(session.stimuli['stimulus_id'])
            rows = np.array([[1.0, 0.0] if truth[sid] == 'real' else [0.0, 1.0] for sid in ids],
                            dtype=np.float32)
            session.respond(StreamEvent('behavior:choice', BehavioralAssembly(
                rows, dims=['presentation', 'choice'], coords={
                    'stimulus_id': ('presentation', ids), 'choice': ['real', 'pseudo'],
                    'raw_answer': ('presentation', [truth[sid] for sid in ids]),
                    'answer_valid': ('presentation', [sid != 'w1' for sid in ids])},
                attrs={'response': response_record('behavior:choice', 'test')}), None))

        def reset(self):
            pass

    score = _benchmark()(_FlagsFirst())
    assert float(score.attrs['raw']) == pytest.approx(.75)
    assert (score.attrs['n_valid'], score.attrs['n_invalid']) == (3, 1)
