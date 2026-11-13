"""Public lab vocabulary preserves existing plugin and session behavior."""
import numpy as np
import pytest

from brainscore import BrainScoreModel
from brainscore_core.events import StateChange, PerturbationApplied, Perturbation, Selection
from brainscore_core.streaming import FixedSource, Session, Source, StreamEvent
from brainscore_core.streaming_helpers import stimulus_session


def test_session_delivers_inputs_and_collects_responses():
    event = StreamEvent(channel='text', payload='hello', t_ms=0)
    session = Session([event])
    assert session.next_input() is event
    assert session.next_input() is None
    session.respond(event)
    session.respond(event)
    assert session.collect() == [event, event]


def test_a_source_receives_the_response_its_session_collects():
    class Recording(Source):
        def next_input(self):
            return None

        def respond(self, event):
            self.response = event

    event = StreamEvent(channel='behavior', payload='yes', t_ms=None)
    source = Recording()
    session = Session(source)
    session.respond(event)
    assert source.response is event
    assert session.collect() == [event]


def test_intervention_cleanup_on_reset():
    active = []
    def intervene(change):
        active.append('installed')
        return PerturbationApplied(handle_id='one', target=change.target,
                                   perturbation=change.perturbation), active.clear

    subject = BrainScoreModel(identifier='intervention', intervention=intervene)
    from brainscore_core.streaming_helpers import apply_state_change
    apply_state_change(subject, StateChange(kind='ablation', target=Selection(layer='layer'),
                                            perturbation=Perturbation(kind='zero')))
    assert active == ['installed']
    subject.reset()
    assert active == []


@pytest.mark.parametrize('modality,column', [('vision', 'image_path'), ('text', 'sentence'),
                                          ('audio', 'audio_path'), ('video', 'video_path')])
def test_extractor_and_buffered_session_work_across_modalities(modality, column):
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    def extract(stimuli, *, layers):
        return NeuroidAssembly(np.ones((len(stimuli), 2)), dims=('presentation', 'neuroid'),
                            coords={'layer': ('neuroid', [layers[0]] * 2),
                                    'neuroid_id': ('neuroid', [0, 1])})
    subject = BrainScoreModel(identifier='extractor', extractor={modality: extract},
                              region_layer_map={'region': 'layer'})
    session = stimulus_session(StimulusSet({column: ['one', 'two']}), ('neural:region',))
    subject.participate(session)
    np.testing.assert_array_equal(session.collect().values, np.ones((2, 2)))


def test_answer_gives_one_choice_for_each_stimulus():
    from brainscore_core.contract import TaskContext
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    def answer(stimulus_row, instruction, label_set):
        return label_set[0] if stimulus_row.sentence == 'first' else label_set[1]
    from brainscore_core.streaming_helpers import behavioral_response
    stimuli = StimulusSet({'sentence': ['first', 'second'], 'stimulus_id': ['a', 'b']})
    stimuli.identifier = 'answer-choices'
    task = TaskContext(task_type='label', instruction='choose', label_set=['yes', 'no'])
    subject = BrainScoreModel(identifier='answer', in_channels={'text'}, answer=answer)
    output = behavioral_response(subject, stimuli, task)
    assert output['stimulus_id'].values.tolist() == ['a', 'b']
    assert output['raw_answer'].values.tolist() == ['yes', 'no']
    assert output.values.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_source_subclasses_can_extend_responses_with_super():
    class Counting(FixedSource):
        def respond(self, event):
            self.calls = getattr(self, 'calls', 0) + 1
            super().respond(event)

    event = StreamEvent(channel='behavior', payload='answer', t_ms=0)
    source = Counting()
    session = Session(source)
    session.respond(event)
    assert session.collect() == [event]
    assert source.calls == 1


# -- Human-facing guidance names the canonical parameters --------------------

def test_extractor_that_is_not_a_mapping_names_extractor_mapping():
    with pytest.raises(TypeError, match=r'extractor must be a mapping \{modality: extractor\}'):
        BrainScoreModel(identifier='shared', extractor=lambda stimuli, layers: None)


def test_non_callable_extractor_names_extractor_signature():
    with pytest.raises(TypeError, match=r'extractor\(stimuli, layers=\.\.\.\)'):
        BrainScoreModel(identifier='bad', extractor={'vision': 'not callable'})


def _text_task(**fields):
    from brainscore_core.contract import TaskContext
    return TaskContext(task_type='label', **fields)


def test_unviable_behavioral_task_names_answer_parameter():
    subject = BrainScoreModel(identifier='no-path', in_channels={'text'})
    with pytest.raises(ValueError, match='Paths tried') as error:
        subject.start_task(_text_task())
    assert 'generation needs answer=' in str(error.value)
    assert 'readout needs TaskContext.fitting_stimuli' in str(error.value)


# -- Core guide teaching example ---------------------------------------------

def test_guide_echo_subject_fixture_returns_labeled_behavioral_assembly():
    """Self-contained copy of the UMI_MIGRATION.md EchoSubject (no docs needed)."""
    from brainscore_core import Subject
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly

    class EchoSubject(Subject):
        identifier = 'echo'
        in_channels = {'text'}
        out_channels = {'behavior'}
        required_channels = {'text'}

        def participate(self, session):
            while (event := session.next_input()) is not None:
                answer = BehavioralAssembly(
                    [event.payload],
                    coords={'stimulus_id': ('presentation', [event.meta['stimulus_id']]),
                            'sentence': ('presentation', [event.payload])},
                    dims=['presentation'])
                session.respond(StreamEvent(channel='behavior', payload=answer,
                                            t_ms=event.t_ms, meta=dict(event.meta)))

    session = Session([StreamEvent(channel='text', payload='hello', t_ms=0.0,
                            meta={'stimulus_id': 's1'})], ['behavior'])
    EchoSubject().participate(session=session)
    (response,) = [event.payload for event in session.emitted]
    assert isinstance(response, BehavioralAssembly)
    assert response['stimulus_id'].values.tolist() == ['s1']
    assert response['sentence'].values.tolist() == ['hello']
    assert response.values.tolist() == ['hello']


# -- Missing policy names the canonical parameter ----------------------------

@pytest.mark.parametrize('event', [{}, {'text': 'hello'}])
def test_missing_policy_names_policy_parameter(event):
    subject = BrainScoreModel(identifier='no-policy')
    with pytest.raises(NotImplementedError, match=r'BrainScoreModel\(policy=\.\.\.\)') as error:
        subject.process(event)
    assert 'has no policy' in str(error.value)
