"""Provenance adds metadata without changing the subject/session contract."""
import numpy as np
import pytest

from brainscore import BrainScoreModel
from brainscore_core import TaskContext
from brainscore_core.responses import Text
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.model_helpers.provenance import ProvenanceSubject


def test_declared_subject_records_neural_outputs_with_separate_provenance():
    calls = []
    def extract(stimuli, layers):
        calls.append(len(stimuli))
        return NeuroidAssembly(np.ones((len(stimuli), 2)), dims=['presentation', 'neuroid'],
            coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                    'layer': ('neuroid', ['layer', 'layer'])})
    raw = BrainScoreModel('fixture', extractor={'text': extract}, region_layer_map={'IT': 'layer'},
                         behavioral_readout_layer='layer')
    provenance = {'dtype': 'torch.bfloat16', 'weights_sha256': {'shard': 'fixture'}}
    subject = ProvenanceSubject(raw, provenance)
    provenance['weights_sha256']['shard'] = 'edited'
    for name in ('identifier', 'in_channels', 'out_channels', 'required_channels', 'traits'):
        assert getattr(subject, name) == getattr(raw, name)
    stimuli = StimulusSet([{'stimulus_id': 'a', 'sentence': 'test'}])
    task = TaskContext(task_type='passive')
    session = stimulus_session(stimuli, ['neural:IT'], task=task)
    subject.participate(session)
    result = session.collect('neural:IT')
    np.testing.assert_array_equal(result, np.ones((1, 2)))
    assert calls == [1]
    assert result.attrs['provenance']['weights_sha256']['shard'] == 'fixture'
    assert session.emitted[0].meta['provenance']['dtype'] == 'torch.bfloat16'
    result.attrs['provenance']['weights_sha256']['shard'] = 'changed response'
    assert session.emitted[0].meta['provenance']['weights_sha256']['shard'] == 'fixture'


def test_text_task_and_reset_delegate_without_discovering_model_abilities():
    prompts = []
    raw = BrainScoreModel('reply-fixture', supported_modalities={'text'},
                         reply=lambda prompt: prompts.append(prompt) or 'Yes')
    subject = ProvenanceSubject(raw, {'dtype': 'bfloat16'})
    stimuli = StimulusSet([{'stimulus_id': 'a', 'sentence': 'red?'}])
    def run():
        session = stimulus_session(stimuli, ['behavior:text'], task=TaskContext(response=Text()))
        subject.participate(session)
        return session.collect('behavior:text')
    first = run()
    first.attrs['provenance']['dtype'] = 'edited'
    subject.reset()
    second = run()
    assert first.values.tolist() == second.values.tolist() == ['Yes']
    assert second.attrs['provenance']['dtype'] == 'bfloat16'
    assert len(prompts) == 2


def test_event_metadata_shares_values_without_mutating_source():
    from brainscore.model_helpers.provenance import ProvenanceSubject
    assembly = NeuroidAssembly([[1., np.nan]], dims=['presentation', 'neuroid'], attrs={'original': True})
    event = StreamEvent('neural:IT', assembly, None, {'driver': 'fixture'})

    class Responder:
        def participate(self, session):
            session.respond(event)

    target = Session(())
    ProvenanceSubject(Responder(), {'dtype': 'bfloat16'}).participate(target)
    saved = target.emitted[0]
    assert target.complete
    assert np.shares_memory(saved.payload.values, assembly.values)
    assert saved.payload.attrs['original'] is True
    assert 'provenance' not in assembly.attrs and 'provenance' not in event.meta
    assert saved.meta['driver'] == 'fixture'


def test_failure_and_reset_are_not_swallowed():
    class FailingSubject:
        def participate(self, session):
            raise RuntimeError('original failure')
        def reset(self):
            self.reset_called = True
    raw = FailingSubject()
    subject = ProvenanceSubject(raw, {})
    with pytest.raises(RuntimeError, match='original failure'):
        subject.participate(Session(()))
    subject.reset()
    assert raw.reset_called
