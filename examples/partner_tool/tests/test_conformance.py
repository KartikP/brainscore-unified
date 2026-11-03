"""Adapter conformance checks an external package can run on its own integration.

Copy this file and replace PartnerEnvironment, partner_policy and
partner_evaluator with your adapter. Only public entry points are used. Every
check runs before the first action, in two explicit stages:

1. Cheap: Experiment(...).validate(), also called by run(). A bad declaration
   raises with no output directory, no reset(), no step() and no model call.
2. Initialization: EnvironmentSession(environment=env, require_specs=True, seed=...) and one
   next_input(). The first observation is checked against observation_spec()
   before any step().

What a declaration does not say is unknown, not a default:
- describe() reports an undeclared observation or action spec as None.
- ArraySpec and DiscreteSpec carry no units, coordinate frame or control
  period. Only ActionSpec declares those, and it requires all three.
- An observation without t_ms has no physical clock: its context has
  t_ms=None and time_source='step_index' (step counts, not milliseconds).

External evaluators have further declarations that are tested in the unified
repository, not here:
- CogGym: the pinned revision and check_channel_compatibility
  (brainscore/harnesses/coggym.py).
- LIBERO bridge: Bridge.__init__ / check_start and the LiberoChunkPolicy shape
  checks (examples/libero).
"""
from contextlib import contextmanager
import os

import numpy as np
import pytest
import torch

from brainscore.experiments import (
    Ablate, CallableProtocol, Experiment, RecordActivity, RecordInputsOutputs,
    SessionProtocol, TorchInstrumentation,
)
from brainscore_core.environment import ArraySpec, DiscreteSpec, MappingSpec
from brainscore_core.events import EnvironmentResponse, EnvironmentStep, Selection
from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.streaming_helpers import EnvironmentSession

STEPS = 3


class PartnerEnvironment:
    """A minimal declared environment: two values in [0, 1], two actions."""

    def __init__(self, observation=None):
        self.bad_observation = observation
        self.resets = self.steps = 0
        self.closed = False

    def observation_spec(self):
        return MappingSpec({'state': ArraySpec((2,), 'float32', lower=0, upper=1)})

    def action_spec(self):
        return DiscreteSpec(2)

    def _step(self):
        state = np.array([self.steps / STEPS, 0.5], dtype=np.float32)
        if self.bad_observation is not None:
            state = self.bad_observation
        return EnvironmentStep(observation={'state': state}, step_num=self.steps,
                               is_first=self.steps == 0, is_last=self.steps == STEPS)

    def reset(self, *, seed=None):
        self.resets += 1
        self.steps = 0
        return self._step()

    def step(self, action):
        self.steps += 1
        return self._step()

    def close(self):
        self.closed = True


class Partner:
    """One environment, one torch layer and one policy, with call counts."""

    def __init__(self, observation=None):
        torch.manual_seed(0)
        self.layer = torch.nn.Linear(2, 2)
        self.environment = PartnerEnvironment(observation)
        self.calls = 0

    def policy(self, step):
        self.calls += 1
        with torch.no_grad():
            scores = self.layer(torch.from_numpy(step.observation['state']))
        return EnvironmentResponse(action=int(scores.argmax()))

    def subject(self, *, with_policy=True):
        return BrainScoreModel(identifier='partner-policy',
                               policy=self.policy if with_policy else None)

    @contextmanager
    def session(self, trial):
        with EnvironmentSession(environment=self.environment, seed=trial.identifier,
                                require_specs=True) as session:
            yield session

    def native(self, **kwargs):
        return SessionProtocol(identifier='partner-episode', session_factory=self.session, trials=(7,),
                               input_channels={'observation'}, output_channels={'motor'},
                               **kwargs)

    def external(self, methods=('process',)):
        def partner_evaluator(subject, context):
            # The external evaluator owns its loop, resets and scoring.
            env = self.environment
            step, total = env.reset(seed=7), 0
            while not step.is_last:
                total += int(subject.process(step).action)
                step = env.step(0)
            return {'score': total}
        return CallableProtocol(identifier='partner-evaluator', evaluate=partner_evaluator,
                                methods=methods)

    def untouched(self):
        return self.environment.resets == self.environment.steps == self.calls == 0


def experiment(partner, output, *, protocol=None, tools=(), instrument=True, **subject):
    return Experiment(subject=partner.subject(**subject), protocol=protocol or partner.native(),
                      tools=tools, output_dir=output,
                      instrumentation=TorchInstrumentation({'proj': partner.layer}) if instrument else None)


@pytest.mark.parametrize('protocol', ['native', 'external'])
def test_valid_declarations_run(tmp_path, protocol):
    partner = Partner()
    tools = [RecordInputsOutputs(max_bytes=8 * 1024**2), RecordActivity(['proj'])]
    run = experiment(partner, tmp_path / 'run', tools=tools,
                     protocol=getattr(partner, protocol)())
    run.validate()
    assert not (tmp_path / 'run').exists() and partner.untouched()
    result = run.run()
    assert result.manifest['status'] == 'complete'
    assert partner.calls == STEPS and not partner.layer._forward_hooks


@pytest.mark.parametrize('case, error', [
    ('missing channel', ValueError),
    ('missing tool provider', ValueError),
    ('unknown target', ValueError),
    ('wrong tool order', ValueError),
    ('missing method', ValueError),
    ('existing output', FileExistsError),
])
def test_bad_declarations_fail_before_any_work(tmp_path, case, error):
    partner = Partner()
    output = tmp_path / 'run'
    record = RecordInputsOutputs()
    run = {
        'missing channel': lambda: experiment(partner, output, with_policy=False),
        'missing tool provider': lambda: experiment(
            partner, output, tools=[record, RecordActivity(['proj'])], instrument=False),
        'unknown target': lambda: experiment(partner, output, tools=[record, RecordActivity(['nope'])]),
        'wrong tool order': lambda: experiment(partner, output, tools=[
            record, Ablate([Selection(layer='proj')]), RecordActivity(['proj'], when='before', name='before')]),
        'missing method': lambda: experiment(partner, output, protocol=partner.external(methods=('infer',))),
        'existing output': lambda: experiment(partner, output),
    }[case]()
    if case == 'existing output':
        output.mkdir()
    with pytest.raises(error):
        run.validate()
    with pytest.raises(error):
        run.run()
    assert partner.untouched()
    assert list(output.iterdir()) == [] if case == 'existing output' else not output.exists()


@pytest.mark.skipif(os.name != 'posix' or os.geteuid() == 0, reason='needs POSIX permissions as non-root')
def test_unwritable_output_parent_fails_before_any_work(tmp_path):
    partner = Partner()
    parent = tmp_path / 'locked'
    parent.mkdir()
    parent.chmod(0o500)
    try:
        run = experiment(partner, parent / 'run', tools=[RecordInputsOutputs()])
        # Required: run() stops before any reset or model call. Whether validate()
        # already catches this depends on the runner version, so it is not asserted.
        with pytest.raises(PermissionError):
            run.run()
        assert partner.untouched() and not (parent / 'run').exists()
    finally:
        parent.chmod(0o700)


@pytest.mark.parametrize('observation', [
    np.zeros(3, dtype=np.float32),         # wrong shape
    np.zeros(2, dtype=np.float64),         # wrong dtype
    np.array([0, 2], dtype=np.float32),    # outside the declared range
])
def test_first_observation_is_checked_before_any_step(observation):
    partner = Partner(observation)
    with EnvironmentSession(environment=partner.environment, require_specs=True, seed=3) as session:
        with pytest.raises(ValueError):
            session.next_input()
    assert partner.environment.resets == 1 and partner.environment.steps == 0
    assert partner.calls == 0 and partner.environment.closed


def test_undeclared_specs_are_unknown_not_defaults():
    class Untyped:
        def reset(self, *, seed=None):
            return EnvironmentStep(observation={'state': np.zeros(2)}, step_num=0, is_first=True)

        def step(self, action):
            raise AssertionError('not reached')

    with pytest.raises(TypeError):
        EnvironmentSession(environment=Untyped(), require_specs=True)
    with EnvironmentSession(environment=Untyped(), seed=1) as session:
        description = session.describe()
        assert description['observation_spec'] is None and description['action_spec'] is None
        first = session.next_input()
    assert first.context['t_ms'] is None and first.context['time_source'] == 'step_index'


def test_declared_action_spec_has_no_units_frame_or_period():
    description = PartnerEnvironment().action_spec().describe()
    assert not {'units', 'frame', 'period_ms'} & set(description)
