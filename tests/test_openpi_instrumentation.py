"""Real pinned OpenPI sampler/Policy with a tiny untrained NNX network on CPU.

These tests exercise JIT and the upstream denoising loop. They are not trained
policy or simulator qualification. Set OPENPI_SOURCE to the pinned checkout.
"""
import os
from pathlib import Path
import sys

import numpy as np
import pytest

source = os.environ.get('OPENPI_SOURCE')
if not source:
    pytest.skip('Set OPENPI_SOURCE for optional OpenPI/JAX qualification', allow_module_level=True)
sys.path[:0] = [str(Path(source) / 'src'), str(Path(source) / 'packages/openpi-client/src')]

import xarray  # Load before OpenPI/Orbax to avoid the legacy dask import cycle.
import jax
import jax.numpy as jnp
from flax import nnx
from openpi.models.pi0 import Pi0
from openpi.models.model import IMAGE_KEYS
from openpi.policies.policy import Policy
from brainscore_core.events import Selection
from brainscore.experiments.backends.openpi import OpenPIInstrumentation

pytestmark = pytest.mark.integration


class TinyTransformer(nnx.Module):
    def __call__(self, tokens, **kwargs):
        prefix, suffix = tokens
        if suffix is None:
            return [prefix, None], jnp.mean(prefix)
        return [None, jnp.tanh(suffix + kwargs['kv_cache'])], None


class TinyPi0(Pi0):
    """Use the real sampling function with small deterministic stand-in layers."""
    def __init__(self):
        self.action_dim, self.action_horizon = 2, 3
        self.action_in_proj = nnx.Linear(2, 4, rngs=nnx.Rngs(0))
        self.action_out_proj = nnx.Linear(4, 2, rngs=nnx.Rngs(1))
        self.PaliGemma = nnx.Dict(llm=TinyTransformer())

    def embed_prefix(self, obs):
        tokens = jnp.ones((obs.state.shape[0], 2, 4)) * 0.1
        return tokens, jnp.ones(tokens.shape[:2], dtype=bool), jnp.zeros(2, dtype=bool)

    def embed_suffix(self, obs, actions, time):
        tokens = self.action_in_proj(actions) + time[:, None, None] + obs.state[:, :1, None]
        return (tokens, jnp.ones(tokens.shape[:2], dtype=bool),
                jnp.ones(tokens.shape[1], dtype=bool), None)


@pytest.fixture
def policy():
    return Policy(TinyPi0(), sample_kwargs={'num_steps': 4})


@pytest.fixture
def observation():
    return {'state': np.array([0.2, 0.3], dtype=np.float32),
            'image': {key: np.zeros((224, 224, 3), dtype=np.float32) for key in IMAGE_KEYS},
            'image_mask': {key: np.array(True) for key in IMAGE_KEYS}}


def provider(policy, **kwargs):
    return OpenPIInstrumentation(policy, checkpoint='untrained-tiny-test-fixture', **kwargs)


def test_recording_preserves_actions_rng_and_sampler(policy, observation):
    reference = Policy(TinyPi0(), sample_kwargs={'num_steps': 4})
    original = policy._sample_actions
    instrumentation = provider(policy)
    captured = []
    with instrumentation.record(list(instrumentation.widths), lambda *args: captured.append(args)):
        for _ in range(2):
            expected = reference.infer(observation)
            actual = policy.infer(observation)
            np.testing.assert_array_equal(actual['actions'], expected['actions'])
            np.testing.assert_array_equal(actual['state'], expected['state'])
    assert policy._sample_actions is original
    np.testing.assert_array_equal(jax.random.key_data(policy._rng), jax.random.key_data(reference._rng))
    assert len(captured) == 24
    assert [row[1]['denoising_step'] for row in captured[:12]] == [0]*3 + [1]*3 + [2]*3 + [3]*3
    assert all(row[1]['dtype'] == 'float32' for row in captured)


@pytest.mark.parametrize('site', ['embed_suffix', 'PaliGemma.llm.suffix', 'action_out_proj'])
def test_indexed_ablation_before_after_cleanup(policy, observation, site):
    instrumentation = provider(policy, steps=[1])
    noise = np.ones((3, 2), dtype=np.float32)
    baseline = policy.infer(observation, noise=noise)['actions']
    before, after = [], []
    with instrumentation.record([site], lambda _, value: before.append(value)):
        with instrumentation.ablate([Selection(layer=site, indices=[0])]):
            with instrumentation.record([site], lambda _, value: after.append(value)):
                changed = policy.infer(observation, noise=noise)['actions']
    assert len(before) == len(after) == 1
    assert before[0]['denoising_step'] == 1
    assert np.any(before[0]['array'][..., 0] != 0)
    np.testing.assert_array_equal(after[0]['array'][..., 0], 0)
    np.testing.assert_array_equal(before[0]['array'][..., 1:], after[0]['array'][..., 1:])
    assert not np.array_equal(changed, baseline)
    np.testing.assert_array_equal(policy.infer(observation, noise=noise)['actions'], baseline)


def test_full_velocity_ablation_and_exception_cleanup(policy, observation):
    instrumentation = provider(policy)
    original = policy._sample_actions
    noise = np.ones((3, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match='evaluator failed'):
        with instrumentation.ablate(['action_out_proj']):
            np.testing.assert_array_equal(policy.infer(observation, noise=noise)['actions'], noise)
            raise RuntimeError('evaluator failed')
    assert policy._sample_actions is original
    with provider(policy).record(['action_out_proj'], lambda *args: None):
        policy.infer(observation, noise=noise)


@pytest.mark.parametrize('target', ['unknown', Selection(layer='action_out_proj', indices=[2]),
                                    Selection(layer='action_out_proj', indices=[-1]),
                                    Selection(layer='action_out_proj', indices=[])])
def test_invalid_selection(policy, target):
    with pytest.raises(ValueError):
        provider(policy).validate('record', [target])


def test_unsupported_source(policy, monkeypatch):
    import brainscore.experiments.backends.openpi as backend
    monkeypatch.setattr(backend, 'SAMPLER_SHA256', 'changed')
    with pytest.raises(ValueError, match='Unsupported OpenPI'):
        provider(policy)


def test_step_outside_loop_and_duplicate_provider(policy, observation):
    original = policy._sample_actions
    with provider(policy, steps=[4]).ablate(['action_out_proj']):
        with pytest.raises(ValueError, match='outside'):
            policy.infer(observation)
        with pytest.raises(RuntimeError, match='already'):
            with provider(policy).ablate(['action_out_proj']):
                pass
    assert policy._sample_actions is original


def test_real_jax_remote_experiment(policy, observation, tmp_path):
    from tests.test_openpi_remote import listening
    from brainscore.experiments import Experiment, CallableProtocol, RecordInputsOutputs, RecordActivity, Ablate
    from brainscore.experiments.backends.openpi_remote import (
        OpenPIToolServer, OpenPIPolicyClient, RemoteOpenPIInstrumentation,
    )
    from brainscore.run_record import RunRecord

    server = OpenPIToolServer(policy, provider(policy, steps=[0, 2]))
    noise = np.ones((3, 2), dtype=np.float32)
    baseline = policy.infer(observation, noise=noise)['actions']
    with listening(server) as uri, OpenPIPolicyClient(uri) as client:
        result = Experiment(
            subject=client,
            protocol=CallableProtocol(
                'openpi-cpu-fixture',
                lambda subject, context: subject.infer(observation, noise=noise),
                methods=['infer'],
            ),
            instrumentation=RemoteOpenPIInstrumentation(client),
            tools=[
                RecordInputsOutputs(),
                RecordActivity(['action_out_proj'], when='before', name='before'),
                Ablate([Selection(layer='action_out_proj', indices=[0])]),
                RecordActivity(['action_out_proj'], name='after'),
            ],
            output_dir=tmp_path / 'experiment',
        ).run()
        assert result.manifest['status'] == 'complete'
        assert not np.array_equal(result.value['actions'], baseline)
        np.testing.assert_array_equal(client.infer(observation, noise=noise)['actions'], baseline)
    records = list(RunRecord(result.directory / 'inputs_outputs').events())
    activity = [record['payload'] for record in records if record['payload']['kind'] == 'activity']
    assert len(activity) == 4
    for event in activity:
        assert event['event_id']
        assert event['trial_id'] == 'external'
        assert event['payload']['value']['request_id']
    after = [event['payload']['value']['array'] for event in activity if event['source'] == 'after']
    assert len(after) == 2
    for array in after:
        np.testing.assert_array_equal(array[..., 0], 0)


def test_record_callback_failure_restores_sampler(policy, observation):
    original = policy._sample_actions
    def fail(*args):
        raise RuntimeError('recorder failed')
    with pytest.raises(Exception, match='recorder failed'):
        with provider(policy).record(['action_out_proj'], fail):
            policy.infer(observation)
    assert policy._sample_actions is original
    assert np.isfinite(policy.infer(observation)['actions']).all()


def test_record_selected_indices_and_scope_reuse(policy, observation):
    instrumentation = provider(policy, steps=[0])
    first, second = [], []
    target = Selection(layer='embed_suffix', indices=[3, 1])
    for rows in [first, second]:
        with instrumentation.record([target], lambda _, v: rows.append(v)):
            policy.infer(observation)
    assert len(first) == len(second) == 1
    assert first[0]['array'].shape == (1, 3, 2)
    assert first[0]['indices'] == (3, 1)


def test_input_output_transforms_are_preserved(observation):
    def input_transform(obs):
        return dict(obs, state=obs['state'] + 1)
    def output_transform(output):
        return dict(output, actions=output['actions'] * 2, transformed=True)
    options = dict(sample_kwargs={'num_steps': 4}, transforms=[input_transform],
                   output_transforms=[output_transform])
    plain, instrumented = Policy(TinyPi0(), **options), Policy(TinyPi0(), **options)
    with provider(instrumented).record(['action_out_proj'], lambda *args: None):
        actual = instrumented.infer(observation)
    expected = plain.infer(observation)
    np.testing.assert_array_equal(actual['actions'], expected['actions'])
    np.testing.assert_array_equal(actual['state'], expected['state'])
    assert actual['transformed'] is True


def test_qualification_bundle(policy, observation, tmp_path):
    import importlib.util
    path = Path(__file__).parents[1] / 'examples/libero/qualify_tools.py'
    spec = importlib.util.spec_from_file_location('qualify_tools', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = jax.random.key_data(policy._rng).copy()
    report = module.qualify(policy, [observation, observation], tmp_path / 'qualification',
                            checkpoint='untrained-tiny-test-fixture', steps=[0, 1])
    assert report['status'] == 'passed'
    assert report['calls_per_route'] == 2
    np.testing.assert_array_equal(jax.random.key_data(policy._rng), rng)


def test_repeated_plan_reuses_compilation_but_not_callbacks(policy, observation, monkeypatch):
    import brainscore.experiments.backends.openpi as backend
    instrumentation = provider(policy, steps=[0])
    original = backend._instrument_sampler
    compilations = []
    def count(*args):
        compilations.append(True)
        return original(*args)
    monkeypatch.setattr(backend, '_instrument_sampler', count)
    results = [[], []]
    for result in results:
        with instrumentation.record(['action_out_proj'], lambda _, v: result.append(v)):
            policy.infer(observation)
    assert len(compilations) == 1
    assert [len(result) for result in results] == [1, 1]


@pytest.mark.parametrize('steps', [[], [-1], [True], [0, 0], [1.5]])
def test_invalid_steps(policy, steps):
    with pytest.raises(ValueError, match='steps'):
        provider(policy, steps=steps)


def test_wrong_thread_cannot_attach_or_infer(policy, observation):
    from concurrent.futures import ThreadPoolExecutor
    instrumentation = provider(policy)
    with instrumentation.ablate(['action_out_proj']), ThreadPoolExecutor(1) as executor:
        with pytest.raises(RuntimeError, match='scope thread'):
            executor.submit(policy.infer, observation).result()
    assert not hasattr(policy, '_umi_instrumentation_owner')


def test_bfloat16_measurements_keep_original_dtype(observation):
    class BFloatTransformer(TinyTransformer):
        def __call__(self, tokens, **kwargs):
            outputs, cache = super().__call__(tokens, **kwargs)
            return [None if value is None else value.astype(jnp.bfloat16) for value in outputs], cache
    model = TinyPi0()
    model.PaliGemma.llm = BFloatTransformer()
    policy = Policy(model, sample_kwargs={'num_steps': 4})
    noise = np.ones((3, 2), dtype=np.float32)
    baseline = policy.infer(observation, noise=noise)['actions']
    captured = []
    with provider(policy, steps=[1]).record(['PaliGemma.llm.suffix'], lambda _, value: captured.append(value)):
        actual = policy.infer(observation, noise=noise)['actions']
    np.testing.assert_array_equal(actual, baseline)
    assert len(captured) == 1
    assert captured[0]['dtype'] == 'bfloat16'
    assert captured[0]['array'].dtype == np.float32
