"""Explicit probes inside the pinned OpenPI Pi0/Pi0.5 JAX sampler.

OpenPI has no intermediate-value hook API. We insert three probes into its
verified sampling function, retaining its loop, RNG and numerical operations.
This module imports JAX/OpenPI only when a provider is constructed.
"""
import ast
from contextlib import contextmanager
import hashlib
import inspect
import numbers
import textwrap
import threading
from types import MethodType

from brainscore_core.events import Selection

OPENPI_REVISION = '215abfb217dbac7d5f1273282331b9b1866c0479'
# SHA256 of dedented inspect.getsource(Pi0.sample_actions), including decorators.
SAMPLER_SHA256 = 'da5a021ff1e58d893428c4c4915348a112668e65403ac3f9bf73c8b3265574fe'
SITES = {
    'embed_suffix': 'Action/state/time embeddings before the transformer',
    'PaliGemma.llm.suffix': 'Action transformer output before the projection',
    'action_out_proj': 'Denoising velocity before integration, not returned actions',
}


def _instrument_sampler(method, probe):
    source = textwrap.dedent(inspect.getsource(method))
    if hashlib.sha256(source.encode()).hexdigest() != SAMPLER_SHA256:
        raise ValueError('Unsupported OpenPI sampler; use the documented pinned revision')
    tree = ast.parse(source)
    function = tree.body[0]
    function.decorator_list = []
    step = next(node for node in function.body
                if isinstance(node, ast.FunctionDef) and node.name == 'step')
    sites = {'suffix_tokens': 'embed_suffix', 'suffix_out': 'PaliGemma.llm.suffix',
             'v_t': 'action_out_proj'}
    body, inserted = [], set()
    for statement in step.body:
        body.append(statement)
        if not isinstance(statement, ast.Assign):
            continue
        names = {node.id for target in statement.targets for node in ast.walk(target)
                 if isinstance(node, ast.Name)}
        for variable, site in sites.items():
            if variable in names:
                body.extend(ast.parse(
                    f'{variable} = _umi_probe({site!r}, {variable}, time, num_steps)'
                ).body)
                inserted.add(site)
    if inserted != set(SITES):
        raise ValueError('OpenPI probe sites changed')
    step.body = body
    namespace = dict(method.__globals__, _umi_probe=probe)
    exec(compile(ast.fix_missing_locations(tree), '<UMI OpenPI probes>', 'exec'), namespace)
    return namespace[function.name]


class OpenPIInstrumentation:
    """Record or zero selected last-axis units during JAX policy inference.

    ``steps`` selects zero-based denoising iterations (all by default). Scopes
    and inference must run serially on one thread. Keep model weights fixed for
    the lifetime of the policy, as required by OpenPI's frozen module_jit state.
    """
    def __init__(self, policy, *, steps=None, checkpoint):
        import jax
        import flax
        from openpi.models.pi0 import Pi0
        from openpi.policies.policy import Policy

        if not isinstance(policy, Policy) or policy._is_pytorch_model:
            raise TypeError('Expected a local OpenPI JAX Policy')
        if not isinstance(policy._model, Pi0) or policy._model.sample_actions.__func__ is not Pi0.sample_actions:
            raise TypeError('Only the pinned Pi0/Pi0.5 sampling method is supported')
        # Validate the exact function before ever attaching to the policy.
        _instrument_sampler(Pi0.sample_actions, lambda *args: None)
        if jax.__version__ != '0.5.3' or flax.__version__ != '0.10.2':
            raise ValueError('OpenPI instrumentation requires JAX 0.5.3 and Flax 0.10.2')
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError('Provide a checkpoint identifier (or explicitly label an untrained fixture)')
        self.policy, self.checkpoint = policy, checkpoint
        self.steps = None if steps is None else tuple(steps)
        if self.steps is not None and (not self.steps or any(
                isinstance(s, bool) or not isinstance(s, numbers.Integral) or s < 0
                for s in self.steps) or len(set(self.steps)) != len(self.steps)):
            raise ValueError('steps must contain distinct nonnegative integer iterations')
        model = policy._model
        self.widths = {
            'embed_suffix': model.action_in_proj.kernel.value.shape[-1],
            'PaliGemma.llm.suffix': model.action_out_proj.kernel.value.shape[0],
            'action_out_proj': model.action_out_proj.kernel.value.shape[-1],
        }
        self._scopes = []
        self._owner = None
        self._original = None
        self._compiled = {}
        self._in_call = False

    def describe(self):
        import jax
        return {
            'provider': type(self).__name__, 'backend': 'jax',
            'jax_version': jax.__version__, 'checkpoint': self.checkpoint,
            'sampler_sha256': SAMPLER_SHA256, 'openpi_revision': OPENPI_REVISION,
            'devices': [str(device) for device in jax.devices()],
            'steps': self.steps,
            'targets': {name: {'description': SITES[name], 'width': width}
                        for name, width in self.widths.items()},
            'operations': ['record', 'ablate'],
        }

    def validate(self, operation, targets):
        if operation not in ('record', 'ablate'):
            raise ValueError(f'Unsupported operation: {operation}')
        if not targets:
            raise ValueError('Select at least one target')
        for target in targets:
            selection = Selection(layer=target) if isinstance(target, str) else target
            if not isinstance(selection, Selection) or selection.layer not in self.widths:
                raise ValueError(f'Unknown OpenPI target: {target}')
            indices = selection.indices
            if indices is not None and (not len(indices) or any(
                    isinstance(i, bool) or not isinstance(i, numbers.Integral)
                    or i < 0 or i >= self.widths[selection.layer] for i in indices)):
                raise ValueError(f'Invalid last-axis indices for {selection.layer}')

    def record(self, targets, receive):
        if not callable(receive):
            raise TypeError('receive must be callable')
        return self._attach('record', targets, receive)

    def ablate(self, targets):
        return self._attach('ablate', targets, None)

    @contextmanager
    def _attach(self, operation, targets, receive):
        targets = tuple(targets)
        self.validate(operation, targets)
        owner = threading.get_ident()
        if self._in_call or (self._owner is not None and self._owner != owner):
            raise RuntimeError('OpenPI instrumentation requires serial scopes and inference')
        if not self._scopes:
            if getattr(self.policy, '_umi_instrumentation_owner', None) is not None:
                raise RuntimeError('This policy already has an instrumentation provider attached')
            self.policy._umi_instrumentation_owner = self
            self._owner = owner
            self._original = self.policy._sample_actions
            self.policy._sample_actions = self._sample_actions
        scope = (operation, tuple(
            Selection(layer=t) if isinstance(t, str) else Selection(
                layer=t.layer, indices=None if t.indices is None else tuple(t.indices))
            for t in targets), receive)
        self._scopes.append(scope)
        try:
            yield
        finally:
            self._scopes.pop()
            if not self._scopes:
                self.policy._sample_actions = self._original
                del self.policy._umi_instrumentation_owner
                self._original = self._owner = None

    def _sample_actions(self, *args, **kwargs):
        import jax
        from openpi.shared.nnx_utils import module_jit
        if self._in_call or threading.get_ident() != self._owner:
            raise RuntimeError('OpenPI instrumentation requires serial inference on the scope thread')
        num_steps = kwargs.get('num_steps', 10)
        if isinstance(num_steps, bool) or not isinstance(num_steps, numbers.Integral) or num_steps <= 0:
            raise ValueError('num_steps must be a positive Python integer')
        if self.steps is not None and max(self.steps) >= num_steps:
            raise ValueError('Selected denoising step is outside num_steps')
        self._in_call = True
        try:
            key = tuple((operation, tuple((target.layer, None if target.indices is None
                                           else tuple(target.indices)) for target in targets))
                        for operation, targets, _ in self._scopes)
            if key not in self._compiled:
                model = self.policy._model
                method = _instrument_sampler(model.sample_actions.__func__, self._make_probe())
                # Keep a small cache: remote calls repeat plans, while compiled models are large.
                if len(self._compiled) >= 2:
                    self._compiled.pop(next(iter(self._compiled)))
                self._compiled[key] = module_jit(MethodType(method, model))
            result = self._compiled[key](*args, **kwargs)
            result.block_until_ready()
            return result
        finally:
            try:
                jax.effects_barrier()
            finally:
                self._in_call = False

    def _make_probe(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        scopes = tuple((operation, targets) for operation, targets, _ in self._scopes)
        steps = self.steps

        def probe(site, value, time, num_steps):
            step = jnp.rint((1 - time) * num_steps).astype(jnp.int32)
            active = jnp.asarray(True) if steps is None else jnp.any(step == jnp.asarray(steps))
            for scope_index, (operation, targets) in enumerate(scopes):
                for selection in targets:
                    if selection.layer != site:
                        continue
                    indices = None if selection.indices is None else tuple(selection.indices)
                    if operation == 'ablate':
                        zeroed = (jnp.zeros_like(value) if indices is None
                                  else value.at[..., jnp.asarray(indices)].set(0))
                        value = jnp.where(active, zeroed, value)
                    else:
                        # Select recorded units on the host. A gather inside the JAX
                        # graph can change GPU fusion/rounding even without ablation.
                        captured = value
                        dtype = str(value.dtype)

                        def deliver(array, iteration, clock,
                                    scope_index=scope_index, indices=indices, dtype=dtype):
                            array = np.asarray(array)
                            if indices is not None:
                                array = np.take(array, indices, axis=-1)
                            self._scopes[scope_index][2](site, {
                                'array': np.array(array, copy=True), 'indices': indices,
                                'denoising_step': int(iteration), 'time': float(clock),
                                'dtype': dtype,
                            })
                        # Record bfloat16 through the existing portable NumPy record codec.
                        if str(captured.dtype) == 'bfloat16':
                            captured = captured.astype(jnp.float32)
                        def capture(_):
                            jax.debug.callback(deliver, captured, step, time, ordered=True)
                        # Unselected iterations should not transfer model activity to the host.
                        jax.lax.cond(active, capture, lambda _: None, operand=None)
            return value
        return probe
