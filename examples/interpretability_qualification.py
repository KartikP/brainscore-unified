"""Qualify tool mechanics on pinned pretrained encoders, offline on CPU float32.

Stage the files in interpretability_models.json first. No causal ground truth or
minimum probe accuracy is assumed. Reports distinguish checks from diagnostics.
"""
import argparse
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from brainscore.experiments import Experiment, SessionProtocol, TorchInstrumentation
from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore.run_record import RunRecord
from brainscore.tools import attach_tools, load_tool
from brainscore import BrainScoreModel
from brainscore_core.events import Selection
from brainscore_core.streaming import Session
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from example_qualification import environment, qualification_report, require, verify_model_identity


MODELS = json.loads(Path(__file__).with_name('interpretability_models.json').read_text())
CONDITIONS = ['baseline', 'ablated', 'scaled', 'positive', 'negative', 'zero', 'erased', 'composed', 'restored']


def sentences() -> tuple[list[str], list[str], list[str], list[str]]:
    """Discovery/evaluation differ in objects and sentence templates, not just IDs."""
    training = [f'The {color} {noun} is {place}.'
                for noun in ('car', 'boat', 'cup', 'chair', 'shirt', 'box', 'hat', 'door')
                for place in ('nearby', 'outside') for color in ('red', 'blue')]
    testing = [f'I noticed a {color} {noun} today.'
               for noun in ('bicycle', 'vase', 'jacket', 'umbrella') for color in ('red', 'blue')]
    return training, ['red', 'blue'] * 16, testing, ['red', 'blue'] * 4


def session(texts: list[str]) -> Session:
    stimuli = StimulusSet({'stimulus_id': [f'item-{i}' for i in range(len(texts))], 'sentence': texts})
    return stimulus_session(stimuli, ['neural:population'])


def weights_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode() + b'\0')
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def check_close(observed: Any, expected: Any, label: str, *, exact: bool = False) -> float:
    """Fail on NaNs/shapes as well as differences; checks survive python -O."""
    observed, expected = np.asarray(observed), np.asarray(expected)
    require(observed.shape == expected.shape, f'{label}: shape')
    require(np.isfinite(observed).all() and np.isfinite(expected).all(), f'{label}: finite')
    equal = np.array_equal(observed, expected) if exact else np.allclose(observed, expected, atol=2e-5, rtol=2e-5)
    require(equal, f'{label}: numerical equality')
    return float(np.max(np.abs(observed - expected)))


class Calls:
    def __init__(self) -> None:
        self.started, self.completed = 0, 0

    def on_start(self, call: Any) -> None:
        require(call.method == 'participate', 'explicitly observed participate')
        self.started += 1

    def on_result(self, call: Any, result: Any) -> None:
        self.completed += 1


def check_model(directory: Path, model: Any, tokenizer: Any, layer: str, provenance: dict) -> dict:
    """Run the same numerical gates on a trained model or a fast test fixture."""
    from concept_erasure import LeaceEraser
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    require(os.environ.get('RESULTCACHING_DISABLE') == '1', 'disable extraction caching for hook qualification')
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()),
            'model parameters on CPU float32')
    model.eval()
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'left'
    # Placement is a qualification control, not a change to the wrapper defaults.
    with patch('brainscore.model_helpers._device.select_device', return_value=torch.device('cpu')):
        extractor = TextWrapper(model, tokenizer, identifier='qualification',
                                layer_aggregation='last_token', batch_size=64, max_length=64)
    subject = BrainScoreModel(identifier='interpretability-qualification', extractor={'text': extractor},
                              region_layer_map={'population': layer})
    site = model.get_submodule(layer)
    provider = TorchInstrumentation({layer: site})
    weights, rng = weights_digest(model), torch.random.get_rng_state().clone()
    training, labels, testing, test_labels = sentences()
    require(not set(training) & set(testing), 'disjoint discovery and evaluation inputs')

    def evaluate(texts: list[str]) -> Any:
        current = session(texts)
        subject.participate(current)
        return current.collect()

    tokens = tokenizer(testing, padding=True, truncation=True, max_length=64, return_tensors='pt')
    with torch.no_grad():
        native = model(**tokens).last_hidden_state.detach().clone()
    lengths = tokens['attention_mask'].sum(1).numpy().astype(int)

    def pool(array: np.ndarray) -> np.ndarray:
        return np.stack([array[i, n - 1] for i, n in enumerate(lengths)])

    discovery, baseline = evaluate(training), evaluate(testing)
    errors = {'native_session': check_close(baseline.values, pool(native.numpy()), 'native/session', exact=True)}
    provenance = {**provenance, 'site': layer, 'dataset': 'disjoint-color-sentences-v1'}
    fitted = {
        'linear-probe': load_tool('linear-probe', target=layer).fit(discovery, labels, provenance=provenance),
        'leace': load_tool('leace', target=layer).fit(discovery, labels, provenance=provenance),
        'contrastive-steering': load_tool('contrastive-steering', target=layer, strength=0.5).fit(
            discovery, labels, positive='red', negative='blue', provenance=provenance),
    }
    for method, tool in fitted.items():
        tool.save(directory / f'{method}.json')
        loaded = load_tool(method, artifact=directory / f'{method}.json')
        apply = 'predict' if method == 'linear-probe' else 'transform'
        errors[f'{method}_reload'] = check_close(getattr(loaded, apply)(baseline).values,
            getattr(tool, apply)(baseline).values, f'{method} reload', exact=True)

    x, y = discovery.values.astype(np.float64), np.asarray(labels)
    scaler = StandardScaler().fit(x)
    reference_probe = LogisticRegression(C=1., max_iter=1000).fit(scaler.transform(x), y)
    errors['probe_reference'] = check_close(fitted['linear-probe'].predict(baseline).values,
        reference_probe.predict_proba(scaler.transform(baseline.values.astype(np.float64))), 'sklearn reference')
    classes, encoded = np.unique(y, return_inverse=True)
    reference = LeaceEraser.fit(torch.from_numpy(x), torch.from_numpy(np.eye(len(classes))[encoded]))
    expected_erased = reference(torch.from_numpy(baseline.values.astype(np.float64))).numpy()
    errors['leace_reference'] = check_close(fitted['leace'].transform(baseline).values, expected_erased,
                                          'LEACE reference')
    direction = x[y == 'red'].mean(0) - x[y == 'blue'].mean(0)
    errors['direction_reference'] = check_close(fitted['contrastive-steering'].transform(baseline).values,
        baseline.values + 0.5 * direction, 'independent direction')

    calls = Calls()

    @contextmanager
    def factory(trial):
        yield session(testing)

    result = Experiment(subject=subject,
        protocol=SessionProtocol('trained-tool-mechanics', factory, conditions=CONDITIONS,
                                 required_input_channels=['text'], requested_output_channels=['neural:population']),
        tools=[load_tool('record-io'), load_tool('observe-calls', observer=calls, methods=['participate']),
               load_tool('record-reasoning'),
               load_tool('record-activity', targets=[layer], name='before', when='before'),
               load_tool('linear-probe', artifact=directory / 'linear-probe.json', name='probe_before', when='before'),
               load_tool('ablate', targets=[Selection(layer, [0, 3])], conditions=['ablated']),
               load_tool('scale-activity', targets=[Selection(layer, [1, 4])], factor=0.5,
                         conditions=['scaled', 'composed']),
               load_tool('contrastive-steering', artifact=directory / 'contrastive-steering.json',
                         name='positive', conditions=['positive', 'composed']),
               load_tool('contrastive-steering', artifact=directory / 'contrastive-steering.json',
                         name='negative', strength=-0.5, conditions=['negative']),
               load_tool('contrastive-steering', artifact=directory / 'contrastive-steering.json',
                         name='zero', strength=0., conditions=['zero']),
               load_tool('leace', artifact=directory / 'leace.json', conditions=['erased']),
               load_tool('record-activity', targets=[layer], name='after'),
               load_tool('linear-probe', artifact=directory / 'linear-probe.json', name='probe_after')],
        instrumentation=provider, output_dir=directory / 'experiment', metadata={'provenance': provenance}).run()
    events = list(result.record.events())  # Validates persisted checksums while reading.
    for condition, output in zip(CONDITIONS, result.value):
        expected = native.clone()
        if condition == 'ablated':
            expected[..., [0, 3]] = 0
        if condition in ('scaled', 'composed'):
            expected[..., [1, 4]] *= 0.5
        if condition in ('positive', 'negative', 'composed'):
            expected += (0.5 if condition != 'negative' else -0.5) * torch.tensor(direction, dtype=expected.dtype)
        if condition == 'erased':
            live_reference = LeaceEraser(reference.proj_left.float(), reference.proj_right.float(), reference.bias.float())
            expected = live_reference(expected)
        for when, expected_tensor in [('before', native), ('after', expected)]:
            captures = [e['payload']['value']['array'] for e in events
                        if e['kind'] == 'activity' and e['condition'] == condition and e['source'] == when]
            require(len(captures) == 1, f'{condition}/{when}: one real forward')
            errors[f'{condition}_{when}'] = check_close(captures[0], expected_tensor.numpy(),
                f'{condition}/{when}', exact=condition != 'erased')
            predictions = [e['payload'].values for e in events if e['kind'] == 'probe'
                           and e['condition'] == condition and e['source'] == f'probe_{when}']
            require(len(predictions) == 1, f'{condition}/{when}: one probe measurement')
            expected_probabilities = reference_probe.predict_proba(scaler.transform(
                captures[0].astype(np.float64).reshape(-1, x.shape[-1])))
            errors[f'{condition}_{when}_probe'] = check_close(predictions[0], expected_probabilities.reshape(
                *captures[0].shape[:-1], len(classes)), f'{condition}/{when}/probe reference')
        errors[f'{condition}_response'] = check_close(output.values, pool(expected.numpy()),
            f'{condition}/response', exact=condition != 'erased')
    require(calls.started == calls.completed == len(CONDITIONS), 'observer call counts')
    require('participate' not in vars(subject), 'observer restores instance method state')
    probes = [e for e in events if e['kind'] == 'probe']
    require(len(probes) == 2 * len(CONDITIONS), 'both probes execute in every condition')
    for event in probes:
        probabilities = event['payload'].values
        require(np.isfinite(probabilities).all() and np.allclose(probabilities.sum(-1), 1.), 'probe probabilities')
    reasoning = list(RunRecord(directory / 'experiment' / 'reasoning').outputs())
    require(len(reasoning) == len(CONDITIONS) and all(
        not r['reasoning_available'] and r['reasoning'] == [] for r in reasoning), 'encoder reasoning unavailable')

    # Exact donor patch: same shapes/masks, opposite color tokens, whole final output.
    donor_texts = [t.replace('red', 'TEMP').replace('blue', 'red').replace('TEMP', 'blue') for t in testing]
    donor_tokens = tokenizer(donor_texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
    require(torch.equal(tokens['attention_mask'], donor_tokens['attention_mask']), 'explicit donor token alignment')
    analysis = provider.attribute([layer], reference_call=lambda: model(**donor_tokens),
        comparison_call=lambda: model(**tokens), objective=lambda out: out.last_hidden_state[..., 0].sum(),
        normalize=False)
    with attach_tools(subject, tools=[load_tool('record-io'),
                      load_tool('patch-activity', targets=[layer], reference=analysis['reference'])],
                      instrumentation=provider, output_dir=directory / 'patch'):
        patched = evaluate(testing)
    errors['exact_patch'] = check_close(patched.values, pool(analysis['reference']['values'][layer]),
                                       'donor patch', exact=True)

    # Keep an independent hook installed while the tool scope fails.
    witness = []
    handle = site.register_forward_hook(lambda *args: witness.append(1))
    prior_hooks = set(site._forward_hooks)
    try:
        try:
            with attach_tools(subject, tools=[load_tool('record-io'),
                              load_tool('contrastive-steering', artifact=directory / 'contrastive-steering.json')],
                              instrumentation=provider, output_dir=directory / 'interrupted'):
                evaluate(testing)
                raise RuntimeError('intentional qualification interruption')
        except RuntimeError as error:
            require(str(error) == 'intentional qualification interruption', 'expected qualification interruption')
        require(set(site._forward_hooks) == prior_hooks, 'only owned hooks removed after failure')
        errors['failure_restoration'] = check_close(evaluate(testing).values, baseline.values,
                                                   'failure restoration', exact=True)
        require(len(witness) == 2, 'pre-existing hook remains active')
    finally:
        handle.remove()
    require(not site._forward_hooks, 'no leaked hooks')
    require(RunRecord(directory / 'interrupted').manifest['status'] == 'failed', 'failed record retained')
    require(weights_digest(model) == weights, 'all parameter and buffer bytes unchanged')
    require(all(p.grad is None for p in model.parameters()), 'no parameter gradients accumulated')
    require(torch.equal(torch.random.get_rng_state(), rng), 'torch RNG unchanged')

    probe = fitted['linear-probe']
    shuffled = load_tool('linear-probe', target=layer).fit(discovery, np.random.RandomState(0).permutation(labels))
    def accuracy(tool, values):
        predictions = tool.predict(values)
        return float(np.mean(predictions['choice'].values[predictions.values.argmax(-1)] == test_labels))
    return {'checks': 'passed',
            'lifecycle_checks': ['observer restored', 'reasoning unavailable', 'failed record retained',
                                 'pre-existing hook retained', 'no leaked hooks', 'parameter bytes unchanged',
                                 'no parameter gradients', 'torch RNG unchanged'],
            'n_conditions': len(CONDITIONS), 'n_discovery': len(training),
            'n_evaluation': len(testing), 'width': baseline.shape[-1], 'site': layer,
            'max_absolute_errors': errors, 'parameter_sha256': weights,
            'diagnostics': {'held_out_probe_accuracy': accuracy(probe, baseline),
                            'shuffled_label_probe_accuracy': accuracy(shuffled, baseline),
                            'erased_probe_accuracy': accuracy(probe, fitted['leace'].transform(baseline))},
            'evidence': {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(directory.rglob('*')) if p.is_file()}}


def qualify(directory: Path, key: str, snapshot: Path) -> dict:
    """Verify assets before loading and preserve a failed report on any exception."""
    directory.mkdir(parents=True, exist_ok=False)
    report = {'model': key, 'environment': environment(), 'device': 'cpu', 'dtype': 'float32',
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'numerical mechanics and integration; no causal or human validity claim'}
    with qualification_report(directory / 'qualification.json', report):
        manifest = MODELS[key]
        report['identity'] = verify_model_identity(snapshot, manifest)
        from transformers import AutoTokenizer, CLIPTextModel, GPT2Model
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        if key == 'gpt2':
            tokenizer.pad_token = tokenizer.eos_token
            model = GPT2Model.from_pretrained(snapshot, local_files_only=True, use_safetensors=True)
        else:
            model = CLIPTextModel.from_pretrained(snapshot, local_files_only=True, use_safetensors=False)
        report['packages'] = {name: metadata.version(name) for name in ('scikit-learn', 'concept-erasure')}
        report['results'] = check_model(directory, model, tokenizer, manifest['layer'], report['identity'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=sorted(MODELS), required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = qualify(args.out, args.model, args.snapshot)
    print(json.dumps({'status': result['status'], 'results': result['results']}, indent=2))
