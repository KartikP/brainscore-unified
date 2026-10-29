"""Qualification scripts fail loudly, also under python -O, and always leave a report.

The fast tests use a tiny fixed-weight torch module, not a trained model.
test_native_trained_smolvlm_qualification is the trained-model check; it is
marked slow and runs only in the native-environment-smolvlm integration
profile, with the snapshot pre-staged in the Hugging Face cache.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


@pytest.fixture(autouse=True)
def examples_on_path(monkeypatch):
    """Import the example scripts for this test only; sys.path is restored after."""
    monkeypatch.syspath_prepend(str(EXAMPLES))


def _optimized(code: str, *args: str) -> subprocess.CompletedProcess:
    """Run code with python -O, where assert statements are removed."""
    env = dict(os.environ)
    paths = [str(EXAMPLES), str(Path(__file__).resolve().parent), env.get('PYTHONPATH')]
    env['PYTHONPATH'] = os.pathsep.join(filter(None, paths))
    return subprocess.run([sys.executable, '-O', *(['-c', code] if code else []), *args],
                          capture_output=True, text=True, env=env, timeout=110)


# Fixture "generated answers": one per action, and one that names no direction.
FIXTURE_ANSWERS = ('Action: up', 'I would go down.', 'left', 'go right')
UNSURE_ANSWER = 'not sure'


def _tiny_fixture(drift_after: int | None = None, *, answers: bool = False):
    """A fixed-weight one-layer policy over the mean frame colour.

    With drift_after=n, calls after the first n choose a different action.
    With answers=True, the policy returns an EnvironmentResponse built from a
    fixture answer, like the SmolVLM policy, instead of a bare integer. While
    the player is in the left column it answers UNSURE_ANSWER, so the
    fallback is exercised too.
    """
    import torch

    torch.manual_seed(0)
    model = torch.nn.Sequential()
    model.add_module('proj', torch.nn.Linear(3, 8))
    calls = []

    def policy(observation, history):
        calls.append(1)
        x = torch.from_numpy(observation['frame'].mean(axis=(0, 1)) / 255).float()
        with torch.no_grad():
            scores = model(x)[:4]
        drifted = drift_after is not None and len(calls) > drift_after
        choice = int(scores.argmin() if drifted else scores.argmax())
        if answers:
            from native_environment_toolbox import answer_response
            unsure = any(row.startswith('P') for row in observation['ascii'].splitlines())
            return answer_response(UNSURE_ANSWER if unsure else FIXTURE_ANSWERS[choice])
        return choice

    return policy, model


DRIFT_RUN = '''
import sys
from pathlib import Path
from test_external_qualification import _tiny_fixture
from native_environment_toolbox import direct_episode, qualify
out, assets = Path(sys.argv[1]), Path(sys.argv[2])
# Drift right after the direct loop, so the recorded run must disagree with it.
reference = _tiny_fixture()[0]
calls = sum(len(direct_episode(reference, s, size=4, max_steps=6)['actions']) for s in (0, 1))
qualify(out, lambda: _tiny_fixture(drift_after=calls), asset_dir=assets, layer='proj',
        identifier='tiny-fixture', provenance={'fixture': True})
'''


@pytest.mark.timeout(120)
def test_gates_raise_under_python_optimize():
    result = _optimized(
        'import example_qualification as t\n'
        'from types import SimpleNamespace\n'
        'import numpy as np\n'
        'event = lambda when, x: {"kind": "activity", "payload": {"when": when, "value": {"array": x}}}\n'
        'record = SimpleNamespace(events=lambda: [event("before", np.ones(4)), event("after", np.ones(4))])\n'
        't.check_ablation(record, [0, 1], 4)\n')
    assert result.returncode != 0
    assert 'Qualification check failed: selected units are zero' in result.stderr


@pytest.mark.parametrize('max_steps', [1, 3, 40])
def test_direct_loop_and_session_end_on_the_same_step(max_steps):
    """Goal reached and step limit both stop the direct loop and the session alike."""
    from brainscore.harnesses.grid_game import GridGameEnv, greedy_oracle_policy, play_game
    from brainscore.model_helpers.policy_wrapper import PolicyWrapper
    from brainscore_core.model_interface import BrainScoreModel
    from native_environment_toolbox import direct_episode

    for seed in (0, 1, 5):
        direct = direct_episode(greedy_oracle_policy, seed, size=4, max_steps=max_steps)
        subject = BrainScoreModel(identifier='oracle', policy=PolicyWrapper(greedy_oracle_policy))
        session = play_game(subject, GridGameEnv(size=4, seed=seed, max_steps=max_steps), seed=seed)
        assert session['actions'] == direct['actions']
        assert session['total_reward'] == round(direct['total_reward'], 4)
        assert len(direct['actions']) == min(max_steps, session['optimal_steps'])


def test_native_tiny_fixture_passes_and_restores(tmp_path):
    from native_environment_toolbox import qualify
    assets = tmp_path / 'assets'
    assets.mkdir()
    (assets / 'weights.bin').write_bytes(b'fixture')
    policy, model = _tiny_fixture()
    report = qualify(tmp_path / 'run', lambda: (policy, model), asset_dir=assets, layer='proj',
                     identifier='tiny-fixture', provenance={'fixture': True})
    saved = json.loads((tmp_path / 'run' / 'qualification.json').read_text())
    assert saved == json.loads(json.dumps(report, default=str))
    assert saved['status'] == 'passed'
    assert saved['recording_preserves_actions_rewards'] and saved['restored_actions_match']
    assert saved['ablation']['units'] == 4 and saved['ablation']['selected_outputs_zero']
    assert len(saved['provenance']['model_sha256']) == 64
    packages = saved['provenance']['environment']['packages']
    assert {'gymnasium', 'minigrid', 'torch'} <= set(packages)
    assert not any(m._forward_hooks for m in model.modules())


@pytest.mark.parametrize('answer, action, valid', [
    ('Action: left', 2, True),
    ('up, then right', 3, True),
    ('I cannot tell from this image.', 0, False),
    ('', 0, False),
])
def test_answer_response_keeps_answer_and_fallback(answer, action, valid):
    """The applied action is unchanged; the answer and the parse result are kept."""
    from brainscore.model_helpers.local_vlm_policy import parse_directional_action
    from native_environment_toolbox import UNPARSED_ACTION, answer_response

    response = answer_response(answer)
    parsed = parse_directional_action(answer, prefer_last=True)
    assert response.action == action == (parsed if parsed >= 0 else UNPARSED_ACTION)
    assert response.metadata['answer'] == answer
    assert response.metadata['valid'] is valid
    assert (response.metadata['parse_error'] is None) is valid
    assert response.metadata['fallback_action'] == (None if valid else UNPARSED_ACTION)


def test_with_provenance_accepts_responses_and_bare_actions():
    from brainscore_core.events import EnvironmentResponse
    from native_environment_toolbox import answer_response, direct_episode, with_provenance

    provenance = {'model_sha256': 'f' * 64}
    bare = with_provenance(lambda observation, history: 3, provenance=provenance)
    answered = with_provenance(lambda observation, history: answer_response('Action: down'),
                               provenance=provenance)
    assert bare(None, ()) == EnvironmentResponse(action=3, metadata={'provenance': provenance})
    response = answered(None, ())
    assert response.action == 1
    assert response.metadata['provenance'] == provenance
    assert response.metadata['answer'] == 'Action: down' and response.metadata['valid']
    response.metadata['provenance']['model_sha256'] = 'changed by a consumer'
    assert answered(None, ()).metadata['provenance'] == provenance
    # The direct loop applies the same actions for both policy return types.
    for seed in (0, 1):
        plain = direct_episode(lambda o, h: 1, seed, size=4, max_steps=6)
        wrapped = direct_episode(lambda o, h: EnvironmentResponse(action=1), seed, size=4, max_steps=6)
        assert plain == wrapped


def _responses(directory):
    """The EnvironmentResponses the environment applied, from one Experiment's record."""
    from brainscore.run_record import RunRecord
    from brainscore_core.events import EnvironmentResponse
    return [event['payload'].payload for event in RunRecord(directory).events()
            if event['kind'] == 'output'
            and isinstance(getattr(event['payload'], 'payload', None), EnvironmentResponse)
            and event['payload'].meta.get('action_status') == 'applied']


def test_native_records_response_provenance_and_answers(tmp_path):
    """Recorded responses keep provenance and the generated answer; actions are unchanged."""
    from native_environment_toolbox import answer_response, qualify
    assets = tmp_path / 'assets'
    assets.mkdir()
    (assets / 'weights.bin').write_bytes(b'fixture')
    policy, model = _tiny_fixture(answers=True)
    report = qualify(tmp_path / 'run', lambda: (policy, model), asset_dir=assets, layer='proj',
                     identifier='tiny-fixture', provenance={'fixture': True})
    assert report['status'] == 'passed', report.get('error')
    provenance = report['provenance']
    assert provenance['fixture'] is True and len(provenance['model_sha256']) == 64

    seen_valid = set()
    for condition in ('recorded', 'ablated', 'restored'):
        plan = json.loads((tmp_path / 'run' / condition / 'experiment.json').read_text())['plan']
        assert plan['metadata']['provenance'] == json.loads(json.dumps(provenance, default=str))
        if condition == 'restored':
            continue  # No RecordInputsOutputs in the restored run.
        responses = _responses(tmp_path / 'run' / condition)
        actions = [a for seed in map(str, (0, 1)) for a in (
            report['direct'][seed]['actions'] if condition == 'recorded'
            else report['ablation']['actions'][seed])]
        assert [r.action for r in responses] == actions
        for response in responses:
            assert response.metadata['provenance'] == provenance
            answer = response.metadata['answer']
            assert answer in (*FIXTURE_ANSWERS, UNSURE_ANSWER)
            expected = answer_response(answer)
            assert response.action == expected.action
            for key in ('valid', 'parse_error', 'fallback_action'):
                assert response.metadata[key] == expected.metadata[key]
            seen_valid.add(response.metadata['valid'])
    assert seen_valid == {True, False}, 'the fixture exercises both a parsed answer and the fallback'


@pytest.mark.timeout(120)
def test_native_mismatch_fails_under_optimize_and_keeps_provenance(tmp_path):
    assets = tmp_path / 'assets'
    assets.mkdir()
    (assets / 'weights.bin').write_bytes(b'fixture')
    result = _optimized(DRIFT_RUN, str(tmp_path / 'run'), str(assets))
    assert result.returncode != 0
    report = json.loads((tmp_path / 'run' / 'qualification.json').read_text())
    assert report['status'] == 'failed'
    assert report['error'].startswith('RuntimeError: Qualification check failed')
    assert report['provenance']['fixture'] is True
    assert len(report['provenance']['model_sha256']) == 64
    assert report['direct']  # Evidence gathered before the failure is kept.
    assert 'restored_actions_match' not in report


@pytest.mark.timeout(120)
@pytest.mark.parametrize('missing', ['model', 'checkout'])
def test_coggym_missing_assets_write_failed_report(tmp_path, missing):
    model = tmp_path / 'model'
    if missing != 'model':
        model.mkdir()
        (model / 'config.json').write_text('{}')
    out = tmp_path / 'out'
    result = _optimized('', str(EXAMPLES / 'coggym_toolbox.py'), '--checkout', str(tmp_path / 'coggym'),
                        '--model-path', str(model), '--output-dir', str(out))
    assert result.returncode != 0
    report = json.loads((out / 'qualification.json').read_text())
    assert report['status'] == 'failed'
    assert report['error'].startswith('FileNotFoundError: Stage')
    assert report['leaderboard_reproduction'] is False
    # Nothing is hashed or attributed to the pinned model before both inputs are staged.
    assert 'model_sha256' not in report['provenance']
    assert 'model_identity' not in report['provenance']
    assert 'unified_source' in report['provenance']['environment']


@pytest.mark.slow
def test_native_trained_smolvlm_qualification(tmp_path, request):
    from huggingface_hub import snapshot_download
    from example_qualification import SMOLVLM_MANIFEST
    from native_environment_toolbox import MODEL_ID, MODEL_REVISION, load_smolvlm_policy, qualify
    from brainscore.validation.integration_checks import record_placement

    # Resolve the exact pinned commit from the staged cache before any inference.
    path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    assert path.name == MODEL_REVISION

    def load():
        policy, model = load_smolvlm_policy(path, max_new_tokens=32)
        record_placement(request.config, model, identifier=MODEL_ID, dtype='torch.float32')
        return policy, model

    report = qualify(tmp_path / 'run', load, asset_dir=path, manifest=SMOLVLM_MANIFEST, provenance={
        'model_id': MODEL_ID, 'revision': MODEL_REVISION, 'dtype': 'float32',
        'device': 'cpu', 'decoding': 'greedy', 'max_new_tokens': 32})
    assert report['status'] == 'passed', report.get('error')


# Model identity: a tiny fixture snapshot and a manifest built for it.

FIXTURE_FILES = {'config.json': b'{"model_type": "fixture"}\n',
                 'tokenizer.json': b'{"version": "1.0"}\n',
                 'model.safetensors': b'fixture weights'}
FIXTURE_REVISION = 'a' * 40


def _fixture_manifest():
    import hashlib
    files = {}
    for name, content in FIXTURE_FILES.items():
        if name.endswith('.safetensors'):
            files[name] = {'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        else:
            blob = b'blob %d\0' % len(content) + content
            files[name] = {'size': len(content), 'git_sha1': hashlib.sha1(blob).hexdigest()}
    return {'repository': 'fixture/model', 'revision': FIXTURE_REVISION, 'files': files,
            'ignored': ['README.md', 'onnx', '.cache']}


def _local_dir(root, *, record_revision=FIXTURE_REVISION):
    """The layout huggingface-cli download --local-dir writes."""
    root.mkdir()
    records = root / '.cache' / 'huggingface' / 'download'
    records.mkdir(parents=True)
    for name, content in FIXTURE_FILES.items():
        (root / name).write_bytes(content)
        (records / f'{name}.metadata').write_text(f'{record_revision}\netag\n1700000000.0\n')
    (root / 'README.md').write_text('not read by the loader')
    return root


def _cache_snapshot(root, revision=FIXTURE_REVISION):
    """The Hugging Face cache layout: snapshots/<commit>/<file> links into blobs/."""
    blobs, snapshot = root / 'blobs', root / 'snapshots' / revision
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for index, (name, content) in enumerate(FIXTURE_FILES.items()):
        (blobs / str(index)).write_bytes(content)
        (snapshot / name).symlink_to(Path('..', '..', 'blobs', str(index)))
    return snapshot


def test_git_blob_hash_matches_git():
    from example_qualification import file_hash
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'hello'
        path.write_bytes(b'hello\n')
        # `printf 'hello\n' | git hash-object --stdin`
        assert file_hash(path, 'git_sha1') == 'ce013625030ba8dba906f756967f9e9ca394464a'


@pytest.mark.parametrize('layout', ['local_dir', 'cache'])
def test_staged_snapshot_identity_is_verified(tmp_path, layout):
    from example_qualification import verify_model_identity
    path = _local_dir(tmp_path / 'm') if layout == 'local_dir' else _cache_snapshot(tmp_path / 'hub')
    identity = verify_model_identity(path, _fixture_manifest())
    assert identity['revision'] == FIXTURE_REVISION
    assert identity['verified_files'] == sorted(FIXTURE_FILES)


@pytest.mark.parametrize('defect, message', [
    ('missing', 'tokenizer.json is missing'),
    ('tampered', 'model.safetensors does not match'),
    ('resized', 'config.json has the wrong size'),
    ('extra', "unexpected files in the model snapshot: ['adapter_config.json']"),
    ('record', 'download record for config.json names another revision'),
    ('snapshot', 'is not the pinned revision'),
])
def test_wrong_identity_fails(tmp_path, defect, message):
    from example_qualification import verify_model_identity
    if defect == 'snapshot':
        path = _cache_snapshot(tmp_path / 'hub', revision='b' * 40)
    else:
        path = _local_dir(tmp_path / 'm')
    if defect == 'missing':
        (path / 'tokenizer.json').unlink()
    elif defect == 'tampered':
        (path / 'model.safetensors').write_bytes(b'fixture weightz')  # same size
    elif defect == 'resized':
        (path / 'config.json').write_bytes(b'{}')
    elif defect == 'extra':
        (path / 'adapter_config.json').write_text('{}')  # could change what loads
    elif defect == 'record':
        record = path / '.cache' / 'huggingface' / 'download' / 'config.json.metadata'
        record.write_text('b' * 40 + '\netag\n0\n')
    with pytest.raises(RuntimeError, match='Qualification check failed: .*' + re.escape(message)):
        verify_model_identity(path, _fixture_manifest())


def test_pinned_manifest_lists_the_loader_files():
    from example_qualification import SMOLVLM_MANIFEST
    from native_environment_toolbox import MODEL_REVISION
    assert SMOLVLM_MANIFEST['revision'] == MODEL_REVISION == '7e3e67edbbed1bf9888184d9df282b700a323964'
    files = SMOLVLM_MANIFEST['files']
    assert {'config.json', 'model.safetensors', 'preprocessor_config.json', 'processor_config.json',
            'tokenizer.json', 'tokenizer_config.json', 'chat_template.json'} <= set(files)
    assert all(len(f.get('sha256', '')) == 64 or len(f.get('git_sha1', '')) == 40
               for f in files.values())
    assert 'sha256' in files['model.safetensors']


def test_native_identity_mismatch_fails_before_load(tmp_path):
    from native_environment_toolbox import qualify
    path = _local_dir(tmp_path / 'm')
    (path / 'config.json').write_bytes(b'{"model_type": "fixturX"}\n')
    loads = []
    with pytest.raises(RuntimeError, match='config.json does not match'):
        qualify(tmp_path / 'run', lambda: loads.append(1), asset_dir=path,
                manifest=_fixture_manifest(), identifier='tiny-fixture')
    report = json.loads((tmp_path / 'run' / 'qualification.json').read_text())
    assert report['status'] == 'failed' and loads == []
    assert 'model_identity' not in report['provenance']
    assert 'model_sha256' not in report['provenance']


@pytest.mark.timeout(120)
def test_coggym_unverified_model_fails_under_optimize(tmp_path):
    """A directory that is not the pinned snapshot is refused, also under python -O."""
    (tmp_path / 'coggym').mkdir()
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    out = tmp_path / 'out'
    result = _optimized('', str(EXAMPLES / 'coggym_toolbox.py'), '--checkout', str(tmp_path / 'coggym'),
                        '--model-path', str(model), '--output-dir', str(out))
    assert result.returncode != 0
    report = json.loads((out / 'qualification.json').read_text())
    assert report['status'] == 'failed'
    assert report['error'].startswith('RuntimeError: Qualification check failed: model file')
    assert 'model_identity' not in report['provenance']
    assert 'model_sha256' not in report['provenance'] and 'coggym_source' not in report['provenance']
