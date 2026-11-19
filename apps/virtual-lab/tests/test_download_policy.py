"""The lab's download switch stays local to its worker and honors both states."""
import json
import os
from queue import Queue
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from umi_lab.server import LabController, make_server
from umi_lab.worker import apply_download_policy


def test_policy_controls_the_existing_guard_without_a_download(tmp_path, monkeypatch):
    from brainscore.models._downloads import file_preflight, ModelDownloadRequired
    monkeypatch.setenv('BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK', '0')
    destination = tmp_path / 'missing.bin'
    apply_download_policy(True)
    file_preflight('fixture', 'https://example.invalid/weights', destination, .001)
    assert not destination.exists()
    apply_download_policy(False)
    with pytest.raises(ModelDownloadRequired):
        file_preflight('fixture', 'https://example.invalid/weights', destination, .001)


def test_worker_defaults_on_and_restores_guard_without_changing_parent_env(tmp_path, monkeypatch):
    from umi_lab.benchmark import wait
    log = tmp_path / 'policy.json'
    module = tmp_path / 'lab_download_profile.py'
    module.write_text(
        'import json,os\nfrom pathlib import Path\nfrom umi_lab.demo import create_profile as calibration\n'
        'def create_profile():\n'
        f'    log=Path({str(log)!r})\n'
        '    entries=json.loads(log.read_text()) if log.exists() else []\n'
        '    entries.append(os.environ.get("BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK"))\n'
        '    log.write_text(json.dumps(entries))\n'
        '    return calibration()\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK', '0')
    controller = LabController(tmp_path / 'runs', 'lab_download_profile:create_profile')
    try:
        assert wait(controller, ready=True)['allow_model_downloads'] is True
        for allow in (False, True):
            controller.configure_downloads({'allow_model_downloads': allow})
            controller.load({'identifier': '__calibration__'})
            assert wait(controller, ready=True)['allow_model_downloads'] is allow
        assert json.loads(log.read_text()) == ['1', '0', '1']
        assert os.environ['BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK'] == '0'
    finally:
        controller.close()


def controller_stub():
    controller = LabController.__new__(LabController)
    controller.lock = threading.Lock()
    controller.state = {'phase': 'ready', 'run': None, 'allow_model_downloads': True}
    controller.commands = Queue()
    controller.token = 'download-policy-fixture'
    return controller


def test_settings_validate_and_reach_the_next_operation():
    controller = controller_stub()
    for bad in ({}, {'allow_model_downloads': 1}, {'allow_model_downloads': 'false'},
                {'allow_model_downloads': False, 'extra': True}):
        with pytest.raises(ValueError):
            controller.configure_downloads(bad)
    controller.configure_downloads({'allow_model_downloads': False})
    controller._enqueue({'kind': 'validate'})
    assert controller.commands.get_nowait() == {'kind': 'validate', 'allow_model_downloads': False}
    for phase, run in [('loading', None), ('validating', None), ('ready', {'phase': 'running'})]:
        controller.state.update(phase=phase, run=run)
        with pytest.raises(ValueError, match='finishes'):
            controller.configure_downloads({'allow_model_downloads': True})
    assert controller.state['allow_model_downloads'] is False


def test_settings_endpoint_requires_the_lab_token(tmp_path):
    controller = controller_stub()
    server = make_server(controller, tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/api/settings'
    body = json.dumps({'allow_model_downloads': False}).encode()
    headers = {'Content-Type': 'application/json'}
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url, body, headers))
        assert error.value.code == 403
        with urlopen(Request(url, body, {**headers, 'X-Lab-Token': controller.token})) as response:
            assert json.load(response) == {'allow_model_downloads': False}
        assert controller.snapshot()['allow_model_downloads'] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
