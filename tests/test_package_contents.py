"""Check package errors without installing malformed wheels."""
import json
from zipfile import ZipFile

import pytest

from brainscore.validation.package_contents import inspect_wheel


def make_wheel(tmp_path, extra=None, missing=None):
    files = {
        'brainscore/models/vjepa_v1/backbone/LICENSE': 'license',
        'brainscore/models/vjepa_v1/backbone/NOTICE.md': 'attribution',
        'brainscore/benchmarks/rajalingham2018_2afc/trials.csv': 'trial\n',
    }
    files.update(extra or {})
    if missing:
        files.pop(missing)
    path = tmp_path / 'brainscore-test.whl'
    with ZipFile(path, 'w') as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return path


def test_portable_notebook_is_kept(tmp_path):
    notebook = {'cells': [{'cell_type': 'code', 'source': ['data = input("Dataset path: ")'],
                           'outputs': [], 'execution_count': None}]}
    wheel = make_wheel(tmp_path, {'brainscore/example.ipynb': json.dumps(notebook)})
    assert inspect_wheel(wheel, 'brainscore')['errors'] == []


@pytest.mark.parametrize('name', ['tests/__init__.py', 'scripts/run.py',
                                  'brainscore/__pycache__/model.pyc'])
def test_rejects_accidental_packages_and_bytecode(tmp_path, name):
    assert inspect_wheel(make_wheel(tmp_path, {name: ''}), 'brainscore')['errors']


def test_requires_third_party_notice(tmp_path):
    wheel = make_wheel(tmp_path, missing='brainscore/models/vjepa_v1/backbone/LICENSE')
    assert inspect_wheel(wheel, 'brainscore')['errors']


@pytest.mark.parametrize('source,outputs', [(['path = "/Users/author/data"'], []),
                                           (['print(1)'], [{'text': 'old output'}])])
def test_rejects_personal_notebook_state(tmp_path, source, outputs):
    notebook = {'cells': [{'cell_type': 'code', 'source': source, 'outputs': outputs}]}
    wheel = make_wheel(tmp_path, {'brainscore/example.ipynb': json.dumps(notebook)})
    assert inspect_wheel(wheel, 'brainscore')['errors']


def test_rejects_duplicate_or_missing_wheels(tmp_path, monkeypatch):
    from brainscore.validation.package_contents import main
    output = tmp_path / 'report.json'
    monkeypatch.setattr('sys.argv', ['package_contents', '--wheels', str(tmp_path),
                                    '--out', str(output)])
    assert main() == 1
    assert all(row['errors'] for row in json.loads(output.read_text()).values())
