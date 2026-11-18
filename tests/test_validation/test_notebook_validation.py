"""Check notebook failure accounting without needing tutorial dependencies."""
import json

import pytest

from brainscore.validation import notebooks


@pytest.mark.parametrize('condition', ['pass', 'failure', 'missing', 'new'])
def test_notebook_selection_fails_closed(tmp_path, monkeypatch, condition):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'assets').mkdir()
    (source / 'assets/figure.png').write_bytes(b'figure bytes')
    monkeypatch.setattr(notebooks, 'SELECTED', ('tutorial.ipynb',))
    monkeypatch.setattr(notebooks, 'EXCLUDED', {})
    if condition != 'missing':
        (source / 'tutorial.ipynb').write_text('{}')
    if condition == 'new':
        (source / 'unclassified.ipynb').write_text('{}')

    def execute(path, output, timeout):
        if condition == 'failure':
            raise RuntimeError('tutorial failed')
        output.write_text('{}')

    monkeypatch.setattr(notebooks, 'execute', execute)
    report = notebooks.run(source, tmp_path / 'outputs')
    assert report['passed'] == (condition == 'pass')
    assert json.loads((tmp_path / 'outputs/report.json').read_text()) == report
    assert (tmp_path / 'outputs/assets/figure.png').read_bytes() == b'figure bytes'
    if condition == 'failure':
        assert 'tutorial failed' in report['results'][0]['error']


def test_execution_retains_failure_without_changing_source(tmp_path):
    import nbformat

    source = tmp_path / 'tutorial.ipynb'
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell(
            "import torch\n"
            "assert not torch.cuda.is_available() and not torch.backends.mps.is_available()\n"
            "print('executed before failure')"
        ),
        nbformat.v4.new_code_cell("raise ValueError('broken tutorial')"),
    ])
    nbformat.write(notebook, source)
    original = source.read_bytes()
    output = tmp_path / 'executed.ipynb'
    with pytest.raises(Exception, match='broken tutorial'):
        notebooks.execute(source, output, timeout=30)
    executed = nbformat.read(output, as_version=4)
    assert 'executed before failure' in executed.cells[1].outputs[0].text
    assert executed.cells[2].outputs[0].ename == 'ValueError'
    assert source.read_bytes() == original


def test_missing_figure_fails_before_execution(tmp_path):
    import nbformat

    source = tmp_path / 'tutorial.ipynb'
    nbformat.write(nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('![Measured responses](assets/missing.png)'),
    ]), source)
    with pytest.raises(FileNotFoundError, match='Missing notebook image'):
        notebooks.execute(source, tmp_path / 'output.ipynb', timeout=30)


def test_reject_overwriting_sources(tmp_path):
    with pytest.raises(ValueError, match='must not overwrite'):
        notebooks.run(tmp_path, tmp_path)


def test_interrupted_rerun_cannot_reuse_a_passing_report(tmp_path, monkeypatch):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    output.mkdir()
    (source / 'tutorial.ipynb').write_text('{}')
    (output / 'report.json').write_text('{"passed": true}')
    monkeypatch.setattr(notebooks, 'SELECTED', ('tutorial.ipynb',))
    monkeypatch.setattr(notebooks, 'EXCLUDED', {})

    def interrupt(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(notebooks, 'execute', interrupt)
    with pytest.raises(KeyboardInterrupt):
        notebooks.run(source, output)
    assert not json.loads((output / 'report.json').read_text())['passed']
