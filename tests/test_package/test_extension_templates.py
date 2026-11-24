"""Run the capability and benchmark templates as shipped.

The capability template is registered with isolated registries and driven
through ``Session`` + ``subject.participate``. The benchmark template
scores a subject that implements only ``participate``, and scores a
``BrainScoreModel`` exactly as the earlier ``start_recording`` + ``process``
code did. All inputs are small synthetic fixtures; no scores here are evidence
about any model. The model template needs Torch; it is exercised by
``test_api_cohesion.py::test_new_model_template_supports_public_recording_workflow``.
"""
import ast
import importlib.util
import math
import re
import zlib
from pathlib import Path

import numpy as np
import pytest

from brainscore_core import io_catalog
from brainscore_core.compatibility import check_compatibility, ExecutionNotFeasibleError
from brainscore import BrainScoreModel
from brainscore_core.contract import Subject
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.streaming_helpers import neural_response
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

pytestmark = pytest.mark.unit

TEMPLATES = Path(__file__).parents[2] / 'templates'
OLD_SLOT_WORDS = re.compile(r'\b(generation_fn|action_fn|state_change_fn|activations_model|preprocessors)\b')


def _load(relative, name):
    spec = importlib.util.spec_from_file_location(name, TEMPLATES / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_package(folder, name):
    """Import a template folder as a package, so its relative imports resolve."""
    import sys
    spec = importlib.util.spec_from_file_location(name, TEMPLATES / folder / '__init__.py',
                                                  submodule_search_locations=[str(TEMPLATES / folder)])
    module = importlib.util.module_from_spec(spec)
    for loaded in [key for key in sys.modules if key.startswith(name + '.')]:
        del sys.modules[loaded]                        # run the submodules again, too
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def test_templates_teach_only_public_slot_names():
    for path in sorted(TEMPLATES.rglob('*')):
        if path.suffix in ('.py', '.md'):
            hits = [line for line in path.read_text(encoding='utf-8').splitlines()
                    if OLD_SLOT_WORDS.search(line)]
            assert not hits, (path, hits)


# --- new_capability ---------------------------------------------------------------

@pytest.fixture
def capability(monkeypatch):
    monkeypatch.setattr(io_catalog, '_CATALOG', dict(io_catalog._CATALOG))
    monkeypatch.setattr(io_catalog, '_STIMULUS_COLUMNS', dict(io_catalog._STIMULUS_COLUMNS))
    module = _load('new_capability/capability.py', 'umi_template_capability')
    assert not io_catalog.has(module.REQUEST)                # importing registers nothing
    module.register()
    yield module


def _requests(module, *texts):
    return [StreamEvent(channel=module.REQUEST, payload=module.WordRequest(text=text),
                        t_ms=10.0 * index, meta={'stimulus_id': f'r{index}'})
            for index, text in enumerate(texts)]


def test_capability_runs_through_participate_with_state_and_reset(capability):
    subject = capability.build_model(min_length=3)
    assert subject.check_session_support({capability.RESULT}).identifier == capability.IDENTIFIER
    for _ in range(2):   # the second round reuses the same subject after reset
        session = Session(_requests(capability, 'a red ball rolls', 'two cats'), [capability.RESULT])
        subject.participate(session)
        answers = session.collect(capability.RESULT)
        assert [event.payload for event in answers] == [
            capability.WordCount(words=3, words_so_far=3),    # 'a' is shorter than 3
            capability.WordCount(words=2, words_so_far=5)]
        assert [(event.t_ms, event.meta['stimulus_id']) for event in answers] == [(0.0, 'r0'), (10.0, 'r1')]
        subject.reset()


def test_capability_is_opt_in_per_model(capability):
    plain = BrainScoreModel(identifier='plain')
    assert not {capability.REQUEST, capability.RESULT} & (plain.in_channels | plain.out_channels)
    with pytest.raises(ExecutionNotFeasibleError, match='does not produce'):
        plain.check_session_support({capability.RESULT})
    opted = capability.build_model()
    assert capability.REQUEST in opted.in_channels and capability.RESULT in opted.out_channels


@pytest.mark.parametrize('payload', ['text', {'text': 'a b'}, None])
def test_capability_rejects_wrong_payload_types(capability, payload):
    with pytest.raises(ValueError, match='channel catalog'):
        Session([StreamEvent(channel=capability.REQUEST, payload=payload, t_ms=0.0)], [capability.RESULT])


@pytest.mark.parametrize('text', [42, None, b'a b', ['a', 'b']])
def test_capability_rejects_invalid_request_fields_before_compute(capability, text):
    request = capability.WordRequest(text=text)
    with pytest.raises(ValueError, match='channel catalog'):
        Session([StreamEvent(channel=capability.REQUEST, payload=request, t_ms=0.0)], [capability.RESULT])


@pytest.mark.parametrize('words,words_so_far', [(-1, 0), (2, 1), (True, 1), (1.0, 1), (1, None)])
def test_capability_result_validator_rejects_invalid_counts(capability, words, words_so_far):
    result = capability.WordCount(words=words, words_so_far=words_so_far)
    assert io_catalog.validate(capability.RESULT, result, io_catalog.OUTPUT)


def test_capability_model_refuses_an_invalid_result(capability):
    for valid in (capability.WordCount(words=0, words_so_far=0), capability.WordCount(words=2, words_so_far=5)):
        assert not io_catalog.validate(capability.RESULT, valid, io_catalog.OUTPUT)
    counter = capability.WordCountCapability()
    subject = BrainScoreModel(identifier='direct', capabilities=[counter])
    subject.participate(Session(_requests(capability, 'a b'), [capability.RESULT]))
    counter.words_so_far = -10   # corrupt the running total
    with pytest.raises(ValueError, match='shape_validator'):   # the session checks each answer
        subject.participate(Session(_requests(capability, 'a b'), [capability.RESULT]))


@pytest.mark.parametrize('min_length', [True, False, None, 0, -1, 1.5, '2', 2.0])
def test_capability_checks_settings_when_built(capability, min_length):
    with pytest.raises(ValueError, match='min_length'):
        capability.build_model(min_length=min_length)


def test_capability_registers_once(capability):
    with pytest.raises(ValueError, match='already registered'):
        capability.register()


def test_capability_readme_example_runs():
    text = (TEMPLATES / 'new_capability/README.md').read_text(encoding='utf-8')
    blocks = re.findall(r'```python\n(.*?)```', text, flags=re.S)
    assert blocks
    namespace = {}
    for block in blocks:
        exec(compile(block, 'new_capability/README.md', 'exec'), namespace)
    from brainscore_core.streaming import respond_event
    session = Session([StreamEvent('observation', {}, None), respond_event()], ['motor'])
    namespace['model'].participate(session)
    assert [event.payload for event in session.emitted] == [0]


# --- every code block of the user documents ----------------------------------------

DOCUMENTS = Path(__file__).parents[2]
# Blocks that need weights, data or a GPU, by (page, index of its python block).
NEEDS_DOWNLOADS = {
    ('docs/guides/score_a_model.md', 1): 'CLIP weights and the MajajHong2015 data',
    ('docs/guides/existing_models.md', 0): 'alexnet weights and the MajajHong2015 data',
    ('docs/guides/data.md', 0): 'the hosted LeBel2023 assembly',
    ('docs/guides/data.md', 1): 'the EWoK native table and the LeBel source pickle',
    ('docs/guides/behavior.md', 7): 'a Hugging Face chat model and its processor',
    ('docs/guides/wrap_your_model.md', 6): 'CLIP and its image and text wrappers',
    ('docs/guides/new_output.md', 3): 'a Hugging Face vision-language model and its processor',
}


def _documented_pages():
    pages = [DOCUMENTS / 'README.md', DOCUMENTS / 'EXTENDING.md']
    for folder in ('docs', 'docs/guides', 'docs/reference'):
        pages += sorted((DOCUMENTS / folder).glob('*.md'))
    pages += sorted(TEMPLATES.glob('**/README.md'))
    return [page for page in pages
            if re.search(r'^```python', page.read_text(encoding='utf-8'), flags=re.M)]


def test_new_output_episode_ends_as_documented():
    """The screen task of the new-output guide ends with a click on its third step."""
    text = (DOCUMENTS / 'docs/guides/new_output.md').read_text(encoding='utf-8')
    blocks = re.findall(r'^```python[^\n]*\n(.*?)^```', text, flags=re.S | re.M)
    namespace = {}
    exec(compile('\n'.join(blocks[:3]), 'new_output.md', 'exec'), namespace)
    assert namespace['reason'] == 'terminal'
    assert [namespace['KINDS'][step['kind']] for step in namespace['actions']] == ['run', 'run', 'click']


@pytest.mark.integration
@pytest.mark.parametrize('page', _documented_pages(),
                         ids=lambda page: str(page.relative_to(DOCUMENTS)))
def test_documented_code_blocks_run(page, tmp_path):
    """Each page's python blocks run in order, as written, in a fresh interpreter.

    The template READMEs and EXTENDING.md import the templates by folder name, so they
    run from templates/; every other page runs in an empty folder. Blocks that need a
    download are left out (NEEDS_DOWNLOADS, with the reason).
    """
    import os
    import subprocess
    import sys
    relative = str(page.relative_to(DOCUMENTS))
    blocks = re.findall(r'^```python[^\n]*\n(.*?)^```', page.read_text(encoding='utf-8'),
                        flags=re.S | re.M)
    code, expected = _with_output_markers([block for index, block in enumerate(blocks)
                                           if (relative, index) not in NEEDS_DOWNLOADS])
    in_templates = relative.startswith('templates/') or relative == 'EXTENDING.md'
    environment = dict(os.environ, RESULTCACHING_DISABLE='1', MPLBACKEND='Agg')
    result = subprocess.run([sys.executable, '-'], input=code, capture_output=True, text=True,
                            cwd=TEMPLATES if in_templates else tmp_path, env=environment,
                            timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    printed = dict(re.findall(r'@@start (\d+)\n(.*?)@@end', result.stdout, flags=re.S))
    for index, (line, comment) in enumerate(expected):
        assert _same_output(printed[str(index)], comment), (
            f'{relative}, line {line}: the page states\n{comment}\nthe code prints\n{printed[str(index)]}')


def _with_output_markers(blocks):
    """The code of the blocks, with markers around each top-level statement that prints and
    is followed by comment lines: those lines state its output. Returns the code and, for
    each such statement, its line in the block and the stated output."""
    expected, code = [], []
    for block in blocks:
        lines = block.splitlines()
        inserts = []                                   # (index in lines, text)
        for node in ast.parse(block).body:
            if not any(isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'print'
                       for n in ast.walk(node)):
                continue
            end = node.end_lineno
            comment = []
            while end + len(comment) < len(lines) and lines[end + len(comment)].startswith('#'):
                comment.append(lines[end + len(comment)].lstrip('#').strip())
            if comment:
                inserts += [(node.lineno - 1, f"print('@@start {len(expected)}')"), (end, "print('@@end')")]
                expected.append((node.lineno, ' '.join(comment)))
        for position, text in sorted(inserts, reverse=True):
            lines.insert(position, text)
        code.append('\n'.join(lines))
    return '\n'.join(code), expected


def _same_output(printed, stated):
    """Equal word by word; numbers equal to six digits."""
    printed, stated = printed.split(), stated.split()
    if len(printed) != len(stated):
        return False
    for a, b in zip(printed, stated):
        try:
            if not math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9):
                return False
        except ValueError:
            if a != b:
                return False
    return True


# --- new_benchmark -----------------------------------------------------------------

@pytest.fixture
def benchmark():
    import brainscore
    # The template registers 'your-benchmark' on import; remove it afterwards.
    _load_package('new_benchmark', 'umi_template_benchmark')
    try:
        yield brainscore.benchmark_registry['your-benchmark']()
    finally:
        del brainscore.benchmark_registry['your-benchmark']


class ParticipateOnlySubject(Subject):
    """Deterministic subject with no start_recording or process."""
    identifier = 'participate-only'
    in_channels = {'vision:image'}
    out_channels = {'neural:IT'}

    def participate(self, session):
        event = session.next_input()
        while event is not None:
            stimulus_id = event.meta['stimulus_id']
            values = np.random.RandomState(zlib.crc32(str(stimulus_id).encode())).randn(1, 16)
            response = NeuroidAssembly(
                values,
                coords={'stimulus_id': ('presentation', [stimulus_id]),
                        'stimulus_index': ('presentation', [event.meta['stimulus_index']]),
                        'neuroid_id': ('neuroid', [f'IT.{unit}' for unit in range(16)]),
                        'region': ('neuroid', ['IT'] * 16)},
                dims=['presentation', 'neuroid'])
            session.respond(StreamEvent(channel='neural:IT', payload=response,
                                        t_ms=event.t_ms, meta=dict(event.meta)))
            event = session.next_input()


class StimulusExtractor:
    """Deterministic extractor returning a real assembly that carries stimulus metadata."""

    def __init__(self):
        self.calls = []

    def __call__(self, stimuli, layers):
        self.calls.append((list(stimuli['stimulus_id']), list(layers)))
        values = np.array([np.random.RandomState(zlib.crc32(sid.encode())).randn(8)
                           for sid in stimuli['stimulus_id']], dtype=np.float32)
        coords = {column: ('presentation', list(stimuli[column].values)) for column in stimuli.columns}
        coords['neuroid_id'] = ('neuroid', [f'unit{i}' for i in range(8)])
        coords['layer'] = ('neuroid', [layers[0]] * 8)
        return NeuroidAssembly(values, coords=coords, dims=['presentation', 'neuroid'])


def test_benchmark_keeps_its_identity_and_scores_a_participate_only_subject(benchmark):
    assert (benchmark.identifier, benchmark.version) == ('your-benchmark', 1)
    assert benchmark.ceiling.attrs['status'] == 'not_estimated'
    assert benchmark.parent == 'neural'
    assert len(benchmark.stimulus_set) == 32
    subject = ParticipateOnlySubject()
    assert not hasattr(subject, 'process') and not hasattr(subject, 'start_recording')
    check_compatibility(subject, benchmark)
    responses = neural_response(subject=subject, stimuli=benchmark.stimulus_set,
                                outputs=['neural:IT'], time_bins=[(70, 170)])
    assert dict(responses.sizes) == {'presentation': 32, 'neuroid': 16}
    assert list(responses['stimulus_id'].values) == list(benchmark.stimulus_set['stimulus_id'])
    score = benchmark(subject)
    assert np.isfinite(float(score))
    assert score.attrs['ceiling'] is benchmark.ceiling and 'raw' in score.attrs
    assert score.attrs['normalized'] is False and float(score) == float(score.attrs['raw'])
    assert float(score) == float(benchmark(ParticipateOnlySubject()))   # deterministic


def test_benchmark_scores_a_model_as_the_earlier_template_did(benchmark):
    """Same stimuli, region, time bins, metric and ceiling as one session for the region."""
    def subject(extractor):
        return BrainScoreModel(identifier='template-fixture', region_layer_map={'IT': 'layer2'},
                               extractor={'vision': extractor})
    from brainscore_core.streaming_helpers import neural_response
    earlier = neural_response(subject(StimulusExtractor()), benchmark.stimulus_set, ['neural:IT'],
                              time_bins=[(70, 170)])
    expected = benchmark._metric(earlier, benchmark._assembly)

    extractor = StimulusExtractor()
    current = subject(extractor)
    check_compatibility(current, benchmark)
    score = benchmark(current)
    assert float(score) == float(expected)   # the ceiling is not estimated: the score is raw
    np.testing.assert_array_equal(score.attrs['raw'].values, expected.values)
    assert extractor.calls == [(list(benchmark.stimulus_set['stimulus_id']), ['layer2'])]


def test_benchmark_template_dummy_candidate_is_scorable(benchmark):
    template_tests = _load('new_benchmark/test.py', 'umi_template_benchmark_tests')
    subject = template_tests.DummySubject()
    score = benchmark(subject)
    assert np.isfinite(float(score))
    assert subject.requested == ('neural:IT',)


# --- new_data ----------------------------------------------------------------------

def test_data_template_check_accepts_a_text_stimulus_set():
    """A language data plugin (only a `sentence` column) passes the template's check."""
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    template_tests = _load('new_data/test_data.py', 'umi_template_data_tests')
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['s0', 's1'], 'sentence': ['a cat', 'a dog']}))
    template_tests.check_dispatchable(stimuli)
    with pytest.raises(AssertionError, match='empty'):
        template_tests.check_dispatchable(
            StimulusSet(pd.DataFrame({'stimulus_id': ['s0'], 'sentence': ['  ']})))


def test_data_template_choices_are_one_row_per_participant_and_trial():
    data = _load('new_data/data.py', 'umi_template_data')
    choices = data.load_choices()
    assert choices.dims == ('presentation',)
    assert choices.sizes['presentation'] == data.N_STIMULI * data.N_SUBJECTS
    assert set(np.unique(choices.values)) <= set(data.CATEGORIES)


# --- new_tool ----------------------------------------------------------------------

def test_tool_template_tests_pass():
    """The template's own tests, run in place."""
    from brainscore.tools import _registry
    template_tests = _load('new_tool/test_tool.py', 'umi_template_tool_tests')
    try:
        template_tests.test_loads_by_identifier()
        template_tests.test_mean_equals_a_direct_recording(template_tests.tiny_subject_and_stimuli())
    finally:
        _registry._TOOLS.pop('your-tool', None)   # the template registers itself for the test


def test_the_benchmark_template_writes_its_stimuli_once_to_one_folder(tmp_path, monkeypatch):
    """Every load reuses one folder in the Brain-Score home; the images are the seeded ones."""
    import io
    from PIL import Image
    import brainscore
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path))
    module = _load('new_benchmark/standin_data.py', 'umi_template_benchmark_stimuli')
    if 'your-benchmark' in brainscore.benchmark_registry:      # the import registers it
        del brainscore.benchmark_registry['your-benchmark']
    first = module.load_assembly().stimulus_set.stimulus_paths
    second = module.load_assembly().stimulus_set.stimulus_paths
    assert first == second
    assert [path.name for path in tmp_path.iterdir()] == ['your_benchmark_stimuli']
    folder = tmp_path / 'your_benchmark_stimuli'
    assert sorted(p.name for p in folder.iterdir()) == sorted(f'{sid}.png' for sid in first)
    for i, sid in enumerate(sorted(first)):
        expected = io.BytesIO()
        Image.fromarray(np.random.RandomState(i).randint(0, 255, (64, 64, 3), dtype=np.uint8)) \
            .save(expected, format='PNG')
        assert (folder / f'{sid}.png').read_bytes() == expected.getvalue()


@pytest.mark.parametrize('template, folder, load', [
    ('new_benchmark/standin_data.py', 'your_benchmark_stimuli', 'load_assembly'),
    ('new_data/data.py', 'your_data_stimuli', '_build_example_files'),
])
def test_a_template_refuses_a_link_where_a_stimulus_goes(tmp_path, monkeypatch, template, folder, load):
    import brainscore
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path))
    module = _load(template, 'umi_template_link_' + folder)
    if 'your-benchmark' in brainscore.benchmark_registry:      # the import registers it
        del brainscore.benchmark_registry['your-benchmark']
    target = tmp_path / 'elsewhere.txt'
    target.write_text('not a stimulus')
    (tmp_path / folder).mkdir()
    (tmp_path / folder / 'stim000.png').symlink_to(target)
    with pytest.raises(ValueError, match='is a link'):
        getattr(module, load)()
    assert target.read_text() == 'not a stimulus'


@pytest.mark.parametrize('template, folder, load', [
    ('new_benchmark/standin_data.py', 'your_benchmark_stimuli', 'load_assembly'),
    ('new_data/data.py', 'your_data_stimuli', '_build_example_files'),
])
def test_two_threads_write_the_template_stimuli_at_once(tmp_path, monkeypatch, template, folder, load):
    import brainscore
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path))
    module = _load(template, 'umi_template_threads_' + folder)
    if 'your-benchmark' in brainscore.benchmark_registry:
        del brainscore.benchmark_registry['your-benchmark']
    with ThreadPoolExecutor(2) as pool:
        results = [pool.submit(getattr(module, load)) for _ in range(2)]
        [result.result() for result in results]                  # both succeed
    written = sorted(p.name for p in (tmp_path / folder).iterdir())
    assert written == [f'stim{i:03d}.png' for i in range(module.N_STIMULI)]   # no staging file left
