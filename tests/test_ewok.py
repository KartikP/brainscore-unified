"""Local data preparation, protocol fidelity and ordinary UMI tool integration."""
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

import brainscore
from brainscore.data.preparation import DataBuilder, data_builder_registry, prepare_dataset
from brainscore.data.ewok.data import load_prepared
from brainscore.data.ewok.data_packaging.prepare import HF_REPOSITORY, HF_REVISION, normalize, resolve
from brainscore.benchmarks.ewok.benchmark import logprob_accuracy
from brainscore.harnesses.ewok import EWoKProvider
from brainscore.model_helpers.response_trace import build_trace_subject
from brainscore.experiments import Experiment, RecordInputsOutputs
from brainscore_core.contract import Subject

pytestmark = pytest.mark.unit


@pytest.fixture
def native(tmp_path):
    # Entirely synthetic text. No EWoK dataset sentences are committed to tests.
    rows = [dict(Domain='synthetic-domain', MetaTemplateID='1', TemplateID=str(i),
                 Version=str(version), Context1=f'fixture context A {i}',
                 Context2=f'fixture context B {i}', Target1=f'fixture target A {i}',
                 Target2=f'fixture target B {i}') for i, version in [(0, 0), (1, 0), (2, 1)]]
    path = tmp_path / 'native.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def prepared(native, tmp_path):
    return prepare_dataset('EWoK-core-1.0', source=native, output=tmp_path / 'build')


def subject(provider):
    return build_trace_subject('fixture', provider=provider, parse=json.loads,
                               provenance={'model': 'synthetic-test'},
                               input_channels=['conditional_logprob_request', 'context_choice_request'])


def test_data_and_benchmark_registrations():
    assert 'EWoK-core-1.0' in brainscore.data_registry
    assert 'EWoK-core-1.0' in brainscore.stimulus_set_registry
    assert {'EWoK-core-1.0-logprobs', 'EWoK-core-1.0-choice'} <= brainscore.benchmark_registry.keys()


def test_benchmark_reads_exactly_the_prepared_items_through_the_registry(prepared):
    rows, manifest = load_prepared(prepared)
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=prepared)
    assert benchmark._rows == rows
    assert benchmark._manifest == manifest


def test_build_round_trip_and_private_permissions(prepared):
    rows, manifest = load_prepared(prepared)
    assert len(rows) == manifest['items'] == 3
    assert manifest['versions'] == ['0', '1']
    assert prepared.stat().st_mode & 0o777 == 0o700
    assert (prepared / 'items.json').stat().st_mode & 0o777 == 0o600
    data = brainscore.load_dataset('EWoK-core-1.0', root=prepared)
    stimuli = brainscore.load_stimulus_set('EWoK-core-1.0', root=prepared)
    assert data.shape == (3, 2)
    assert data.values.tolist() == [[1, 2]] * 3
    assert len(stimuli) == 3
    assert data.attrs['target_type'] == 'correct_context_not_human_recordings'


def test_deterministic_build_and_content_identity(native, prepared, tmp_path):
    other = prepare_dataset('EWoK-core-1.0', source=native, output=tmp_path / 'second')
    assert (other / 'items.json').read_bytes() == (prepared / 'items.json').read_bytes()
    before = brainscore.load_stimulus_set('EWoK-core-1.0', root=prepared).identifier
    native.write_text(native.read_text().replace('fixture context A 0','edited context A 0'))
    changed = prepare_dataset('EWoK-core-1.0', source=native, output=tmp_path / 'third')
    assert brainscore.load_stimulus_set('EWoK-core-1.0', root=changed).identifier != before


def test_no_overwrite_or_partial_build(native, prepared, tmp_path):
    before = (prepared / 'items.json').read_bytes()
    with pytest.raises(FileExistsError):
        prepare_dataset('EWoK-core-1.0', source=native, output=prepared)
    assert (prepared / 'items.json').read_bytes() == before
    native.write_text('wrong,columns\n1,2\n')
    output = tmp_path / 'broken'
    with pytest.raises(ValueError, match='columns'):
        prepare_dataset('EWoK-core-1.0', source=native, output=output)
    assert not output.exists()
    assert not list(tmp_path.glob('.brainscore-build-*'))


def test_corruption_rejected_before_model_use(prepared):
    (prepared / 'items.json').write_text('[]')
    with pytest.raises(ValueError, match='checksum'):
        brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=prepared)


@pytest.mark.parametrize('mutation,match', [
    (lambda d:d.drop(columns='Target1'),'columns'),
    (lambda d:pd.concat([d,d]),'Duplicate'),
    (lambda d:d.assign(Context1=''),'Empty'),
    (lambda d:d.assign(Context1=d.Context2),'contrasting'),
    (lambda d:d.iloc[:0],'no items'),
])
def test_native_validation(native, mutation, match):
    with pytest.raises(ValueError, match=match):
        normalize(mutation(pd.read_csv(native)))


def test_native_column_case_and_version_from_directory(native, tmp_path):
    frame=pd.read_csv(native).drop(columns='Version')
    frame.columns=[c.lower() for c in frame.columns]
    path=tmp_path/'vers=4';path.mkdir();frame.to_csv(path/'items.csv',index=False)
    result=prepare_dataset('EWoK-core-1.0',source=path,output=tmp_path/'out')
    assert load_prepared(result)[1]['versions']==['4']


def test_parquet_native_round_trip(native, tmp_path):
    pytest.importorskip('pyarrow', reason='Native Parquet requires brainscore[ewok-data]')
    path=tmp_path/'native.parquet';pd.read_csv(native,dtype=str).to_parquet(path,index=False)
    result=prepare_dataset('EWoK-core-1.0',source=path,output=tmp_path/'out')
    assert len(load_prepared(result)[0])==3


def test_request_resolver_is_plugin_specific_and_not_persisted(native, tmp_path, monkeypatch):
    seen=[]
    def resolver(request_id):
        seen.append(request_id)
        return native, {'method':'author-download'}
    from brainscore.data.ewok.data_packaging.prepare import build
    monkeypatch.setitem(data_builder_registry,'fixture',lambda:DataBuilder(build, resolver))
    output=prepare_dataset('fixture',request_id='private-author-request',output=tmp_path/'resolved')
    assert seen==['private-author-request']
    assert 'private-author-request' not in (output/'build.json').read_text()
    assert load_prepared(output)[1]['items']==3


def test_huggingface_uses_pinned_native_file(native, monkeypatch):
    import huggingface_hub
    calls=[]
    def download(**kwargs):
        calls.append(kwargs)
        return str(native.parent)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',download)
    path,meta=resolve(HF_REPOSITORY)
    assert path==native.parent/'data/test/ewok-core-1.0.parquet'
    assert calls[0]['revision']==HF_REVISION
    assert calls[0]['repo_type']=='dataset'
    assert 'token' not in calls[0]
    assert meta['revision']==HF_REVISION


def test_access_failures_do_not_reveal_credentials(monkeypatch):
    import huggingface_hub
    def fail(**kwargs):raise RuntimeError('secret-token-in-url')
    monkeypatch.setattr(huggingface_hub,'snapshot_download',fail)
    with pytest.raises(PermissionError) as error:resolve(HF_REPOSITORY)
    assert 'secret-token' not in str(error.value)
    with pytest.raises(ValueError, match='approval codes'):resolve('secret-request')


def test_cli_build(native, tmp_path):
    result=subprocess.run([sys.executable,'-m','brainscore.data','prepare','EWoK-core-1.0',
        '--source',str(native),'--output',str(tmp_path/'cli')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert len(load_prepared(tmp_path/'cli')[0])==3


@pytest.mark.parametrize('values,expected',[
    ([-1,-2,-2,-1],1),([-2,-1,-1,-2],0),([-1,-1,-1,-1],.5),([-1,-1,-2,-1],.75),
])
def test_logprob_metric(values, expected):
    assert logprob_accuracy(values)==expected


@pytest.mark.parametrize('values',[[0,1,2], [0,1,2,float('nan')],[0,1,2,float('inf')],[True,1,2,3]])
def test_invalid_logprobs_fail(values):
    with pytest.raises(ValueError):logprob_accuracy(values)


class NativeModel:
    def __init__(self):self.calls=[]
    def score(self,targets,contexts):
        self.calls.append((targets,contexts))
        return [-1.0 if (' A ' in t)==(' A ' in c) else -2.0 for t,c in zip(targets,contexts)]
    def complete_choice(self,targets,contexts1,contexts2,gen_type,prompt_type):
        assert (gen_type,prompt_type)==('constrained','optimized')
        return ['1' if ' A ' in t else '2' for t in targets]


@pytest.mark.parametrize('mode',['logprobs','choice'])
def test_registered_benchmark_through_score_and_tools(mode, prepared, tmp_path, monkeypatch):
    monkeypatch.setenv('RESULTCACHING_HOME',str(tmp_path/'cache'))
    native=NativeModel();provider=EWoKProvider(native);requests=[]
    def record(request):
        requests.append(request)
        assert not any(k in request for k in ('expected','labels','target_number'))
        return provider(request)
    benchmark=brainscore.load_benchmark(f'EWoK-core-1.0-{mode}',root=prepared,
        batch_size=2,output_dir=tmp_path/'scored')
    value=brainscore.score(subject(record),benchmark,check_mem=False)
    assert float(value)==1 and value.attrs['normalized'] is False
    assert value.attrs['items']==3 and value.attrs['questions']==6
    assert value.attrs['paper_replication'] is False
    run=Experiment(subject=subject(provider),protocol=benchmark.protocol(),
        tools=[RecordInputsOutputs()],output_dir=tmp_path/'tools').run()
    assert float(run.value)==float(value)
    assert sum(e['kind']=='input' for e in run.record.events())==len(requests)
    assert (run.directory/'item-scores.json').exists()


def test_equal_weight_versions_not_pooled_rows(prepared, tmp_path):
    def provider(request):
        values=['2' if ' A ' in t else '1' for t in request['targets']]
        # Only version 1 (item 2) is correct: 1/2 over versions, not 1/3 over rows.
        for i,t in enumerate(request['targets']):
            if t.endswith('2'):values[i]='1' if ' A ' in t else '2'
        return {'text':json.dumps(values)}
    b=brainscore.load_benchmark('EWoK-core-1.0-choice',root=prepared,output_dir=tmp_path/'run')
    assert float(b(subject(provider)))==.5


def test_invalid_choice_counts_as_wrong_not_dropped(prepared, tmp_path):
    def provider(request):return {'text':json.dumps(['Reasoning says 1 but answer is 2']*len(request['targets']))}
    b=brainscore.load_benchmark('EWoK-core-1.0-choice',root=prepared,output_dir=tmp_path/'run')
    value=b(subject(provider));assert float(value)==0;assert value.attrs['invalid_answers']==6


def test_failure_preserves_failed_run(prepared, tmp_path):
    b=brainscore.load_benchmark('EWoK-core-1.0-logprobs',root=prepared,output_dir=tmp_path/'run')
    with pytest.raises(ValueError,match='one value'):
        b(subject(lambda request:{'text':'[]'}))
    assert json.loads((tmp_path/'run/experiment.json').read_text())['status']=='failed'


def test_missing_data_and_bad_domains_fail_early(tmp_path, prepared):
    with pytest.raises(FileNotFoundError,match='prepare'):
        brainscore.load_benchmark('EWoK-core-1.0-logprobs',root=tmp_path/'missing')
    with pytest.raises(ValueError,match='domains'):
        brainscore.load_benchmark('EWoK-core-1.0-choice',root=prepared,domains=['missing'])


def test_existing_local_asset_cli_is_preserved(capsys):
    from brainscore.data.__main__ import main
    assert main([])==0
    assert 'lana-atlas' in capsys.readouterr().out
    assert main(['lana-atlas'])==0
    assert 'Fedorenko' in capsys.readouterr().out


def test_archive_build_applies_final_exclusions(native, tmp_path, monkeypatch):
    pyzipper = pytest.importorskip('pyzipper', reason='Protected archives require brainscore[ewok-data]')
    rows=pd.read_csv(native,dtype=str)
    archive_root=tmp_path/'paper';(archive_root/'analyses').mkdir(parents=True)
    for filename,member,frame in [
        ('analyses/data.zip','data/items_in_results.csv',rows),
        ('config.zip','config/utils/remove_from_results.csv',rows.iloc[:1]),
    ]:
        with pyzipper.AESZipFile(archive_root/filename,'w',encryption=pyzipper.WZ_AES) as z:
            z.setpassword(b'fixture-password');z.writestr(member,frame.to_csv(index=False))
            z.writestr('../never-extract.txt','fixture')
    monkeypatch.setenv('EWOK_ARCHIVE_PASSWORD','fixture-password')
    result=prepare_dataset('EWoK-core-1.0',source=archive_root,output=tmp_path/'out')
    data,manifest=load_prepared(result)
    assert len(data)==2 and manifest['final_exclusions']==1
    assert not (tmp_path/'never-extract.txt').exists()
    assert 'fixture-password' not in (result/'manifest.json').read_text()


def test_missing_archive_password_fails_without_partial_data(tmp_path, monkeypatch):
    # Password validation must work even without the optional archive reader.
    monkeypatch.setitem(sys.modules, 'pyzipper', None)
    source=tmp_path/'paper';(source/'analyses').mkdir(parents=True)
    (source/'analyses/data.zip').write_bytes(b'not-read')
    monkeypatch.delenv('EWOK_ARCHIVE_PASSWORD',raising=False)
    with pytest.raises(ValueError,match='EWOK_ARCHIVE_PASSWORD'):
        prepare_dataset('EWoK-core-1.0',source=source,output=tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_native_text_whitespace_preserved(native):
    frame=pd.read_csv(native,dtype=str);frame.loc[0,'Context1']='  fixture text  '
    assert next(r for r in normalize(frame) if r['TemplateID']=='0')['Context1']=='  fixture text  '


def test_data_source_choice_required(tmp_path):
    for options in ({},{'source':tmp_path,'request_id':'id'}):
        with pytest.raises(ValueError,match='exactly one'):
            prepare_dataset('EWoK-core-1.0',output=tmp_path/'out',**options)


def test_feature_only_subject_rejected(prepared):
    from brainscore_core.compatibility import CompatibilityError
    benchmark=brainscore.load_benchmark('EWoK-core-1.0-choice',root=prepared)
    with pytest.raises(CompatibilityError):benchmark(brainscore.load_model('chance-baseline'))


def test_real_hooks_change_score_and_restore(prepared, tmp_path):
    import torch
    from brainscore.experiments import Ablate, RecordActivity, TorchInstrumentation
    class Native:
        def __init__(self):
            self.model=torch.nn.Sequential()
            self.model.add_module('projection',torch.nn.Linear(1,1,bias=False))
            with torch.no_grad():self.model.projection.weight.fill_(2)
        def score(self,targets,contexts):
            inputs=torch.tensor([[1.] if (' A ' in t)==(' A ' in c) else [-1.]
                                 for t,c in zip(targets,contexts)])
            with torch.no_grad():
                return torch.nn.functional.logsigmoid(self.model(inputs)).flatten().tolist()
    native=Native();candidate=subject(EWoKProvider(native))
    b=brainscore.load_benchmark('EWoK-core-1.0-logprobs',root=prepared)
    outcomes=[]
    for name,interventions in [('baseline',[]),('ablated',[Ablate(['projection'])]),('restored',[])]:
        result=Experiment(subject=candidate,protocol=b.protocol(),
            tools=[RecordInputsOutputs(),*interventions,RecordActivity(['projection'])],
            instrumentation=TorchInstrumentation(native.model),output_dir=tmp_path/name).run()
        outcomes.append(float(result.value))
        assert any(e['kind']=='activity' for e in result.record.events())
        assert not native.model.projection._forward_hooks
    assert outcomes==[1.,.5,1.]
    assert native.model.projection.weight.item()==2.


class _ParticipateOnlyReader(Subject):
    """Answers EWoK requests through participate only: no process, no private state."""
    identifier = 'participate-only-ewok'
    in_channels = {'conditional_logprob_request', 'context_choice_request'}
    out_channels = {'response_trace'}

    def __init__(self, provider):
        self.provider, self.sessions = provider, 0

    def participate(self, session):
        from brainscore_core.streaming import StreamEvent
        self.sessions += 1
        assert tuple(session.requested_output_channels) == ('response_trace',)
        while (event := session.next_input()) is not None:
            answer = json.loads(self.provider(event.payload)['text'])
            session.respond(StreamEvent('response_trace', {
                'raw': {}, 'answer': answer, 'valid': True, 'parse_error': None,
                'provenance': {}}, event.t_ms, dict(event.meta)))


@pytest.mark.parametrize('mode', ['logprobs', 'choice'])
def test_participate_only_subject_scores_like_the_trace_subject(mode, prepared, tmp_path):
    provider = EWoKProvider(NativeModel())
    reader = _ParticipateOnlyReader(provider)
    assert not hasattr(reader, 'process')
    benchmark = brainscore.load_benchmark(f'EWoK-core-1.0-{mode}', root=prepared,
                                          batch_size=2, output_dir=tmp_path / 'stub')
    value = benchmark(reader)
    reference = brainscore.load_benchmark(f'EWoK-core-1.0-{mode}', root=prepared, batch_size=2,
                                          output_dir=tmp_path / 'trace')(subject(provider))
    assert float(value) == float(reference) == 1
    # One session per request: two batches, four (logprobs) or two (choice) requests each.
    assert reader.sessions == 2 * (4 if mode == 'logprobs' else 2)
    record = json.loads((tmp_path / 'stub' / 'experiment.json').read_text())
    assert record['status'] != 'failed'


def _old_process_path(subject_, context, request, repetition):
    """What the benchmark did before it ran through participate."""
    return subject_.process(request)


def _streaming_provider(provider):
    """Yield the provider's answer as two fragments, the second one final."""
    def stream(request):
        text = provider(request)['text']
        yield {'text': '', 'final': False}
        yield {'text': text, 'final': True}
    return stream


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('mode', ['logprobs', 'choice'])
def test_same_subject_old_process_path_and_participate_give_identical_item_scores(
        mode, streaming, prepared, tmp_path, monkeypatch):
    import brainscore.benchmarks.ewok.benchmark as ewok
    provider = EWoKProvider(NativeModel())
    candidate = build_trace_subject(
        'fixture', provider=_streaming_provider(provider) if streaming else provider,
        parse=json.loads, provenance={'model': 'synthetic-test'}, streaming=streaming,
        input_channels=['conditional_logprob_request', 'context_choice_request'])

    def run(name):
        benchmark = brainscore.load_benchmark(f'EWoK-core-1.0-{mode}', root=prepared,
                                              batch_size=2, output_dir=tmp_path / name)
        return benchmark(candidate), (tmp_path / name / 'item-scores.json').read_bytes()

    new_score, new_items = run('participate')
    with monkeypatch.context() as patch:
        patch.setattr(ewok, '_ask', _old_process_path)
        old_score, old_items = run('process')
    assert new_items == old_items
    assert repr(float(new_score)) == repr(float(old_score))
    for key in ('version_scores', 'domain_scores', 'items', 'invalid_answers'):
        assert new_score.attrs[key] == old_score.attrs[key]


def test_replay_rebuilds_one_session_per_request(prepared, tmp_path):
    from brainscore.experiments import compare_outputs, replay_sessions
    from brainscore.run_record import RunRecord
    provider = EWoKProvider(NativeModel())
    candidate = subject(provider)
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=prepared, batch_size=2)
    original = Experiment(subject=candidate, protocol=benchmark.protocol(),
                          tools=[RecordInputsOutputs()], output_dir=tmp_path / 'original').run()
    events = list(RunRecord(original.directory).events())
    starts = [e for e in events if e['kind'] == 'repetition_start']
    # Three items in batches of two, four requests per batch: eight sessions.
    assert len(starts) == 8 == sum(e['kind'] == 'input' for e in events)
    by_repetition = {}
    for event in events:
        if event['kind'] in ('input', 'output'):
            by_repetition.setdefault(event['repetition_id'], []).append(event['kind'])
    assert list(by_repetition.values()) == [['input', 'output']] * 8

    calls = []
    counting = subject(lambda request: calls.append(request) or provider(request))
    replay = Experiment(subject=counting, protocol=replay_sessions(original.directory),
                        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'replay').run()
    assert len(calls) == 8
    replayed = list(RunRecord(replay.directory).events())
    assert sum(e['kind'] == 'repetition_start' for e in replayed) == 8
    assert compare_outputs(original.directory, replay.directory)['equal']


class _UnfinishedReader(_ParticipateOnlyReader):
    """Sends only a non-final fragment: valid text, but the stream never ends."""
    def participate(self, session):
        from brainscore_core.streaming import StreamEvent
        while (event := session.next_input()) is not None:
            answer = json.loads(self.provider(event.payload)['text'])
            session.respond(StreamEvent('response_trace', {
                'raw': {}, 'answer': answer, 'valid': True, 'parse_error': None,
                'provenance': {}, 'final': False}, event.t_ms, dict(event.meta)))


def test_unfinished_stream_is_refused_not_scored(prepared, tmp_path):
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=prepared,
                                          output_dir=tmp_path / 'run')
    with pytest.raises(ValueError, match='final response'):
        benchmark(_UnfinishedReader(EWoKProvider(NativeModel())))
