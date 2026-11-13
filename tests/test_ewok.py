"""Local data preparation, protocol fidelity and ordinary UMI tool integration."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import brainscore
from brainscore.data.preparation import DataBuilder, data_builder_registry, prepare_dataset
from brainscore.data.ewok.data import load_prepared
from brainscore.data.ewok.data_packaging.prepare import HF_REPOSITORY, HF_REVISION, normalize, resolve
from brainscore.benchmarks.ewok.benchmark import logprob_accuracy
from brainscore.harnesses.ewok import EWoKProvider
from brainscore.experiments import Experiment, RecordInputsOutputs
from brainscore import BrainScoreModel
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


def likelihood_subject(likelihood):
    """A normal model with a ``likelihood`` part: it runs both modes."""
    return BrainScoreModel('fixture', in_channels={'text'}, likelihood=likelihood)


def scenario(prompt):
    """The scenario of the test example in EWoK's choice prompt."""
    return prompt.text().split('# TEST EXAMPLE')[1].split('## Scenario\n"')[1].split('"\n')[0]


def native_choice(prompt):
    """Answer as a perfect EWoK model: context 1 for target A, context 2 for target B."""
    return '1' if ' A ' in scenario(prompt) else '2'


def chooser(reply=native_choice):
    """A normal model with a ``reply`` part: the common channels, no EWoK channel."""
    return BrainScoreModel('fixture', in_channels={'text'}, reply=reply)


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


@pytest.mark.parametrize('mode',['logprobs','choice'])
def test_registered_benchmark_through_score_and_tools(mode, prepared, tmp_path, monkeypatch):
    monkeypatch.setenv('RESULTCACHING_HOME',str(tmp_path/'cache'))
    native=NativeModel();provider=EWoKProvider(native);requests=[]
    def record(prompt, candidates):
        requests.append(prompt)
        # Raw text: the context exactly as given, the two targets as options.
        assert prompt.names == ('text',) and len(candidates) == 2
        return provider(prompt, candidates)
    def reply(prompt):
        requests.append(prompt)
        # Raw text: EWoK's prompt with no sentence added.
        assert prompt.names == ('text',)
        return native_choice(prompt)
    make = (lambda: likelihood_subject(record)) if mode == 'logprobs' else (lambda: chooser(reply))
    benchmark=brainscore.load_benchmark(f'EWoK-core-1.0-{mode}',root=prepared,
        batch_size=2,output_dir=tmp_path/'scored')
    value=brainscore.score(make(),benchmark,check_mem=False)
    assert float(value)==1 and value.attrs['normalized'] is False
    assert value.attrs['items']==3 and value.attrs['questions']==6
    assert value.attrs['paper_replication'] is False
    sent=len(requests)
    run=Experiment(subject=make(),protocol=benchmark.protocol(),
        tools=[RecordInputsOutputs()],output_dir=tmp_path/'tools').run()
    assert float(run.value)==float(value)
    inputs=[e for e in run.record.events() if e['kind']=='input' and e['payload'].channel!='control']
    # One text row per question: a context (log probabilities) or a prompt (choice).
    assert len(inputs)==len(requests)-sent==sent==6
    assert (run.directory/'item-scores.json').exists()
    if mode=='choice':
        assert value.attrs['input_adaptation']['kind']=='subject_prompt'
        assert all(e['payload'].channel=='text' for e in inputs)
    else:
        assert value.attrs['input_adaptation']['kind']=='subject_scoring'
        assert 'add_special_tokens=True' in value.attrs['input_adaptation']['reason']


def test_equal_weight_versions_not_pooled_rows(prepared, tmp_path):
    def reply(prompt):
        target=scenario(prompt)
        # Only version 1 (item 2) is correct: 1/2 over versions, not 1/3 over rows.
        if target.endswith('2'):return native_choice(prompt)
        return '2' if ' A ' in target else '1'
    b=brainscore.load_benchmark('EWoK-core-1.0-choice',root=prepared,output_dir=tmp_path/'run')
    assert float(b(chooser(reply)))==.5


def test_invalid_choice_counts_as_wrong_not_dropped(prepared, tmp_path):
    def reply(prompt):return 'Reasoning says 1 but answer is 2'
    b=brainscore.load_benchmark('EWoK-core-1.0-choice',root=prepared,output_dir=tmp_path/'run')
    value=b(chooser(reply));assert float(value)==0;assert value.attrs['invalid_answers']==6


def test_choice_prompt_is_ewoks_own_text_with_nothing_added(prepared, tmp_path):
    """The subject sees EWoK's prompt as the trial's text, with no directive after it."""
    from brainscore.benchmarks.ewok.benchmark import _choice_prompt
    prompts=[]
    b=brainscore.load_benchmark('EWoK-core-1.0-choice',root=prepared,batch_size=8)
    b(chooser(lambda prompt: prompts.append(prompt) or native_choice(prompt)))
    rows=b._rows
    assert [p.parts for p in prompts]==[
        (('text',_choice_prompt(r[f'Target{t}'],r['Context1'],r['Context2'])),)
        for t in (1,2) for r in rows]
    first=prompts[0].parts[0][1]
    assert first.startswith('# INSTRUCTIONS\n\nIn this study')
    assert first.endswith('"\n\n## Task\nWhich context makes more sense given the scenario? '
                          'Please answer using either "1" or "2".\n\n## Response\n')
    assert '1. "fixture context A 0"\n2. "fixture context B 0"' in first
    assert all(p.response.labels==('1','2') and p.response.raw_text for p in prompts)


def test_a_likelihood_part_also_runs_choice(prepared, tmp_path, parts={'likelihood'}):
    """A likelihood part gives behavior:choice, not only a reply. An answer part cannot: raw text."""
    def likelihood(prompt, candidates):
        return {label: (0. if label==native_choice(prompt) else -1.) for label in candidates}
    def answer(stimulus_row, instruction, label_set):
        return '1' if ' A ' in stimulus_row['text'].split('# TEST EXAMPLE')[1].split('## Scenario')[1] else '2'
    model=BrainScoreModel('fixture', in_channels={'text'},
                          **({'likelihood':likelihood} if 'likelihood' in parts else {'answer':answer}))
    b=brainscore.load_benchmark('EWoK-core-1.0-choice',root=prepared,output_dir=tmp_path/'run')
    assert float(b(model))==1


def test_an_answer_part_cannot_run_the_raw_text_choice(prepared, tmp_path):
    model=BrainScoreModel('fixture', in_channels={'text'}, answer=lambda row, instruction, labels: '1')
    b=brainscore.load_benchmark('EWoK-core-1.0-choice',root=prepared,output_dir=tmp_path/'run')
    with pytest.raises(ValueError,match='cannot give a Choice answer'):
        b(model)


def test_failure_preserves_failed_run(prepared, tmp_path):
    b=brainscore.load_benchmark('EWoK-core-1.0-logprobs',root=prepared,output_dir=tmp_path/'run')
    with pytest.raises(ValueError,match='finite'):
        b(likelihood_subject(lambda prompt, candidates:{c: float('nan') for c in candidates}))
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
    with pytest.raises(CompatibilityError):brainscore.score(brainscore.load_model('chance-baseline'),benchmark,check_mem=False)


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
    native=Native();subject=likelihood_subject(EWoKProvider(native))
    b=brainscore.load_benchmark('EWoK-core-1.0-logprobs',root=prepared)
    outcomes=[]
    for name,interventions in [('baseline',[]),('ablated',[Ablate(['projection'])]),('restored',[])]:
        result=Experiment(subject=subject,protocol=b.protocol(),
            tools=[RecordInputsOutputs(),*interventions,RecordActivity(['projection'])],
            instrumentation=TorchInstrumentation(native.model),output_dir=tmp_path/name).run()
        outcomes.append(float(result.value))
        assert any(e['kind']=='activity' for e in result.record.events())
        assert not native.model.projection._forward_hooks
    assert outcomes==[1.,.5,1.]
    assert native.model.projection.weight.item()==2.


class _ParticipateOnlyReader(Subject):
    """Answers EWoK contexts with raw option scores through participate only: no process."""
    identifier = 'participate-only-ewok'
    in_channels = {'text'}
    out_channels = {'behavior:probabilities'}
    scale = 'log'

    def __init__(self, provider):
        self.provider, self.sessions = provider, 0

    def participate(self, session):
        from brainscore_core.streaming import StreamEvent
        from brainscore_core.responses import Prompt
        from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
        self.sessions += 1
        assert tuple(session.outputs) == ('behavior:probabilities',)
        while (event := session.next_input()) is not None:
            options = list(session.stimuli.loc[event.meta['stimulus_index'], 'options'])
            scores = self.provider(Prompt((('text', event.payload),)), options)
            session.respond(StreamEvent('behavior:probabilities', BehavioralAssembly(
                [[scores[option] for option in options]],
                coords={'stimulus_id': ('presentation', [event.meta['stimulus_id']]),
                        'choice': [0, 1]}, dims=['presentation', 'choice'],
                attrs={'measure': {'kind': 'implied', 'source': 'likelihood', 'scale': self.scale}}),
                event.t_ms, dict(event.meta)))

    def reset(self):
        pass


class _ParticipateOnlyChooser(Subject):
    """Answers EWoK choice prompts on behavior:choice through participate only."""
    identifier = 'participate-only-ewok-choice'
    in_channels = {'text'}
    out_channels = {'behavior:choice'}

    def __init__(self):
        self.sessions = 0

    def participate(self, session):
        from brainscore_core.streaming import StreamEvent
        from brainscore_core.responses import Prompt
        from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
        self.sessions += 1
        assert tuple(session.outputs) == ('behavior:choice',)
        labels = list(session.task.response.labels)
        while (event := session.next_input()) is not None:
            chosen = native_choice(Prompt((('text', event.payload),)))
            values = [[1.0 if label == chosen else 0.0 for label in labels]]
            session.respond(StreamEvent('behavior:choice', BehavioralAssembly(
                values, coords={'stimulus_id': ('presentation', [event.meta['stimulus_id']]),
                                'choice': labels}, dims=['presentation', 'choice']),
                event.t_ms, dict(event.meta)))

    def reset(self):
        pass


@pytest.mark.parametrize('mode', ['logprobs', 'choice'])
def test_participate_only_subject_scores_like_a_brainscore_model(mode, prepared, tmp_path):
    """A plain contract subject scores like a BrainScoreModel with the matching part."""
    provider = EWoKProvider(NativeModel())
    reader = _ParticipateOnlyReader(provider) if mode == 'logprobs' else _ParticipateOnlyChooser()
    assert not hasattr(reader, 'process')
    benchmark = brainscore.load_benchmark(f'EWoK-core-1.0-{mode}', root=prepared,
                                          batch_size=2, output_dir=tmp_path / 'stub')
    value = benchmark(reader)
    reference = brainscore.load_benchmark(f'EWoK-core-1.0-{mode}', root=prepared, batch_size=2,
                                          output_dir=tmp_path / 'trace')(
        likelihood_subject(provider) if mode == 'logprobs' else chooser())
    assert float(value) == float(reference) == 1
    # One session per context (logprobs) or target (choice): two batches, two each.
    assert reader.sessions == 2 * 2
    record = json.loads((tmp_path / 'stub' / 'experiment.json').read_text())
    assert record['status'] != 'failed'


WORDS = ['[PAD]', '[UNK]', 'the', 'cat', 'dog', 'sat', 'ran', 'is', 'kind', 'mean', 'a']


@pytest.fixture
def tiny_lm(monkeypatch):
    """A tiny causal LM on a whole-word vocabulary: plain text, no chat template."""
    torch = pytest.importorskip('torch')
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    raw = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(WORDS)}, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]',
                                  eos_token='[PAD]', model_max_length=64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        net = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(WORDS), hidden_size=16, intermediate_size=24,
                                           num_hidden_layers=2, num_attention_heads=2,
                                           num_key_value_heads=1, max_position_embeddings=64)).eval()
    monkeypatch.setattr('brainscore.model_helpers._device.select_device', lambda device=None: 'cpu')
    return net, tok


@pytest.fixture
def word_items(tmp_path):
    """Synthetic EWoK items made of in-vocabulary words, so the scores differ."""
    words = ['the cat sat', 'the dog ran', 'a cat is kind', 'a dog is mean', 'the cat ran',
             'the dog sat', 'a cat is mean', 'a dog is kind', 'the cat is', 'the dog is']
    rows = [dict(Domain='synthetic-domain', MetaTemplateID='1', TemplateID=str(i), Version=str(i % 2),
                 Context1=words[i], Context2=words[(i + 1) % 10], Target1=words[(i + 3) % 10],
                 Target2=words[(i + 5) % 10]) for i in range(10)]
    path = tmp_path / 'words.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return prepare_dataset('EWoK-core-1.0', source=path, output=tmp_path / 'word-build')


def _trace_path_logprobs(net, tok, contexts, targets, batch_size=8):
    """log p(target | context) as the removed request channel's provider computed it.

    A copy of ``CausalLMProvider.logprobs`` (sum reduction, raw contexts):
    context and target tokenized separately and joined, right-padded batches of
    ``batch_size`` items, as the EWoK benchmark sent them (eight items per request).
    """
    import torch
    encoded = [(tok(c, add_special_tokens=False)['input_ids'], tok(t, add_special_tokens=False)['input_ids'])
               for c, t in zip(contexts, targets)]
    values = []
    for start in range(0, len(encoded), batch_size):
        pairs = encoded[start:start + batch_size]
        width = max(len(c) + len(t) for c, t in pairs)
        pad = tok.pad_token_id
        ids = torch.tensor([c + t + [pad] * (width - len(c) - len(t)) for c, t in pairs])
        mask = torch.tensor([[1] * (len(c) + len(t)) + [0] * (width - len(c) - len(t)) for c, t in pairs])
        scored = torch.tensor([[0] * len(c) + [1] * len(t) + [0] * (width - len(c) - len(t))
                               for c, t in pairs])[:, 1:].float()
        with torch.no_grad():
            logp = net(input_ids=ids, attention_mask=mask).logits[:, :-1].float().log_softmax(-1)
        values += (logp.gather(-1, ids[:, 1:, None])[..., 0] * scored).sum(1).tolist()
    return values


@pytest.mark.parametrize('parts', ['chat_parts', 'build_causal_lm_subject'])
def test_logprobs_match_the_removed_request_path_on_the_same_model(tiny_lm, word_items, tmp_path, parts):
    """EWoK log-probability numbers do not move for a subject that scores plain text.

    The old path sent 8 items per request and the provider scored them in one
    padded batch; the new path scores the two targets of one context per
    trial. Right padding keeps every real token's prediction the same, but the
    batches differ in shape, so the comparison allows float32 rounding (about
    5e-7 here). Both helpers that give a likelihood part are checked.
    """
    from brainscore.benchmarks.ewok.benchmark import _TASKS
    from brainscore.model_helpers.replies import chat_parts
    from brainscore_core.contract import TaskContext
    from brainscore_core.streaming_helpers import stimulus_session
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    net, tok = tiny_lm
    if parts == 'chat_parts':
        model = likelihood_subject(chat_parts(net, tok)[1])
    else:
        from brainscore.model_helpers.causal_lm import build_causal_lm_subject
        model = build_causal_lm_subject('tiny-lm', net, tok)
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=word_items, batch_size=8,
                                          output_dir=tmp_path / 'new')
    rows = benchmark._rows
    # New path: one session over every context, with both targets as its options.
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': [f'{r["item_id"]}.context{n}' for n in (1, 2) for r in rows],
        'text': [r[f'Context{n}'] for n in (1, 2) for r in rows],
        'options': [(r['Target1'], r['Target2']) for n in (1, 2) for r in rows]}))
    session = stimulus_session(stimuli, ['behavior:probabilities'],
                               TaskContext(response=_TASKS['logprobs'][0]))
    model.participate(session)
    new = session.collect('behavior:probabilities')
    assert new.attrs['measure']['scale'] == 'log'
    # Old path: the requests the old benchmark sent, (target, context) pairs, eight items each.
    old = {}
    for target in (1, 2):
        for number in (1, 2):
            for start in range(0, len(rows), 8):
                batch = rows[start:start + 8]
                values = _trace_path_logprobs(net, tok, [r[f'Context{number}'] for r in batch],
                                              [r[f'Target{target}'] for r in batch])
                for r, value in zip(batch, values):
                    old[r['item_id'], number, target] = value
    expected = np.array([[old[r['item_id'], n, 1], old[r['item_id'], n, 2]] for n in (1, 2) for r in rows])
    np.testing.assert_allclose(new.values, expected, rtol=0, atol=1e-5)
    assert len(set(np.round(expected.ravel(), 4))) > 4   # the items are told apart
    # And the item scores, the accuracy the old benchmark computed from those values.
    score = benchmark(model)
    items = json.loads((tmp_path / 'new' / 'item-scores.json').read_text())
    for item, r in zip(items, rows):
        assert item['accuracy'] == logprob_accuracy([old[r['item_id'], 1, 1], old[r['item_id'], 2, 1],
                                                     old[r['item_id'], 1, 2], old[r['item_id'], 2, 2]])
    assert 0 < float(score) < 1


def test_an_empty_context_is_refused_as_ewok_cannot_score_it(tiny_lm):
    """EWoK starts at token 0 of an empty context; without a BOS token that token
    has no prediction (EWoK returns NaN), so plain-text scoring refuses it."""
    from brainscore.model_helpers.replies import chat_parts
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import trial_prompt
    net, tok = tiny_lm
    _, likelihood = chat_parts(net, tok)
    from brainscore.benchmarks.ewok.benchmark import _TASKS
    task = TaskContext(response=_TASKS['logprobs'][0])
    prompt = trial_prompt(task, {'stimulus_id': 'a', 'text': '', 'options': ('the cat', 'the dog')})
    with pytest.raises(ValueError, match='no tokens to condition on'):
        likelihood(prompt, ['the cat', 'the dog'])


def test_a_subject_without_a_likelihood_part_is_na(prepared):
    """Raw scores come only from a likelihood part; no softmax and no guess."""
    from brainscore_core.compatibility import CompatibilityError
    from brainscore_core.score_status import score_or_na
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=prepared)
    for subject in (chooser(), BrainScoreModel('answers', in_channels={'text'},
                                                 answer=lambda stimulus_row, instruction, label_set: '1')):
        # score() runs the one pre-flight; the benchmark checks nothing itself.
        with pytest.raises(CompatibilityError):
            brainscore.score(subject, benchmark, check_mem=False)
        assert score_or_na(brainscore.score, subject, benchmark, check_mem=False).attrs['status'] == 'N/A'


def test_replay_rebuilds_one_session_per_request(prepared, tmp_path):
    from brainscore.experiments import compare_outputs, replay_sessions
    from brainscore.run_record import RunRecord
    provider = EWoKProvider(NativeModel())
    subject = likelihood_subject(provider)
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=prepared, batch_size=2)
    original = Experiment(subject=subject, protocol=benchmark.protocol(),
                          tools=[RecordInputsOutputs()], output_dir=tmp_path / 'original').run()
    events = list(RunRecord(original.directory).events())
    starts = [e for e in events if e['kind'] == 'repetition_start']
    # Three items in batches of two, one session per context: four sessions, six contexts.
    assert len(starts) == 4
    assert sum(e['kind'] == 'input' and e['payload'].channel == 'text' for e in events) == 6
    outputs = {}
    for event in events:
        if event['kind'] == 'output':
            outputs.setdefault(event['repetition_id'], []).append(event['payload'].channel)
    assert list(outputs.values()) == [['behavior:probabilities']] * 4

    calls = []
    counting = likelihood_subject(lambda prompt, candidates: calls.append(prompt) or provider(prompt, candidates))
    replay = Experiment(subject=counting, protocol=replay_sessions(original.directory),
                        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'replay').run()
    assert len(calls) == 6
    replayed = list(RunRecord(replay.directory).events())
    assert sum(e['kind'] == 'repetition_start' for e in replayed) == 4
    assert compare_outputs(original.directory, replay.directory)['equal']


class _NormalizedReader(_ParticipateOnlyReader):
    """Sends normalized probabilities where raw log scores were asked for."""
    scale = None


def test_normalized_probabilities_are_refused_not_scored(prepared, tmp_path):
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=prepared,
                                          output_dir=tmp_path / 'run')
    with pytest.raises(ValueError, match='raw log scores'):
        benchmark(_NormalizedReader(EWoKProvider(NativeModel())))


class _TorchNative:
    """EWoK native model whose answers pass through one linear layer."""
    def __init__(self):
        import torch
        self.model = torch.nn.Sequential()
        self.model.add_module('projection', torch.nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            self.model.projection.weight.fill_(2)

    def score(self, targets, contexts):
        import torch
        inputs = torch.tensor([[1.] if (' A ' in t) == (' A ' in c) else [-1.]
                               for t, c in zip(targets, contexts)])
        with torch.no_grad():
            return torch.nn.functional.logsigmoid(self.model(inputs)).flatten().tolist()


def test_tools_select_the_repetition_ids_each_request_runs_in(prepared, tmp_path):
    from brainscore.experiments import Ablate, TorchInstrumentation
    from brainscore.run_record import RunRecord
    native = _TorchNative()
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=prepared, batch_size=2)
    assert benchmark.protocol().repetitions == tuple(range(4))
    result = Experiment(subject=likelihood_subject(EWoKProvider(native)), protocol=benchmark.protocol(),
                        tools=[RecordInputsOutputs(), Ablate(['projection'], repetitions=[0])],
                        instrumentation=TorchInstrumentation(native.model),
                        output_dir=tmp_path / 'selected').run()
    starts = [e['repetition_id'] for e in RunRecord(result.directory).events()
              if e['kind'] == 'intervention_start']
    assert starts == [0]
    with pytest.raises(ValueError, match='repetitions must belong'):
        Experiment(subject=likelihood_subject(EWoKProvider(native)), protocol=benchmark.protocol(),
                   tools=[RecordInputsOutputs(), Ablate(['projection'], repetitions=['external'])],
                   instrumentation=TorchInstrumentation(native.model),
                   output_dir=tmp_path / 'external').run()


def _stateful_intervened_subject():
    """A subject whose scores depend on its own history and on a lesion.

    The likelihood part counts its calls and flips its scores on odd ones, so
    a reset or a repeated call would show. The intervention flips them too,
    and its cleanup is the only way to undo it.
    """
    from brainscore_core.events import PerturbationApplied
    base, state = EWoKProvider(NativeModel()), {'requests': 0, 'lesioned': False}

    def likelihood(prompt, candidates):
        scores = base(prompt, candidates)
        flip = (state['requests'] % 2 == 1) != state['lesioned']
        state['requests'] += 1
        return {c: -v for c, v in scores.items()} if flip else scores

    def intervention(change):
        state['lesioned'] = True

        def cleanup():
            state['lesioned'] = False
        return PerturbationApplied(handle_id='lesion', target=change.target,
                                   perturbation=change.perturbation, applied_at=0), cleanup

    model = BrainScoreModel('stateful-fixture', in_channels={'text'},
                            likelihood=likelihood, intervention=intervention)
    return model, state


def test_stateful_subject_inside_a_caller_intervention_keeps_both(prepared, tmp_path):
    """The run neither resets the subject between sessions nor removes the caller's lesion."""
    from brainscore_core.events import Perturbation, Selection, StateChange
    from brainscore_core.streaming_helpers import apply_state_change
    lesion = StateChange(kind='lesion', target=Selection(layer='projection'),
                         perturbation=Perturbation(kind='zero'))

    def run(name):
        model, state = _stateful_intervened_subject()
        benchmark = brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=prepared,
                                              batch_size=2, output_dir=tmp_path / name)
        # The caller's lesion must survive the run and be undone by its own handle.
        applied = apply_state_change(model, lesion)
        try:
            value = benchmark(model)
            assert state['lesioned']
        finally:
            apply_state_change(model, StateChange(kind='reset', handle_id=applied.handle_id))
        assert not state['lesioned'] and state['requests'] == 6
        return value, (tmp_path / name / 'item-scores.json').read_bytes()

    first_score, first_items = run('first')
    second_score, second_items = run('second')
    assert first_items == second_items
    assert repr(float(first_score)) == repr(float(second_score))
    # Lesioned, the even calls are flipped: contexts 1 of each batch are scored upside down.
    assert float(first_score) == 0.5


@pytest.mark.parametrize('mode, inputs, outputs', [
    ('logprobs', {'text'}, {'behavior:probabilities'}),
    ('choice', {'text'}, {'behavior:choice'}),
])
def test_declares_its_inputs_and_outputs(mode, inputs, outputs, prepared):
    from tests.benchmark_declarations import assert_declares, assert_preflight
    benchmark = brainscore.load_benchmark(f'EWoK-core-1.0-{mode}', root=prepared)
    assert_declares(benchmark, inputs, outputs)
    assert_preflight(benchmark)


def test_choice_item_scores_match_the_native_choice_answers(prepared, tmp_path):
    """Choice mode on the common channels gives the item scores the old
    request-channel path gave for the same native answers (all correct)."""
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-choice', root=prepared, batch_size=2,
                                          output_dir=tmp_path / 'run')
    value = benchmark(chooser())
    items = json.loads((tmp_path / 'run' / 'item-scores.json').read_text())
    assert [(item['accuracy'], item['invalid_answers']) for item in items] == [(1.0, 0)] * 3
    assert value.attrs['version_scores'] == {'0': 1.0, '1': 1.0}
    assert value.attrs['invalid_answers'] == 0


def test_choice_replay_rebuilds_one_session_per_request(prepared, tmp_path):
    from brainscore.experiments import compare_outputs, replay_sessions
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-choice', root=prepared, batch_size=2)
    original = Experiment(subject=chooser(), protocol=benchmark.protocol(),
                          tools=[RecordInputsOutputs()], output_dir=tmp_path / 'original').run()
    calls = []
    counting = chooser(lambda prompt: calls.append(prompt) or native_choice(prompt))
    replay = Experiment(subject=counting, protocol=replay_sessions(original.directory),
                        tools=[RecordInputsOutputs()], output_dir=tmp_path / 'replay').run()
    # Two batches, two questions per item: six prompts in four sessions.
    assert len(calls) == 6
    assert compare_outputs(original.directory, replay.directory)['equal']
