#!/usr/bin/env python
"""Donor poster slide 2: a lesion protocol run on Qwen2.5-VL models as lab subjects.

One *agent* = one model size x one decoding temperature x one sampling seed.
Protocol per agent (same 100 held-out word images every session; behavior = the
model's spoken answer "real" or "pseudo"; activity = mean activation of five late
language layers, recorded on every trial):
  S1 baseline   : present the 100 images, record behavior + activity
  localizer     : present the 400 training images once per model (greedy, deterministic);
                  unit selectivity = Cohen's d (real minus pseudoword) per unit per layer
  S2 lesion     : silence the K most pseudoword-selective units per layer
                  (K = frac x hidden width) through process(StateChange); re-run S1
  S3 control    : silence K random units per layer (size-matched); re-run S1
  S4 opposite   : silence the K most real-word-selective units per layer; re-run S1
  S5 restored   : reset(); re-run S1
Outputs: per-trial table (CSV), per-session summaries (JSON), localizer selectivity and
selected units, and per-trial recorded activity (npz) for each agent and session.
"""
import argparse, csv, gc, json, os, random, re, time, warnings
from pathlib import Path
import numpy as np

T0 = time.time()
def log(m): print(f'[{time.time()-T0:7.1f}s] {m}', flush=True)

def short(name): return name.split('/')[-1].replace('-Instruct', '').replace('Qwen2.5-VL-', 'qwen2.5-vl-').lower()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='Qwen/Qwen2.5-VL-3B-Instruct,Qwen/Qwen2.5-VL-7B-Instruct,Qwen/Qwen2.5-VL-32B-Instruct')
    ap.add_argument('--temps', default='0,0.5,1.0')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--frac', type=float, default=0.25)
    ap.add_argument('--depth', default='0.72,0.94', help='relative depth range for the five recorded/lesioned layers')
    ap.add_argument('--max_trials', type=int, default=0, help='smoke test: use only the first N test images')
    ap.add_argument('--max_loc', type=int, default=0, help='smoke test: localizer images per class')
    ap.add_argument('--out', default='/opt/dlami/nvme/donor2/out_lesion')
    args = ap.parse_args()
    OUT = Path(args.out); (OUT / 'activations').mkdir(parents=True, exist_ok=True); (OUT / 'localizer').mkdir(exist_ok=True)
    temps = [float(t) for t in args.temps.split(',')]; seeds = [int(s) for s in args.seeds.split(',')]
    d0, d1 = [float(x) for x in args.depth.split(',')]

    import torch
    from PIL import Image
    import brainscore
    from brainscore_core.model_interface import BrainScoreModel, StateChange, Selection, Perturbation, TaskContext
    from brainscore.perturbation import build_pytorch_ablation_fn
    from brainscore_core.hf_compat import pin_image_processor
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    benchmark = brainscore.load_benchmark('Yeatman2021-lexical_decision-image')
    train = benchmark._train_stimuli; test = benchmark._test_stimuli
    if args.max_trials: test = test.iloc[:args.max_trials]
    truth = {str(r['stimulus_id']): str(r['image_label']) for _, r in test.iterrows()}
    log(f'benchmark: {len(train)} localizer images, {len(test)} test images, human ceiling {float(benchmark.ceiling):.3f}')
    INSTRUCTION = 'Is this a real word or a pseudo word?'
    LABELS = ['real', 'pseudo']

    trial_rows = []
    results = {'protocol': {'frac_units_per_layer': args.frac, 'depth_range': [d0, d1], 'n_layers': 5,
                            'temps': temps, 'seeds': seeds, 'instruction': INSTRUCTION, 'n_test': len(test),
                            'n_localizer': len(train) if not args.max_loc else 2 * args.max_loc,
                            'sessions': ['baseline', 'lesion_pseudo_selective', 'control_random', 'lesion_real_selective', 'restored'],
                            'dtype': 'bfloat16', 'max_new_tokens': 8}, 'models': {}}
    res_path = OUT / 'results_lesion.json'

    for name in args.models.split(','):
        tag = short(name)
        log(f'=== {name} ===')
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, torch_dtype=torch.bfloat16, device_map='auto').eval()
        processor = AutoProcessor.from_pretrained(name)
        try: processor = pin_image_processor(processor, name)
        except Exception as e: log(f'pin_image_processor skipped: {e}')
        cfg = model.config; tcfg = getattr(cfg, 'text_config', cfg)
        L = int(tcfg.num_hidden_layers); H = int(tcfg.hidden_size)
        layer_ids = sorted(set(int(round(x)) for x in np.linspace(d0 * (L - 1), d1 * (L - 1), 5)))
        mods = dict(model.named_modules())
        paths = [f'model.language_model.layers.{i}.mlp' for i in layer_ids]
        if paths[0] not in mods:
            paths = [p for p in (f'language_model.layers.{i}.mlp' for i in layer_ids)]
        assert all(p in mods for p in paths), f'cannot resolve {paths[0]}; have e.g. {[m for m in mods if m.endswith("mlp")][:3]}'
        K = int(round(args.frac * H))
        in_dev = next(model.parameters()).device
        log(f'{tag}: {L} layers, hidden {H}, recording/lesion layers {layer_ids}, K={K} units per layer, device_map={getattr(model, "hf_device_map", in_dev)}')

        # ---- recording hooks: mean over tokens of each recorded layer's output, first forward of each trial ----
        rec = {'armed': False, 'buf': {}}
        def make_hook(p):
            def hook(_m, _i, out):
                if not rec['armed'] or p in rec['buf']: return
                h = out[0] if isinstance(out, tuple) else out
                rec['buf'][p] = h.detach().float().mean(dim=1).squeeze(0).cpu().numpy()
            return hook
        handles = [mods[p].register_forward_hook(make_hook(p)) for p in paths]

        gen_cfg = {'temperature': 0.0, 'seed': 0, 'trial': 0, 'responses': []}
        def generation_fn(stimulus_row, instruction, label_set):
            image = Image.open(stimulus_row['image_file_name']).convert('RGB')
            messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                        {'type': 'text', 'text': instruction + f' Answer with exactly one of: {", ".join(label_set)}.'}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], return_tensors='pt', padding=True).to(in_dev)
            T = gen_cfg['temperature']
            torch.manual_seed(gen_cfg['seed'] * 100003 + gen_cfg['trial'])
            rec['buf'] = {}; rec['armed'] = True
            with torch.no_grad():
                kw = dict(max_new_tokens=8, do_sample=T > 0)
                if T > 0: kw.update(temperature=T, top_p=1.0, top_k=0)
                out = model.generate(**inputs, **kw)
            rec['armed'] = False
            resp = processor.tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().lower()
            gen_cfg['responses'].append((str(stimulus_row['stimulus_id']), resp, {p: rec['buf'].get(p) for p in paths}))
            gen_cfg['trial'] += 1
            for lbl in label_set:
                if resp == lbl.lower(): return lbl
            for lbl in label_set:
                if re.search(rf'\b{re.escape(lbl.lower())}\b', resp): return lbl
            return resp

        bs = BrainScoreModel(identifier=f'{tag}-lesion-demo', model=model, generation_fn=generation_fn,
                             state_change_fn=build_pytorch_ablation_fn(model), supported_modalities={'vision'})

        # ---- localizer (once per model, greedy) ----
        loc = train
        if args.max_loc:
            loc = train.groupby('image_label', group_keys=False).head(args.max_loc)
        acts = {p: [] for p in paths}; labels = []
        for _, row in loc.iterrows():
            image = Image.open(row['image_file_name']).convert('RGB')
            messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': INSTRUCTION}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], return_tensors='pt').to(in_dev)
            rec['buf'] = {}; rec['armed'] = True
            with torch.no_grad(): model.generate(**inputs, max_new_tokens=1, do_sample=False)
            rec['armed'] = False
            for p in paths: acts[p].append(rec['buf'][p])
            labels.append(str(row['image_label']))
        labels = np.array(labels)
        sel = {}; units = {}
        rng = random.Random(0)
        for p in paths:
            A = np.stack(acts[p]); a_real = A[labels == 'real']; a_pseudo = A[labels == 'pseudo']
            pooled = np.sqrt((a_real.var(0) + a_pseudo.var(0)) / 2 + 1e-6)
            d = (a_real.mean(0) - a_pseudo.mean(0)) / pooled
            sel[p] = d
            order = np.argsort(d)
            units[p] = {'lesion_pseudo_selective': order[:K].tolist(), 'lesion_real_selective': order[-K:].tolist(),
                        'control_random': rng.sample(range(H), K)}
        np.savez(OUT / 'localizer' / f'{tag}_localizer.npz', **{f'd__{p}': sel[p].astype(np.float32) for p in paths},
                 **{f'acts__{p}': np.stack(acts[p]).astype(np.float16) for p in paths}, labels=labels,
                 stimulus_id=np.array(loc['stimulus_id'].astype(str)))
        json.dump({'layers': paths, 'K': K, 'units': units}, open(OUT / 'localizer' / f'{tag}_units.json', 'w'))
        overlap = {p: len(set(units[p]['lesion_pseudo_selective']) & set(units[p]['lesion_real_selective'])) for p in paths}
        log(f'{tag}: localizer done on {len(loc)} images; pseudo/real selected-set overlap per layer {list(overlap.values())}')

        task = TaskContext(task_type='probabilities', fitting_stimuli=train, label_set=LABELS, instruction=INSTRUCTION, prefer_path='generation')
        model_res = {'n_layers_total': L, 'hidden': H, 'layers': paths, 'K': K, 'agents': {}}
        results['models'][tag] = model_res

        def session(agent, sname, temperature, seed):
            gen_cfg.update(temperature=temperature, seed=seed, trial=0, responses=[])
            bs.start_task(task)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                preds = bs.process(test)
            choice = list(preds['choice'].values); idx = preds.values.argmax(1)
            sids = [str(s) for s in preds['stimulus_id'].values]
            lab = [choice[i] for i in idx]
            resp_map = {s: (r, a) for s, r, a in gen_cfg['responses']}
            correct = []; per_label = {'real': [], 'pseudo': []}
            A = {p: [] for p in paths}
            for s, l in zip(sids, lab):
                t = truth[s]; c = int(l == t); correct.append(c); per_label[t].append(c)
                r, a = resp_map[s]
                parsed = l if (l.lower() in r) else l
                trial_rows.append([tag, agent, sname, temperature, seed, s, t, r, l, c, int(not any(x in r for x in LABELS))])
                for p in paths: A[p].append(a[p] if a[p] is not None else np.full(H, np.nan, np.float32))
            np.savez(OUT / 'activations' / f'{tag}__{agent}__{sname}.npz', stimulus_id=np.array(sids), truth=np.array([truth[s] for s in sids]),
                     answer=np.array(lab), **{f'acts__{p}': np.stack(A[p]).astype(np.float16) for p in paths})
            out = {'accuracy': float(np.mean(correct)), 'accuracy_real': float(np.mean(per_label['real'])) if per_label['real'] else None,
                   'accuracy_pseudo': float(np.mean(per_label['pseudo'])) if per_label['pseudo'] else None,
                   'n_unparsed': int(sum(1 for s in sids if not any(x in resp_map[s][0] for x in LABELS))),
                   'dyslexic_by_paper_threshold': bool(np.mean(correct) < 0.65)}
            log(f'  {tag} {agent} {sname:24s} acc={out["accuracy"]:.3f} real={out["accuracy_real"]} pseudo={out["accuracy_pseudo"]} unparsed={out["n_unparsed"]}')
            return out

        def lesion(kind):
            applied = []
            for p in paths:
                applied.append(bs.process(StateChange(kind='ablation', target=Selection(layer=p, indices=units[p][kind]),
                                                      perturbation=Perturbation(kind='zero'))))
            return applied

        agents = [(0.0, 0)] if 0.0 in temps else []
        agents += [(t, s) for t in temps if t > 0 for s in seeds]
        for temperature, seed in agents:
            agent = f'T{temperature:g}_s{seed}'
            ar = {'temperature': temperature, 'seed': seed, 'sessions': {}}
            model_res['agents'][agent] = ar
            ar['sessions']['baseline'] = session(agent, 'baseline', temperature, seed)
            for kind in ('lesion_pseudo_selective', 'control_random', 'lesion_real_selective'):
                lesion(kind)
                ar['sessions'][kind] = session(agent, kind, temperature, seed)
                bs.reset()
            ar['sessions']['restored'] = session(agent, 'restored', temperature, seed)
            json.dump(results, open(res_path, 'w'), indent=1)
            with open(OUT / 'trials.csv', 'w', newline='') as fh:
                w = csv.writer(fh); w.writerow(['model', 'agent', 'session', 'temperature', 'seed', 'stimulus_id', 'truth', 'response', 'answer', 'correct', 'unparsed'])
                w.writerows(trial_rows)

        for h in handles: h.remove()
        del bs, model, processor; gc.collect(); torch.cuda.empty_cache()
        log(f'{tag} done')
    log('DONE')

if __name__ == '__main__':
    main()
