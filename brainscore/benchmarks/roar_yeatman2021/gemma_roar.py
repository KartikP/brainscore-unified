"""Standalone Gemma-4 ROAR lexical-decision chooser (isolated gemma4 env).

ROAR (Yeatman 2021 / Honarmand 2026): the model sees a rendered letter-string
image and decides real word vs pseudo-word. This is the generation path — zero-shot
instruction-following, no readout fit. Reads the test manifest dumped by the bsu
env (stimulus_id, image_path, numeric label), asks Gemma yes/no per image, writes
gemma_roar_choices.csv + a summary (accuracy, ceiled by the human mean, dyslexia flag).

Same 4-bit setup as gemma_chooser (skip the encoder-free vision-embedder leaves).
    HF_HUB_OFFLINE=1 python gemma_roar.py --manifest /tmp/roar_test_manifest.csv \
        --out /tmp/roar_gemma --ceiling 0.8112
"""
import argparse, csv, json, os, re

INSTR = ("The image shows a letter string. Is it a real English word "
         "(not a made-up/pseudo-word)? Answer with exactly one word: yes or no.")
THRESHOLD = 0.65   # Honarmand 2026 dyslexia threshold (raw accuracy)
# Pin the original revision: a newer repo commit (66bc78a) breaks offline processor
# loading; e18f459 (the one that first worked) loads cleanly offline.
GEMMA_REV = 'e18f459f54832f4ae2ab6686b935a2268668a9e9'


def parse_yesno(text):
    t = text.lower()
    m = list(re.finditer(r'\b(yes|no)\b', t))
    return m[-1].group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='google/gemma-4-12B-it')
    ap.add_argument('--ceiling', type=float, default=0.8112)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    os.makedirs(args.out, exist_ok=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             llm_int8_skip_modules=['patch_dense', 'embedding_projection', 'lm_head'])
    print(f'loading {args.model} (4-bit)…', flush=True)
    proc = AutoProcessor.from_pretrained(args.model, revision=GEMMA_REV)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, revision=GEMMA_REV, quantization_config=bnb, device_map='auto', dtype=torch.bfloat16).eval()
    dev = next(model.parameters()).device

    rows = list(csv.DictReader(open(args.manifest)))
    if args.limit:
        rows = rows[:args.limit]
    print(f'{len(rows)} ROAR test stimuli', flush=True)

    out, miss, rng_i = [], 0, 0
    for i, r in enumerate(rows):
        img = Image.open(r['image_path']).convert('RGB')
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img},
                                             {'type': 'text', 'text': INSTR}]}]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors='pt').to(dev)
        ilen = inputs['input_ids'].shape[-1]
        with torch.no_grad():
            o = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        ans = proc.decode(o[0][ilen:], skip_special_tokens=True)
        yn = parse_yesno(ans)
        if yn is None:
            miss += 1
            yn = 'yes' if (i % 2 == 0) else 'no'
        pred = 1 if yn == 'yes' else 0
        label = int(r['label'])
        out.append({'stimulus_id': r['stimulus_id'], 'label': label, 'prediction': pred,
                    'correct': int(pred == label)})
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(rows)} miss={miss}', flush=True)

    with open(os.path.join(args.out, 'gemma_roar_choices.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    acc = sum(r['correct'] for r in out) / len(out)
    real_acc = sum(r['correct'] for r in out if r['label'] == 1) / max(1, sum(r['label'] == 1 for r in out))
    pseudo_acc = sum(r['correct'] for r in out if r['label'] == 0) / max(1, sum(r['label'] == 0 for r in out))
    summary = {'model': args.model, 'benchmark': 'Yeatman2021-lexical_decision (generation)',
               'n': len(out), 'accuracy': round(acc, 4), 'ceiling': args.ceiling,
               'ceiled': round(acc / args.ceiling, 4), 'real_word_acc': round(real_acc, 4),
               'pseudo_word_acc': round(pseudo_acc, 4), 'parse_miss': miss,
               'dyslexic': bool(acc < THRESHOLD)}
    json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
