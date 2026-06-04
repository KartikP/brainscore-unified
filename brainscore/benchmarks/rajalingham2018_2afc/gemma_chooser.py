"""Standalone Gemma-4 2-AFC chooser — runs in an ISOLATED env (transformers 5.x,
no brainscore, so no dependency-pin conflict). Reads the montage manifest written
by score_2afc.compose_montages, asks Gemma-4 LEFT/RIGHT per montage, and writes
gemma_choices.csv. The bsu env then scores that CSV via benchmark.score_choices,
so the i2n is computed by the exact same pipeline as every other model.

Gemma-4-12B is apache-2.0 (ungated). 12B/BF16 won't fit a 23 GB A10G, so we load
4-bit nf4 (~7 GB). Two prompt modes mirror the Qwen runs: --mode direct (one-word
first-glance answer) vs --mode cot (Gemma-4 thinking mode, reason-then-answer).

    python gemma_chooser.py --manifest /tmp/raj2afc/montages/manifest.csv \\
        --out /tmp/raj2afc/gemma12b_direct --mode direct
"""
import argparse
import csv
import json
import os
import re


def parse_lr(text):
    t = text.upper()
    m = list(re.finditer(r'\b(LEFT|RIGHT)\b', t))
    if not m:
        return None
    return m[-1].group(1)            # last explicit LEFT/RIGHT


INSTR_DIRECT = ("The image shows a SAMPLE object on top and two options below (LEFT and RIGHT). "
                "Which option is the SAME object as the sample? "
                "Answer with exactly one word: LEFT or RIGHT.")
INSTR_COT = ("The image shows a SAMPLE object on top and two options below (LEFT and RIGHT). "
             "Exactly one option is the same object as the sample. "
             "Reason briefly, then end with a line: 'Answer: LEFT' or 'Answer: RIGHT'.")


def select_demos(rows, k):
    """Pick k balanced solved practice trials (correct side known), held out from
    the test set. Returns (demo_list, held_out_image_ids). The VLM analog of human
    practice trials; balanced LEFT/RIGHT also breaks any positional prior."""
    left, right = [], []
    seen = set()
    for r in rows:
        if r['image_id'] in seen:
            continue
        correct = 'LEFT' if r['sample_obj'] == r['left_obj'] else 'RIGHT'
        bucket = left if correct == 'LEFT' else right
        if len(bucket) < k // 2:
            bucket.append({'image_path': r['image_path'], 'answer': correct})
            seen.add(r['image_id'])
        if len(left) >= k // 2 and len(right) >= k // 2:
            break
    demos = []
    for a, b in zip(left, right):     # interleave so the demo order alternates sides
        demos += [a, b]
    return demos, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='google/gemma-4-12B-it')
    ap.add_argument('--mode', default='direct', choices=['direct', 'cot'])
    ap.add_argument('--n_shots', type=int, default=0,
                    help='in-context practice trials prepended to each prompt (0 = zero-shot)')
    ap.add_argument('--limit', type=int, default=0, help='cap trials (smoke); 0 = all')
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    os.makedirs(args.out, exist_ok=True)
    instr = INSTR_DIRECT if args.mode == 'direct' else INSTR_COT
    max_new = 8 if args.mode == 'direct' else 512
    think = (args.mode == 'cot')

    print(f'loading {args.model} (4-bit nf4)…', flush=True)
    # Keep the encoder-free vision-embedding (patch_ln1/patch_dense) and lm_head in
    # bf16: Gemma-4 casts pixel_values to patch_dense.weight.dtype, which under 4-bit
    # is Byte -> layernorm crashes. These layers are tiny, so skipping them is cheap.
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             # bnb matches LEAF module names: skip the vision-embedder
                             # Linear layers (patch_dense casts pixels to its weight dtype,
                             # which under 4-bit is Byte -> layernorm crash) + lm_head.
                             llm_int8_skip_modules=['patch_dense', 'embedding_projection', 'lm_head'])
    rev = 'e18f459f54832f4ae2ab6686b935a2268668a9e9' if args.model == 'google/gemma-4-12B-it' else None
    proc = AutoProcessor.from_pretrained(args.model, revision=rev)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, revision=rev, quantization_config=bnb, device_map='auto', dtype=torch.bfloat16).eval()
    dev = next(model.parameters()).device

    all_rows = list(csv.DictReader(open(args.manifest)))
    demos, demo_imgs = ([], set())
    if args.n_shots:
        demos, demo_imgs = select_demos(all_rows, args.n_shots)
        demos = [{'img': Image.open(d['image_path']).convert('RGB'), 'answer': d['answer']} for d in demos]
        print(f'{len(demos)} practice demos (held out): '
              f"{[d['answer'] for d in demos]}", flush=True)
    rows = [r for r in all_rows if r['image_id'] not in demo_imgs]   # no leakage
    if args.limit:
        rows = rows[:args.limit]
    print(f'{len(rows)} test trials, mode={args.mode}, n_shots={args.n_shots}', flush=True)

    def demo_turns():
        msgs = []
        for d in demos:                                   # each practice trial: user(image+instr) -> assistant(answer)
            msgs.append({'role': 'user', 'content': [{'type': 'image', 'image': d['img']},
                                                     {'type': 'text', 'text': instr}]})
            msgs.append({'role': 'assistant', 'content': [{'type': 'text', 'text': f'Answer: {d["answer"]}'}]})
        return msgs

    out_rows, parse_miss = [], 0
    for i, r in enumerate(rows):
        img = Image.open(r['image_path']).convert('RGB')
        messages = demo_turns() + [{'role': 'user', 'content': [{'type': 'image', 'image': img},
                                                                {'type': 'text', 'text': instr}]}]
        kw = {}
        try:
            inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                              return_dict=True, return_tensors='pt',
                                              enable_thinking=think).to(dev)
        except TypeError:        # processor may not accept enable_thinking
            inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                              return_dict=True, return_tensors='pt').to(dev)
        ilen = inputs['input_ids'].shape[-1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        ans = proc.decode(out[0][ilen:], skip_special_tokens=True)
        side = parse_lr(ans)
        if side is None:
            parse_miss += 1
            side = 'LEFT' if (i % 2 == 0) else 'RIGHT'   # deterministic fallback
        choice = r['left_obj'] if side == 'LEFT' else r['right_obj']
        out_rows.append({'image_id': r['image_id'], 'sample_obj': r['sample_obj'],
                         'dist_obj': r['dist_obj'], 'choice': choice, 'side': side,
                         'correct': str(choice == r['sample_obj'])})
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(rows)}  parse_miss={parse_miss}', flush=True)

    with open(os.path.join(args.out, 'gemma_choices.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    acc = sum(r['correct'] == 'True' for r in out_rows) / len(out_rows)
    frac_left = sum(r['side'] == 'LEFT' for r in out_rows) / len(out_rows)
    summary = {'model': args.model, 'mode': args.mode, 'n_shots': args.n_shots, 'n': len(out_rows),
               'accuracy': round(acc, 4), 'frac_left': round(frac_left, 4),
               'parse_miss': parse_miss}
    json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
