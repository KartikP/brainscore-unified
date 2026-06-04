"""Encoder-free neural feature extraction for Gemma-4-12B (gemma4_unified).

Gemma-4 has NO vision encoder — image patches project straight into the decoder.
So "vision features" for neural prediction are the decoder hidden states AT THE
IMAGE-TOKEN POSITIONS, at a chosen layer. This script (gemma4 env, 4-bit) introspects
the model (image token id, layer count, a probe forward's shapes), then extracts a
mean-pooled per-image feature vector at --layer for every image in a manifest, saving
features.npy + image_ids.json. The bsu env then PLS-regresses these against MajajHong
V4/IT neural data (the standard pipeline) and reports median per-neuroid Pearson r.

    HF_HUB_OFFLINE=1 python gemma_neural_extract.py --manifest /tmp/majaj_manifest.csv \
        --out /tmp/gemma_neural --layer 20 --probe
"""
import argparse, csv, json, os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)   # csv: stimulus_id,image_path
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='google/gemma-4-12B-it')
    ap.add_argument('--layer', type=int, default=20)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--probe', action='store_true', help='print introspection diagnostics and exit')
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    os.makedirs(args.out, exist_ok=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             llm_int8_skip_modules=['patch_dense', 'embedding_projection', 'lm_head'])
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, quantization_config=bnb, device_map='auto', dtype=torch.bfloat16).eval()
    dev = next(model.parameters()).device

    # --- locate the image token id + decoder ---
    cfg = model.config
    img_tok = (getattr(cfg, 'image_token_id', None) or getattr(cfg, 'image_token_index', None)
               or getattr(getattr(cfg, 'text_config', cfg), 'image_token_id', None))
    # decoder layers live under model.model.layers (or .language_model.model.layers)
    inner = model.model
    n_layers = len(getattr(inner, 'layers', getattr(getattr(inner, 'language_model', inner), 'layers', [])))

    rows = list(csv.DictReader(open(args.manifest)))
    if args.limit:
        rows = rows[:args.limit]

    def encode(path):
        img = Image.open(path).convert('RGB')
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img},
                                             {'type': 'text', 'text': 'Describe the object.'}]}]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors='pt').to(dev)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states                      # tuple: (n_layers+1) x (1, seq, hidden)
        layer = min(args.layer, len(hs) - 1)
        h = hs[layer][0]                            # (seq, hidden)
        ids = inputs['input_ids'][0]
        mask = (ids == img_tok) if img_tok is not None else torch.ones_like(ids, dtype=torch.bool)
        if mask.sum() == 0:                          # fallback: use all non-pad positions
            mask = torch.ones_like(ids, dtype=torch.bool)
        feat = h[mask].float().mean(0).cpu().numpy()
        return feat, int(mask.sum()), len(hs)

    if args.probe:
        feat, npatch, nhs = encode(rows[0]['image_path'])
        diag = {'image_token_id': img_tok, 'n_decoder_layers': n_layers,
                'n_hidden_states': nhs, 'image_patch_tokens': npatch,
                'feature_dim': int(feat.shape[0]), 'layer_used': min(args.layer, nhs - 1)}
        json.dump(diag, open(os.path.join(args.out, 'probe.json'), 'w'), indent=2)
        print(json.dumps(diag, indent=2), flush=True)
        return

    feats, ids = [], []
    for i, r in enumerate(rows):
        f, _, _ = encode(r['image_path'])
        feats.append(f); ids.append(r['stimulus_id'])
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(rows)}', flush=True)
    feats = np.stack(feats)
    np.save(os.path.join(args.out, 'features.npy'), feats)
    json.dump(ids, open(os.path.join(args.out, 'image_ids.json'), 'w'))
    print(json.dumps({'n': len(ids), 'feature_dim': int(feats.shape[1]),
                      'layer': args.layer, 'image_token_id': img_tok}, indent=2), flush=True)


if __name__ == '__main__':
    main()
