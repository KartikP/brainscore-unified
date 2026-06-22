"""GPT-2 lexical decision via the GENERATION/likelihood path.

GPT-2 isn't instruction-tuned, so it has no "answer the question" path. But it
is a generative LM, and a generative model assigns higher likelihood to real
words than to pseudo-words. So the generative lexical-decision is: score each
word by GPT-2's mean per-token log-probability, fit a 1-D logistic threshold on
the 400 train words, predict real/pseudo on the 100 test words. This is the
likelihood-based generation path — complementary to GPT-2's feature readout.

Run on EC2. Prints the accuracy.
"""
import os

import numpy as np
import torch


def main():
    import brainscore  # noqa: ensures ROAR benchmark importable
    bench = brainscore.load_benchmark('Yeatman2021-lexical_decision-text')
    train, test = bench._train_stimuli, bench._test_stimuli

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()

    def mean_logprob(word):
        ids = tok(' ' + str(word), return_tensors='pt').input_ids.to(device)
        if ids.shape[1] < 2:
            ids = tok(str(word) + '.', return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            out = model(ids, labels=ids)
        # loss is mean NLL; log-prob = -loss
        return float(-out.loss.item())

    def feats_labels(stim):
        x, y = [], []
        for _, row in stim.iterrows():
            x.append(mean_logprob(row['sentence']))
            y.append(1 if row['image_label'] == 'real' else 0)
        return np.array(x).reshape(-1, 1), np.array(y)

    Xtr, ytr = feats_labels(train)
    Xte, yte = feats_labels(test)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression().fit(Xtr, ytr)
    acc = float((clf.predict(Xte) == yte).mean())
    # also report raw separability (real should have higher mean logprob)
    print(f'mean logprob  real={Xtr[ytr==1].mean():.3f}  pseudo={Xtr[ytr==0].mean():.3f}', flush=True)
    print(f'GPT-2 generation (likelihood) ROAR accuracy = {acc:.3f}', flush=True)


if __name__ == '__main__':
    main()
