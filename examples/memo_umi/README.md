# memo through Brain-Score UMI

This example reproduces the memo Rational Speech Acts speaker–listener task in
the paper: a speaker chooses a referent from a green square, green circle, and
pink circle, then says green, pink, square, or round. The subject returns a
probability distribution over the three referents for each utterance.

The native memo implementation stays inside a Python 3.12+ worker. UMI stays in
its Python 3.11 environment. Install CPU JAX and `memo-lang` into a separate
worker environment, then run from the UMI environment:

```sh
python3.12 -m venv /private/tmp/memo-rsa-py312
/private/tmp/memo-rsa-py312/bin/python -m pip install --no-cache-dir \
  -r examples/memo_umi/worker-requirements.txt
conda activate brainscore-unified
MEMO_PYTHON=/private/tmp/memo-rsa-py312/bin/python \
  python examples/memo_umi/run.py --out /private/tmp/memo-umi-runs
```

The script evaluates literal depth 0 and recursive depth 1 at beta 1.0 on the
same utterances. It writes exact direct-memo and UMI tables, worker/runtime
versions, configuration, UMI input/output recordings, and input-replay results
under the selected output directory. Input replay feeds recorded events to the
subject and reruns memo; it does not replay saved responses. The script requires
the recorded input events to match across depths, direct-memo and UMI arrays to
be exactly equal, and replay outputs to match exactly. Use another fresh output
directory for each run.

The paper prints the recursive table to two decimal places; those displayed
values are illustrative summaries. The script writes the full runtime values.
This is a model-configuration comparison, not a human-data fit or a claim about
human cognition.

The checked-in [results](results/summary.json) were produced with the pinned
worker requirements. At depth 1, for utterance `green`, the full output is
`P(green square)=0.43022581934928894`,
`P(green circle)=0.5697741508483887`, and `P(pink circle)=0.0`. The saved UMI
records contain the same outputs and inputs for both depths and their replays.
