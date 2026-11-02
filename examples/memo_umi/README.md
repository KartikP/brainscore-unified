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

The checked-in [probability summary](results/summary.json) was produced with the pinned
worker requirements. At depth 1, for utterance `green`, the full output is
`P(green square)=0.43022581934928894`,
`P(green circle)=0.5697741508483887`, and `P(pink circle)=0.0`.

The complete recordings are generated locally. Their `.npy` files contain the
arrays referenced by the recording manifests and are needed to load or replay
that run. They are not versioned; the readable summary preserves the observed
tables and settings without dozens of generated files.

## Comparison notebook

Open [comparison.ipynb](comparison.ipynb) with a Python 3.11 UMI kernel and
`MEMO_PYTHON` set as above. It shows memo's model beside a small independent
[plain Python RSA reference](python_rsa.py), compares both full probability
tables, and runs both models through the same UMI experiment and input replay.
The notebook has executed outputs and generates fresh full recordings in a
temporary directory when rerun.

Python uses binary64 arithmetic; the pinned memo worker defaults to JAX
binary32. The notebook displays their measured differences without imposing a
cross-implementation tolerance. Each model must preserve its own native
predictions exactly through UMI and replay. The task has no physical clock, so
its events use `t_ms=None`.
