# Record model-generated reasoning

Use `RecordReasoning` to preserve exposed reasoning separately from the final answer. It saves completed text and streamed fragments using the same `RunRecorder` / `RunRecord` format as other tools.

**Feature-branch addition:** this recorder and streaming support require the accompanying code changes; they are not yet merged into v2. The example uses a synthetic provider to demonstrate recording, not a trained model's reasoning quality.

## The three jobs

| Component | Job |
| --- | --- |
| Provider adapter | Call the model and identify which returned fields contain reasoning and the answer. |
| `build_trace_subject` | Present those responses through UMI; parse final answers and preserve raw responses. |
| `RecordReasoning` | Save exposed reasoning with its response, input identifier, condition, trial, and timing. |

`RecordInputsOutputs` is required alongside `RecordReasoning`: it preserves the linked inputs and complete experiment log. The reasoning tool adds a focused record in `reasoning/`. Ordinary input/output recording already retains any reasoning present in a response; this tool makes that reasoning consistently accessible for analysis.

## Choose the response format

For a completed response, the provider returns:

```python
{
    'text': '42',  # Final answer text, supplied to the answer parser.
    'reasoning': {
        'text': 'Six groups of seven give forty-two.',
        'kind': 'chain_of_thought',
        'format': 'complete',
    },
}
```

`reasoning` accepts a string, a block, or a list of blocks. A bare string is labeled `kind='unspecified'`, `format='complete'`. Missing or `None` reasoning is recorded as unavailable; an explicitly empty value remains distinct. Extra block metadata is preserved.

| Field | Meaning |
| --- | --- |
| `text` | The exposed reasoning text, unchanged |
| `kind` | Its declared type, such as `chain_of_thought` or `summary`; UMI does not infer the type |
| `format='complete'` | A complete text block |
| `format='delta'` | Only newly emitted text |
| `format='snapshot'` | The provider's current cumulative text |

For streaming, pass `streaming=True` to `build_trace_subject`. The provider yields dictionaries with `text` and a boolean `final`. Intermediate `text` can be empty. Exactly one final fragment must end the stream and contain the complete answer text:

```python
def provider(request):
    # Replace this fixture with a loop over your model's generation stream.
    yield {
        'text': '',
        'reasoning': {'text': 'Six groups of ', 'format': 'delta'},
        'final': False,
    }
    yield {
        'text': '',
        'reasoning': {'text': 'seven give forty-two.', 'format': 'delta'},
        'final': False,
    }
    yield {'text': '42', 'final': True}
```

A session emits each fragment as it arrives. A direct `process()` call waits for completion and retains every raw fragment in `response.payload['stream']`. Only final answer text is parsed. A missing final fragment or content after the final fragment fails the run. Use a session to preserve partial responses if generation fails; a direct call returns no response until it finishes.

Fragments retain their order and input identifiers. `t_ms` remains the input's experiment time; `elapsed_s` in a session record indicates when UMI received each fragment. A direct call cannot reconstruct per-fragment arrival times. Deltas and snapshots are kept separate, without guessed concatenation or deduplication.

## Attach and read the recorder

```python
from brainscore.experiments import Experiment, RecordInputsOutputs, RecordReasoning
from brainscore.run_record import RunRecord

# subject and protocol define your model and experiment procedure.
result = Experiment(
    subject=subject,
    protocol=protocol,
    tools=[RecordInputsOutputs(), RecordReasoning()],
    output_dir='runs/reasoning',
).run()

# Use the same reader as for other UMI records.
for response in RunRecord(result.directory / 'reasoning').outputs():
    print(response['trial_id'], response['reasoning_available'])
    for block in response['reasoning']:
        print(block['format'], block['text'])
```

For another subject's output format, supply `RecordReasoning(extract=...)`. The callback receives the original output and returns reasoning text/blocks or `None`. For example, `extract=lambda output: output.get('thinking')` selects an explicit field in a dictionary response.

The recorder does not request CoT or guess delimiters such as `<think>`. Prompting, generation settings, and provider-specific parsing belong in the model adapter. A provider summary is not a full trace, and generated explanations do not establish the model's internal causal process. Hidden or encrypted reasoning cannot be recovered by this tool.

Run the [complete offline example](../examples/experiment_toolbox/reasoning.py) with:

```bash
python examples/experiment_toolbox/reasoning.py --out /tmp/umi-reasoning-demo
```
