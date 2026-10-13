"""Check attribution and exact patches on a cached GPT-2 model, on CPU.

This one-pair example tests the tools. It does not discover a general circuit,
measure held-out performance, or reproduce the arithmetic paper.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from brainscore.experiments import (
    CallableProtocol, Experiment, PatchActivity, RecordActivity,
    RecordInputsOutputs, TorchInstrumentation,
)
from brainscore.model_helpers.response_trace import build_trace_subject
from brainscore_core.events import Selection
from brainscore_core.streaming import StreamEvent


def run(model_path: Path, output_dir: Path, *, reference: str, comparison: str,
        top_k: int = 16) -> dict:
    if output_dir.exists():
        raise FileExistsError('Choose a new output directory')
    if top_k <= 0:
        raise ValueError('top_k must be positive')
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32,
    ).eval()
    if model.config.model_type != 'gpt2':
        raise ValueError('This example uses GPT-2 layer paths')
    inputs = {text: tokenizer(text, return_tensors='pt') for text in (reference, comparison)}
    if inputs[reference]['input_ids'].shape != inputs[comparison]['input_ids'].shape:
        raise ValueError('Use prompts with equal token counts; this example does not align tokens')

    def forward(text):
        # Bypass generation so gradients can reach the first answer's logits.
        return model(**inputs[text], use_cache=False).logits[0, -1]

    with torch.no_grad():
        answer_a = int(forward(reference).argmax())
        answer_b = int(forward(comparison).argmax())
    if answer_a == answer_b:
        raise ValueError('The model gives the same next token to both prompts; no contrast to localize')

    def preference(logits):
        # For two single tokens, the log-probability difference equals this logit difference.
        return logits[answer_a] - logits[answer_b]

    instrumentation = TorchInstrumentation(model)
    targets = [f'transformer.h.{i}.mlp.act' for i in range(model.config.n_layer)]
    analysis = instrumentation.attribute(
        targets,
        reference_call=lambda: forward(reference),
        comparison_call=lambda: forward(comparison),
        objective=preference,
        location=(0, -1),  # Last prompt token in the first batch item.
    )
    # Rank signed effects globally. These units were selected on this same pair.
    ranked = sorted(
        ((float(value), layer, unit)
         for layer, values in analysis['attributions'].items()
         for unit, value in enumerate(values) if value > 0),
        reverse=True,
    )
    if len(ranked) < top_k:
        raise ValueError('Fewer positive attribution values than requested units')
    chosen = {}
    for _, layer, unit in ranked[:top_k]:
        chosen.setdefault(layer, []).append(unit)
    selections = [Selection(layer, sorted(units)) for layer, units in chosen.items()]

    def respond(request):
        with torch.no_grad():
            logits = forward(request['text'])
        return {
            'text': tokenizer.decode([int(logits.argmax())]),
            'reasoning': None,
            'preference': float(preference(logits)),
        }

    subject = build_trace_subject(
        'gpt2-patch-check', provider=respond, parse=str,
        provenance={'checkpoint': str(model_path.resolve()), 'precision': 'float32', 'device': 'cpu'},
    )
    conditions = {'baseline': None, 'selected': selections}
    for seed in range(3):
        rng = np.random.default_rng(seed)
        conditions[f'random-{seed}'] = [
            Selection(s.layer, sorted(int(i) for i in rng.choice(
                analysis['reference']['shapes'][s.layer][-1], len(s.indices), replace=False,
            ))) for s in selections
        ]
    conditions['restored'] = None
    outputs = {}
    for condition, selection in conditions.items():
        def evaluate(subject, context):
            # Save estimates and source values with the same recorder as model calls.
            context.publish('attribution', analysis, source='TorchInstrumentation')
            return subject.process(StreamEvent('generation_request', {'text': comparison}, None))

        tools = [RecordInputsOutputs()]
        if selection is not None:
            tools.append(PatchActivity(selection, reference=analysis['reference']))
        tools.append(RecordActivity(selections))
        result = Experiment(
            subject=subject,
            protocol=CallableProtocol('paired-input-tool-check', evaluate),
            tools=tools,
            instrumentation=instrumentation,
            output_dir=output_dir / condition,
            metadata={
                'reference_prompt': reference, 'comparison_prompt': comparison,
                'reference_token': answer_a, 'comparison_token': answer_b,
                'selection_pair_is_evaluation_pair': True,
            },
        ).run()
        outputs[condition] = result.value.payload['raw']
    if outputs['baseline'] != outputs['restored']:
        raise AssertionError('Model response did not restore exactly')
    if any(module._forward_hooks for module in model.modules()):
        raise AssertionError('Experiment left hooks attached')
    summary = {
        'scope': 'one-pair trained-model tool check; not held-out localization',
        'reference_token': tokenizer.decode([answer_a]),
        'comparison_token': tokenizer.decode([answer_b]),
        'reference_value': analysis['reference_value'],
        'comparison_value': analysis['comparison_value'],
        'selections': {s.layer: s.indices for s in selections},
        'outputs': outputs,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference', default='The capital of France is')
    parser.add_argument('--comparison', default='The capital of Poland is')
    parser.add_argument('--top-k', type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run(args.model_path, args.output_dir, reference=args.reference,
                         comparison=args.comparison, top_k=args.top_k), indent=2))


if __name__ == '__main__':
    main()
