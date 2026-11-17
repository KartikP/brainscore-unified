# Ways to answer

Each registered model answers with the parts its registration wires: writing (a reply it generates), likelihood (the log probability of each option) or a readout (a classifier fitted on one of its layers). The table lists what each registration declares, and for each kind of answer the one part that gives it.

| Models | Inputs | Regions | Ways | Kinds of answers (the way that gives each) |
| --- | --- | --- | --- | --- |
| `blip2-opt-2.7b` | control, instruction, text, vision:image | IT, V1, V2, V4, language_system | writing, likelihood, readout | choice (writing), log_probabilities (likelihood), probabilities (readout), rating (readout), rt (readout), text (writing) |
| `blip2-wav2vec2`, `clip-random-wav2vec2`, `clip-wav2vec2`, `qwen2.5-vl-wav2vec2`, `random-clip-random-wav2vec2`, `random-clip-wav2vec2` | audio, control, vision:image | A1, IT, V1, V2, V4 | - | - |
| `chance-baseline` | audio, text, vision:image, vision:video | - | - | choice (own code), probabilities (own code) |
| `claude-haiku-behavioral`, `claude-opus-behavioral`, `gpt-4o-behavioral`, `or-claude-sonnet-behavioral` | control, instruction, text, vision:image | - | writing | choice (writing), rating (writing), text (writing) |
| `clip-vit-b-32`, `random-vit-b-32` | control, text, vision:image | IT, V1, V2, V4, language_system | readout | choice (readout), probabilities (readout), rating (readout), rt (readout) |
| `deepseek-chat-behavioral`, `deepseek-r1-behavioral`, `llama-3.3-70b-behavioral` | control, instruction, text | - | writing | choice (writing), rating (writing), text (writing) |
| `gpt2-small` | control, instruction, text | language_system | likelihood, readout | choice (readout), log_probabilities (likelihood), probabilities (readout), rating (readout), rt (readout) |
| `qwen2.5-0.5b-instruct` | control, instruction, lesion, pharmacological, stimulation, text | language_system | writing, likelihood | choice (writing), log_probabilities (likelihood), probabilities (likelihood), rating (writing), rt (likelihood), text (writing) |
| `qwen2.5-vl-3b`, `qwen2.5-vl-3b-bf16`, `qwen2.5-vl-7b` | control, instruction, text, vision:image | IT, V1, V2, V4, language_system | writing, likelihood, readout | choice (writing), log_probabilities (likelihood), probabilities (readout), rating (writing), rt (readout), text (writing) |
| `qwen2.5-vl-3b-vwfa`, `qwen2.5-vl-7b-vwfa` | control, instruction, lesion, pharmacological, stimulation, text, vision:image | IT, V1, V2, V4, VWFA, language_system | writing, likelihood, readout | choice (writing), log_probabilities (likelihood), probabilities (likelihood), rating (writing), rt (likelihood), text (writing) |
| `qwen3-vl-2b` | control, instruction, lesion, pharmacological, stimulation, text, vision:image, vision:video | IT, V1, V2, V4, VWFA, language_system | writing, likelihood, readout | choice (writing), log_probabilities (likelihood), probabilities (readout), rating (writing), rt (readout), text (writing) |
| `qwen3.6-27b` | control, instruction, lesion, pharmacological, stimulation, text, vision:image, vision:video | IT, V1, V2, V4, language_system | writing, likelihood, readout | choice (writing), log_probabilities (likelihood), probabilities (readout), rating (writing), rt (readout), text (writing) |
| `random-vjepa1-random-wav2vec2`, `random-vjepa1-wav2vec2`, `vjepa1-random-wav2vec2`, `vjepa1-wav2vec2` | audio, control, vision:video | A1, IT, V1, V4 | - | - |
| `random-wav2vec2-base` | audio, control | A1 | - | - |
| `smolvlm2-256m` | control, instruction, lesion, pharmacological, stimulation, text, vision:image | IT, V1, V2, V4, VWFA, language_system | writing, likelihood, readout | choice (writing), log_probabilities (likelihood), probabilities (readout), rating (writing), rt (readout), text (writing) |
| `videomae-base`, `vjepa1-vitl`, `vjepa2-vitl` | control, vision:video | IT, V1, V2, V4 | - | - |

"own code": the subject is not a `BrainScoreModel`; its own `participate` gives the answer (`chance-baseline` gives uniform probabilities).

A model that answers through a provider's API builds without a key and without a network; it needs the provider's key only to answer.

The table is generated from the registrations with stand-in parts (no weights, no network). Regenerate it from the folder `unified`:

```bash
python -m tests.ways_to_answer docs/reference/ways_to_answer.md
```
