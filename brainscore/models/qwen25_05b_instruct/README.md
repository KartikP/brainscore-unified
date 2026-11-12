# Qwen2.5-0.5B-Instruct

Load with `brainscore.load_model('qwen2.5-0.5b-instruct')`. This standalone model
registration does not depend on the virtual lab. It uses
`Qwen/Qwen2.5-0.5B-Instruct`, pinned to revision
`7ae557604adf67be50417f59c2c2f167def9a775`.

Set `BRAINSCORE_QWEN25_05B_PATH` to an existing local checkpoint directory for
offline loading; otherwise the standard model-download preflight applies.
The registration chooses CUDA, MPS, then CPU, using float16 on accelerators and
float32 on CPU. A local checkpoint's weight files are hashed into provenance.

The causal-LM subject has a `reply` part (written answers through the chat
template, under the task's generation settings), a `likelihood` part (the summed
log probability of each option; plain text with no template under a
`raw_text` choice, as EWoK scores), neural recording and declared interventions.
It runs CogGym text experiments and both EWoK benchmarks.
Its `language_system` mapping to `model.layers.23` is a provisional last-layer
hypothesis, not a validated brain-region fit. Generated text can be factually
incorrect. Clients, including the lab, read its declared channels like any other subject's.
