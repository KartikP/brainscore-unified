# Qwen2.5-0.5B-Instruct

Load with `brainscore.load_model('qwen2.5-0.5b-instruct')`. This standalone model
registration does not depend on the virtual lab. It uses
`Qwen/Qwen2.5-0.5B-Instruct`, pinned to revision
`7ae557604adf67be50417f59c2c2f167def9a775`.

Set `BRAINSCORE_QWEN25_05B_PATH` to an existing local checkpoint directory for
offline loading; otherwise the standard model-download preflight applies.
The registration chooses CUDA, MPS, then CPU, using float16 on accelerators and
float32 on CPU. A local checkpoint's weight files are hashed into provenance.

The causal-LM subject supports generation requests and verbatim response traces,
conditional log probabilities, neural recording and declared interventions.
Its `language_system` mapping to `model.layers.23` is a provisional last-layer
hypothesis, not a validated brain-region fit. Generated text can be factually
incorrect. Clients, including the lab, discover the registered channels normally.
