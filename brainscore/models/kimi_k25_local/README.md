# Kimi K2.5 on your own GPUs: vision hooks and face localizers

An experiment folder for one study: Kimi K2.5 (`moonshotai/Kimi-K2.5`, revision `94b44e57`, modified MIT
licence) takes part in a NodeKit task on our own machine, and chosen channels of its vision part (MoonViT,
27 blocks of 1152 channels) are inactivated during the task.

- The model: `load_model('kimi-k2.5-local')` (registered in `api_closed`). It reaches a vLLM server through
  the `vllm` provider at `BRAINSCORE_VLLM_URL` (default `http://localhost:8000/v1`). The served weights are the
  released ones: the routed experts of the language part in native INT4 (weight-only, group 32), the vision
  part, the projector and the attention in BF16. Its replies are never stored.
- `vllm_hooks.py`: the class mixed into each vLLM worker; it records block outputs and inactivates channels
  (or projects out a direction) at one block's output for every token, from a lesion file read at each
  encoder call. The module docstring gives the server command and the settings (prefix cache, processor
  cache and encoder CUDA graph off; reset the encoder cache after each change of the lesion file).
- `localizer.py`: four face localizers in the vision part alone (one GPU): L1 the fLoc contrast, L2 an
  ablation effect on a linear face probe, L3 a patching effect on that probe, L4 a direction. Each has a
  held-out test and matched random controls.

Limit: the lesion here is a hook inside a serving program, not the library's lesion channel (an intervention
on the `lesion` channel of a session); the subject does not state it, and the run records state it instead.

fLoc: we used stimuli included in the fLoc functional localizer package (Stigliani et al., 2015; J Neurosci
35(36):12412).
