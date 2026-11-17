"""
BLIP-2 OPT-2.7B registered as a BrainScoreModel — second VLM with a different
architecture from CLIP and Qwen.

Architecture: ViT-G/14 vision encoder + Q-Former + OPT-2.7B causal LM. This
is a third VLM topology (encoder + adapter + decoder), distinct from CLIP's
dual-encoder pair and Qwen's flattened-patch native VLM.

Vision input is standard (batch, C, H, W) — wraps cleanly with PytorchWrapper,
unlike Qwen which needed VLMVisionWrapper. This validates that the unified
interface scales across VLM families with very different vision input layouts:
- CLIP: dual encoder, vision_model.encoder.layers.{N}
- Qwen: flattened patches, VLMVisionWrapper
- BLIP-2: standard ViT, PytorchWrapper

Layer counts:
  vision_model.encoder.layers.{0-38}             — 39 ViT-G blocks
  language_model.model.decoder.layers.{0-31}     — 32 OPT decoder layers

Ways to answer. Writing: the ``answer`` part writes a plain choice (ROAR, the old
label tasks) as before; the ``reply`` part writes a choice with task settings
(``verbatim``, ``present``, options per row) and a written answer, under the
task's generation settings. Likelihood: the ``likelihood`` part gives the log
probability of each option. Readout: a readout fitted on the IT layer gives the
probabilities, the reaction time and the rating; the registration keeps ratings
with the readout. BLIP-2 OPT is not instruction-tuned: its written answers on
ROAR are at the floor (the chance level), so its writing columns are honest
columns, not a claim that it follows instructions.

BLIP-2 OPT has no chat template. Both new parts show a trial in BLIP-2's own
prompt form: the picture first (at most one), then the stimulus text, then
"Question: <instruction> <the task's directive> Answer:". Under a ``verbatim``
choice the text is shown exactly as given, with no question and no answer cue;
the likelihood then scores ``f"{text} {option}"`` with no start token (after the
picture's query tokens when the trial has a picture), as EWoK's ``score`` does.
"""

import functools
import re
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from brainscore.model_helpers.replies import _generation_settings, _plain_text_scores, segment_log_probabilities
from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore import BrainScoreModel
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.responses import Choice, Prompt, Text
from brainscore_vision.model_helpers.activations.pca import LayerPCA
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_core.hf_compat import pin_image_processor
from brainscore.models._shared import _load_preprocess_images


REGION_LAYER_MAP = {
    # Vision regions — relative to vision_model (PytorchWrapper root)
    'V1': 'encoder.layers.4',
    'V2': 'encoder.layers.10',
    'V4': 'encoder.layers.20',
    'IT': 'encoder.layers.34',
    # Language region — relative to language_model.model.decoder (TextWrapper root)
    'language_system': 'layers.28',
}


def _make_answer(blip_model, blip_processor, max_new_tokens: int = 8):
    """The ``answer`` part: BLIP-2 completes a plain choice.

    BLIP-2 OPT-2.7B is not instruction-tuned. The prompt is its own
    question-and-answer form, "Question: {instruction} Choose from: {labels}. Answer:",
    and the model's completion is returned for ``BrainScoreModel`` to parse.
    """
    from brainscore.model_helpers.api_behavioral import _row_parts

    def generate(stimulus_row, instruction: str, label_set: List[str]) -> str:
        image_column, text_column = _row_parts(stimulus_row, 'BLIP-2 answer')
        # BLIP-2's own "Question: ... Answer:" form; with no instruction there is
        # no question. The row's text comes first.
        prompt = ((f"Question: {instruction} " if instruction else '')
                  + f"Choose from: {', '.join(label_set)}. Answer:")
        if text_column:
            prompt = f"{stimulus_row[text_column]}\n\n{prompt}"
        image = stimulus_row[image_column] if image_column else None
        return _write(blip_model, blip_processor, prompt, image,
                      dict(max_new_tokens=max_new_tokens, do_sample=False))

    return generate


def _on_device(blip_model, inputs) -> dict:
    """The processor's tensors on the model's device; floating tensors in the model's dtype."""
    # Explicit per-tensor device move. BatchFeature.to(device) does not
    # reliably move all tensors on MPS; BLIP-2's generate() accesses
    # input_ids inside get_input_embeddings() and needs it on the same
    # device as the embedding weights.
    device = next(blip_model.parameters()).device
    model_dtype = next(blip_model.parameters()).dtype
    return {
        k: (v.to(device, dtype=model_dtype) if torch.is_tensor(v) and v.is_floating_point()
            else (v.to(device) if torch.is_tensor(v) else v))
        for k, v in inputs.items()
    }


def _write(blip_model, blip_processor, prompt: str, image_path, settings: dict) -> str:
    """BLIP-2's generated text for ``prompt`` and at most one picture, under ``settings``."""
    if image_path is not None:
        image = Image.open(image_path).convert('RGB')
        inputs = blip_processor(text=prompt, images=image, return_tensors='pt')
        run = blip_model.generate
    else:
        # No picture: BLIP-2's own language model reads the text alone.
        inputs = blip_processor.tokenizer(prompt, return_tensors='pt')
        run = blip_model.language_model.generate
    inputs = _on_device(blip_model, inputs)
    with torch.no_grad():
        generated = run(**inputs, **settings)

    decoded = blip_processor.tokenizer.decode(generated[0], skip_special_tokens=True)
    # The generation can return the prompt before the reply: only that echo is
    # removed. The rest is the generated text, exactly; BrainScoreModel parses it.
    echoed = decoded.lstrip()
    return echoed[len(prompt):] if echoed.startswith(prompt) else decoded


def _shown(prompt) -> Tuple[Optional[str], str]:
    """One trial in BLIP-2's own prompt form: (the picture's path or None, the text).

    Under a ``verbatim`` choice the text parts are shown exactly as given. Otherwise the
    stimulus text comes first and the instruction and the task's directive (the options
    or the scale) form the question: "Question: ... Answer:".
    """
    kinds = [kind for kind, _ in prompt.parts]
    if set(kinds) - {'text', 'image'} or kinds.count('image') > 1:
        raise ExecutionNotFeasibleError(f'BLIP-2 is shown text and at most one picture; got {kinds}')
    if prompt.history:
        raise ExecutionNotFeasibleError('BLIP-2 keeps no earlier trials; it cannot be shown them')
    image = next((value for kind, value in prompt.parts if kind == 'image'), None)
    names = prompt.names or (None,) * len(prompt.parts)
    texts = [(name, value) for (kind, value), name in zip(prompt.parts, names) if kind == 'text']
    if isinstance(prompt.response, Choice) and prompt.response.verbatim:
        return image, '\n\n'.join(value for _, value in texts)
    question = ' '.join(value for name, value in texts if name in ('instruction', 'directive'))
    shown = [value for name, value in texts if name not in ('instruction', 'directive')]
    return image, '\n\n'.join(shown + [f'Question: {question} Answer:' if question else 'Answer:'])


def _make_parts(blip_model, blip_processor, max_new_tokens: int = 64):
    """The ``reply`` and ``likelihood`` parts of BLIP-2, in its own prompt form (``_shown``)."""
    tokenizer = blip_processor.tokenizer

    def reply(prompt) -> str:
        """Greedy and at most ``max_new_tokens`` long unless the task's Text response says otherwise."""
        image, text = _shown(prompt)
        if isinstance(prompt.response, Text) and prompt.response.seed is not None:
            torch.manual_seed(prompt.response.seed)
        return _write(blip_model, blip_processor, text, image,
                      _generation_settings(prompt, max_new_tokens)).strip()

    def likelihood(prompt, candidates) -> dict:
        """The summed log probability of the tokens of each option in ``f"{text} {option}"``."""
        image, text = _shown(prompt)
        if image is None:
            # No picture: BLIP-2's own language model scores the text alone, as plain text.
            return _plain_text_scores(blip_model.language_model, tokenizer,
                                      Prompt(parts=(('text', text),)), candidates)
        # The picture's query tokens, then the text with no start token.
        inputs = _on_device(blip_model, blip_processor(
            text=text, images=Image.open(image).convert('RGB'), return_tensors='pt',
            add_special_tokens=False))
        prompt_ids, length = inputs['input_ids'], inputs['input_ids'].shape[1]
        context = tokenizer(text, add_special_tokens=False)['input_ids']
        with torch.no_grad():
            blip_model(**inputs)     # the prompt alone first: the pass a recording keeps
        scores = {}
        for candidate in candidates:
            # The option's tokens are those after the text's own tokens in f"{text} {option}".
            ids = tokenizer(f'{text} {candidate}', add_special_tokens=False)['input_ids'][len(context):]
            if not ids:
                raise ValueError(f'option {candidate!r} has no tokens to score')
            ids = torch.tensor([ids], device=prompt_ids.device)
            extended = dict(inputs, input_ids=torch.cat([prompt_ids, ids], dim=1),
                            attention_mask=torch.cat([inputs['attention_mask'], torch.ones_like(ids)], dim=1))
            with torch.no_grad():
                logits = blip_model(**extended).logits[0, length - 1:length - 1 + ids.shape[1]]
            scores[candidate] = float(logits.float().log_softmax(-1).gather(-1, ids[0, :, None]).sum())
        return scores

    def segments(prompt):
        # A Segments task: BLIP-2's own language model scores the text alone.
        return segment_log_probabilities(blip_model.language_model, tokenizer, prompt.response.segments)

    return reply, {'continuation': likelihood, 'segment': segments}


def load_blip2(identifier: str):
    """BLIP-2 OPT-2.7B in float16 and its processor (image part pinned): the one load of the family.

    ``identifier`` names the registration in the download check.
    """
    from brainscore.models._downloads import hf_preflight
    download = hf_preflight(identifier, 'Salesforce/blip2-opt-2.7b', 15.5)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        'Salesforce/blip2-opt-2.7b',
        torch_dtype=torch.float16,
        **download,
    )
    # Pin the image-processor implementation: transformers 5 rebinds the class
    # names, moving the default from PIL to torchvision and shifting pixels.
    blip_processor = AutoProcessor.from_pretrained('Salesforce/blip2-opt-2.7b', **download)
    blip_processor = pin_image_processor(blip_processor, 'Salesforce/blip2-opt-2.7b', **download)
    return blip_model, blip_processor


def picture_extractor(blip_model, blip_processor, identifier: str, backbone_id=None) -> PytorchWrapper:
    """The picture extractor of every BLIP-2 registration: the vision model, the image loader, LayerPCA(1000).

    ``identifier`` labels the activations; ``backbone_id`` names the stored
    path of the legacy wrapper (its ``identifier`` when None).
    """
    # BLIP-2's vision_model can be called standalone (pixel_values ->
    # last_hidden_state), so the full model wrapper isn't needed.
    vision_wrapper = PytorchWrapper(
        identifier=identifier,
        backbone_id=backbone_id,
        model=blip_model.vision_model,
        preprocessing=functools.partial(_load_preprocess_images, processor=blip_processor),
    )
    # ViT-G hook outputs (batch, 257, 1408) flatten to 361K features per image.
    # That's too many for sklearn PLS at scoring time. LayerPCA(1000) is the
    # convention used by LayerSelection / Layer Mapping Explorer — it fits a
    # PCA on 1000 ImageNet validation images and reduces hook output to 1000D.
    # First-time overhead is ~1-2 min per layer; result cached to disk.
    LayerPCA.hook(vision_wrapper, n_components=1000)
    return vision_wrapper


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'blip2-opt-2.7b'
    blip_model, blip_processor = load_blip2(identifier)
    vision_wrapper = picture_extractor(blip_model, blip_processor, identifier)

    # Text: TextWrapper wraps the OPT decoder. Causal LM → last aggregation.
    # Wrap language_model.model.decoder so layer paths are 'layers.{N}'.
    text_wrapper = TextWrapper(
        model=blip_model.language_model.model.decoder,
        tokenizer=blip_processor.tokenizer,
        identifier=f'{identifier}-text',
    )

    # BLIP-2 is composed of vision_model + qformer + language_projection +
    # language_model. Each wrapper moves only its sub-module to device, so
    # qformer/language_projection can end up on a different device than
    # vision_model and language_model. Force-align to one device for
    # end-to-end generate() to work.
    from brainscore.model_helpers._device import select_device
    blip_model = blip_model.to(select_device())

    # The answer part: BLIP-2's OPT decoder completes a plain choice.
    answer = _make_answer(blip_model, blip_processor)
    # Writing for choices with task settings and for written answers, and the likelihood of
    # the options, in BLIP-2's own prompt form.
    reply, likelihood = _make_parts(blip_model, blip_processor)

    return BrainScoreModel(
        identifier=identifier,
        model=blip_model,
        region_layer_map=REGION_LAYER_MAP,
        extractor={
            'vision': vision_wrapper,
            'text': text_wrapper,
        },
        answer=answer,
        reply=reply,
        likelihood=likelihood,
        # Readout path also available — PCA-1000 features from 'IT' layer.
        behavioral_readout_layer='encoder.layers.34',
        # The reply and the readout both give the rating natively; the registration keeps ratings
        # with the readout, the part that gave them before the reply.
        # Bare behavior is declared because an answer part is wired; it carries the requested
        # format's answer (the answer part's choice, the readout's probabilities).
        capability_config={'behavior-sessions': {'behavior:rating': 'readout'}},
    )
