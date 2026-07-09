"""Auto-register — infer the wrapper, recording layers, and a provisional
``region_layer_map`` for an arbitrary model, so a builder can plug a model in
*without* knowing Brain-Score internals.

This is the tool that sits one step **upstream** of the layer-mapping explorer.
The model-builder workflow it enables:

    profile = inspect_model(hf_model, processor)     # what wrapper? which layers?
    print(profile.summary())                         # human-readable report
    model  = auto_register(hf_model, identifier='my-model',
                           preprocessing=my_preproc)  # runnable BrainScoreModel
    # then refine the layer→region map with the layer-mapping explorer:
    #   from brainscore.tools.layer_mapping import sweep_model
    #   sweep_model(wrapper, stimuli, brain, profile.recommendations[0].block_layers)

What it removes from the builder's plate (the friction observed registering
CLIP / Qwen-VL / BLIP-2 / VideoMAE / V-JEPA / Wav2Vec2 by hand):

  1. **Wrapper choice.** Picks among the five activations-model wrappers
     (``PytorchWrapper`` / ``TextWrapper`` / ``VLMVisionWrapper`` /
     ``VideoWrapper`` / ``AudioWrapper``) from the model's class, config, and
     module tree.
  2. **Layer discovery.** Finds the repeated indexed block list
     (``encoder.layers.*`` / ``model.layers.*`` / ``blocks.*`` / …) — the
     natural recording targets — instead of the builder reading the source.
  3. **Sub-module wrapping.** Detects multi-tower models (CLIP, BLIP-2, VLMs)
     whose ``forward()`` needs every modality's input at once, and recommends
     wrapping each tower's sub-module (with layer paths made relative to it).
  4. **Sizing / dtype warnings.** Flags huge feature dims (LayerPCA needed),
     low-precision weights (device-alignment gotcha), and gated weights.

Nothing here forces a layer choice — the provisional map is evenly spaced over
the detected blocks purely so the model is *runnable immediately*; the
layer-mapping explorer is what turns it into a brain-optimal map.

Pure-Python module structure inspection — ``torch`` is imported lazily and only
where a live wrapper is actually constructed.
"""
import dataclasses
import textwrap
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Canonical brain regions, ordered shallow → deep, used only to *space* a
# provisional map across the detected blocks. The layer-mapping explorer
# refines this.
DEFAULT_VISION_REGIONS: Tuple[str, ...] = ('V1', 'V2', 'V4', 'IT')
DEFAULT_LANGUAGE_REGIONS: Tuple[str, ...] = ('language_system',)
DEFAULT_AUDIO_REGIONS: Tuple[str, ...] = ('A1',)


# ── modality → wrapper recommendation ─────────────────────────────────────

# Each modality maps to (wrapper class name, default layer_aggregation,
# the brain regions a provisional map spaces over).
_MODALITY_WRAPPER = {
    'vision':       ('PytorchWrapper',   None,           DEFAULT_VISION_REGIONS),
    'vision_flat':  ('VLMVisionWrapper', 'mean_patches', DEFAULT_VISION_REGIONS),
    'video':        ('VideoWrapper',     None,           DEFAULT_VISION_REGIONS),
    'audio':        ('AudioWrapper',     'mean_time',    DEFAULT_AUDIO_REGIONS),
    'text_causal':  ('TextWrapper',      'last_token',   DEFAULT_LANGUAGE_REGIONS),
    'text_encoder': ('TextWrapper',      'mean_tokens',  DEFAULT_LANGUAGE_REGIONS),
}

# Substring signals (matched case-insensitively against class name + HF
# ``model_type`` + ``architectures``). Checked in this order; first hit wins.
_AUDIO_SIGNALS = ('wav2vec2', 'wavlm', 'hubert', 'whisper', 'w2v', 'seamless',
                  'audio', 'speech', 'mert', 'encodec', 'ast')
_VIDEO_SIGNALS = ('videomae', 'vjepa', 'v_jepa', 'jepa', 'timesformer', 'vivit',
                  'x_clip', 'xclip', 'video')
# VLMs whose vision encoder emits a flattened/variable patch layout
# (concatenated-across-images, grid_thw metadata) — these need VLMVisionWrapper.
_VLM_FLAT_SIGNALS = ('qwen2_5_vl', 'qwen2vl', 'qwen2_vl', 'llava', 'internvl',
                     'idefics', 'fuyu', 'paligemma', 'mllama', 'pixtral',
                     'smolvlm')
_TEXT_CAUSAL_SIGNALS = ('forcausallm', 'gpt', 'llama', 'mistral', 'mixtral',
                        'falcon', 'opt', 'bloom', 'gemma', 'phi', 'mpt',
                        'starcoder', 'pythia', 'olmo')
_TEXT_ENCODER_SIGNALS = ('bert', 'roberta', 'electra', 'distilbert', 'mpnet',
                         'deberta', 'albert', 't5encoder', 'clip_text',
                         'cliptext', 'sentence')
_VISION_SIGNALS = ('vit', 'resnet', 'convnext', 'deit', 'beit', 'swin', 'dino',
                   'regnet', 'efficientnet', 'mobilenet', 'clip_vision',
                   'clipvision', 'image', 'visual', 'vision')


# ── data classes ──────────────────────────────────────────────────────────

@dataclasses.dataclass
class BlockGroup:
    """A run of sibling modules indexed by integer under a common parent.

    e.g. ``encoder.layers.0 … encoder.layers.11`` → prefix ``encoder.layers``,
    indices ``[0..11]``. These are the recording-target candidates.
    """
    prefix: str
    indices: List[int]

    @property
    def paths(self) -> List[str]:
        return [f"{self.prefix}.{i}" for i in self.indices]

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def top(self) -> str:
        """Top-level attribute the group lives under (the tower root)."""
        return self.prefix.split('.')[0]


@dataclasses.dataclass
class WrapperRecommendation:
    """One tower's recommendation: which wrapper, which sub-module, which layers."""
    modality: str                       # 'vision' | 'text_causal' | 'audio' | …
    wrapper: str                        # the activations-model class name
    block_layers: List[str]             # candidate recording layers (relative to
                                        #   submodule_path when that is set)
    submodule_path: Optional[str]       # wrap this sub-module, or None for the full model
    layer_aggregation: Optional[str]    # for text/audio wrappers
    regions: Tuple[str, ...]            # canonical regions a provisional map spaces over
    reason: str

    def provisional_region_layer_map(self) -> Dict[str, str]:
        return space_layers(self.block_layers, self.regions)


@dataclasses.dataclass
class ModelProfile:
    """Everything inferred about a model, plus a human-readable ``summary()``."""
    identifier: str
    n_parameters: int
    dtype: Optional[str]
    recommendations: List[WrapperRecommendation]
    warnings: List[str] = dataclasses.field(default_factory=list)
    notes: List[str] = dataclasses.field(default_factory=list)

    @property
    def is_multimodal(self) -> bool:
        return len(self.recommendations) > 1

    @property
    def primary(self) -> Optional[WrapperRecommendation]:
        """The largest / first tower — the one a single-modality builder cares about."""
        return self.recommendations[0] if self.recommendations else None

    def provisional_region_layer_map(self) -> Dict[str, str]:
        """Provisional map across all towers. Region names are suffixed with the
        tower's modality when two towers would otherwise collide on a region."""
        if not self.recommendations:
            return {}
        if len(self.recommendations) == 1:
            return self.recommendations[0].provisional_region_layer_map()
        out: Dict[str, str] = {}
        for rec in self.recommendations:
            m = rec.provisional_region_layer_map()
            for region, layer in m.items():
                key = region if region not in out else f"{region}_{rec.modality}"
                # prefix the layer with the submodule path so it is addressable
                # from the full model when towers are wrapped separately
                out[key] = layer
        return out

    def required_modalities(self) -> set:
        """Single-modality models hard-require their one modality; multimodal
        models leave it soft (empty) so they degrade to any subset."""
        if len(self.recommendations) != 1:
            return set()
        return {_base_modality(self.recommendations[0].modality)}

    def summary(self) -> str:
        lines = [f"Model: {self.identifier}",
                 f"  parameters: {self.n_parameters:,}"
                 + (f"  dtype: {self.dtype}" if self.dtype else ""),
                 f"  multimodal: {self.is_multimodal}"]
        for i, rec in enumerate(self.recommendations):
            tower = rec.submodule_path or '(full model)'
            lines.append(f"  tower {i}: {rec.modality}  → {rec.wrapper}")
            lines.append(f"    wrap: {tower}")
            agg = f"  aggregation={rec.layer_aggregation}" if rec.layer_aggregation else ""
            lines.append(f"    {len(rec.block_layers)} candidate layers"
                         f" ({rec.block_layers[0]} … {rec.block_layers[-1]}){agg}"
                         if rec.block_layers else "    (no block layers found)")
            lines.append(f"    why: {rec.reason}")
            pmap = rec.provisional_region_layer_map()
            lines.append("    provisional map: "
                         + ", ".join(f"{r}={l}" for r, l in pmap.items()))
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        for n in self.notes:
            lines.append(f"  · {n}")
        return "\n".join(lines)


# ── block detection ────────────────────────────────────────────────────────

def find_block_groups(model: Any, min_size: int = 2) -> List[BlockGroup]:
    """Find every run of integer-indexed sibling modules (the block lists).

    Works on anything exposing ``named_modules()`` (a ``torch.nn.Module``).
    Returned longest-first, so ``find_block_groups(m)[0]`` is the main stack.
    Groups smaller than ``min_size`` (e.g. a 1-element MLP ``Sequential``) are
    dropped.
    """
    if not hasattr(model, 'named_modules'):
        raise TypeError(
            "auto_register inspects torch modules; the object has no "
            "named_modules(). Pass an nn.Module (e.g. a HuggingFace model).")
    groups: Dict[str, set] = {}
    for name, _ in model.named_modules():
        if not name:
            continue
        head, _sep, tail = name.rpartition('.')
        if tail.isdigit() and head:
            groups.setdefault(head, set()).add(int(tail))
    result = [BlockGroup(prefix, sorted(idxs))
              for prefix, idxs in groups.items() if len(idxs) >= min_size]
    result.sort(key=lambda g: (-g.size, g.prefix))
    return result


def _largest_group_per_tower(groups: List[BlockGroup]) -> Dict[str, BlockGroup]:
    """Keep, per top-level attribute, the largest block group (the tower's main
    stack — drops the tower's small MLP/attention sub-groups)."""
    per_top: Dict[str, BlockGroup] = {}
    for g in groups:                       # already sorted longest-first
        if g.top not in per_top:
            per_top[g.top] = g
    return per_top


# ── modality / wrapper classification ──────────────────────────────────────

def _model_meta(model: Any) -> Dict[str, Any]:
    cfg = getattr(model, 'config', None)
    arch = list(getattr(cfg, 'architectures', None) or []) if cfg is not None else []
    return {
        'class': type(model).__name__,
        'model_type': (getattr(cfg, 'model_type', '') or '') if cfg is not None else '',
        'architectures': arch,
        'is_decoder': bool(getattr(cfg, 'is_decoder', False)) if cfg is not None else False,
        'is_encoder_decoder': bool(getattr(cfg, 'is_encoder_decoder', False)) if cfg is not None else False,
    }


def _meta_signal(meta: Dict[str, Any]) -> str:
    return " ".join([meta['class'], meta['model_type'],
                     " ".join(meta['architectures'])]).lower()


def _base_modality(modality: str) -> str:
    """Collapse the wrapper-level modality to the dispatch modality used by
    ``BrainScoreModel.preprocessors`` keys and ``required_modalities``."""
    return {'vision_flat': 'vision', 'text_causal': 'text',
            'text_encoder': 'text'}.get(modality, modality)


def _classify_global(meta: Dict[str, Any]) -> Tuple[str, str]:
    """Classify a single-tower model from its class / config. Returns
    (modality, reason)."""
    sig = _meta_signal(meta)
    if any(s in sig for s in _AUDIO_SIGNALS):
        return 'audio', f"audio backbone signal in {meta['model_type'] or meta['class']!r}"
    if any(s in sig for s in _VIDEO_SIGNALS):
        return 'video', f"video backbone signal in {meta['model_type'] or meta['class']!r}"
    if any(s in sig for s in _VLM_FLAT_SIGNALS):
        return 'vision_flat', f"flattened-patch VLM signal in {meta['model_type'] or meta['class']!r}"
    if meta['is_decoder'] or any(s in sig for s in _TEXT_CAUSAL_SIGNALS):
        return 'text_causal', "causal-LM signal (is_decoder / ForCausalLM / known LM family)"
    if any(s in sig for s in _TEXT_ENCODER_SIGNALS):
        return 'text_encoder', f"text-encoder signal in {meta['model_type'] or meta['class']!r}"
    if any(s in sig for s in _VISION_SIGNALS):
        return 'vision', f"vision backbone signal in {meta['model_type'] or meta['class']!r}"
    # default: assume a standard image model wrapped by PytorchWrapper
    return 'vision', "no decisive signal; defaulting to vision/PytorchWrapper"


def _classify_tower(top: str, submodule: Any, meta: Dict[str, Any]) -> Tuple[str, str]:
    """Classify one tower of a multi-tower model. The tower's attribute name
    (``top``) is the strongest signal; fall back to the sub-module's class then
    the global model meta."""
    sig_top = top.lower()
    sub_cls = type(submodule).__name__.lower() if submodule is not None else ''
    sig = f"{sig_top} {sub_cls}"
    is_flat = any(s in _meta_signal(meta) for s in _VLM_FLAT_SIGNALS) \
        or any(s in sub_cls for s in _VLM_FLAT_SIGNALS)

    if any(k in sig for k in ('audio', 'wav', 'speech', 'sound')):
        return 'audio', f"audio tower {top!r}"
    if 'video' in sig:
        return 'video', f"video tower {top!r}"
    if any(k in sig for k in ('vis', 'visual', 'image', 'vit', 'patch', 'pixel')):
        return ('vision_flat' if is_flat else 'vision'), f"vision tower {top!r}"
    if 'text' in sig:
        return 'text_encoder', f"text-encoder tower {top!r}"
    if any(k in sig for k in ('lang', 'llm', 'decoder')) or sig_top in ('model', 'lm'):
        return 'text_causal', f"language/decoder tower {top!r}"
    # ambiguous tower name → defer to the global classifier
    mod, why = _classify_global(meta)
    return mod, f"tower {top!r}: {why}"


# ── provisional region→layer spacing ────────────────────────────────────────

def space_layers(block_paths: Sequence[str],
                 regions: Sequence[str]) -> Dict[str, str]:
    """Evenly space ``regions`` (shallow→deep) across ``block_paths``.

    Purely a *runnable starting point* — the layer-mapping explorer is what
    finds the brain-optimal assignment. The deepest region is placed at the
    last block; intermediate regions at evenly spaced depths.
    """
    n = len(block_paths)
    m = len(regions)
    if n == 0 or m == 0:
        return {}
    if m == 1:
        return {regions[0]: block_paths[-1]}
    positions = [round(i * (n - 1) / (m - 1)) for i in range(m)]
    return {regions[i]: block_paths[positions[i]] for i in range(m)}


# ── top-level inspection ─────────────────────────────────────────────────────

def inspect_model(model: Any, processor: Any = None,
                  identifier: Optional[str] = None) -> ModelProfile:
    """Inspect a model and return a :class:`ModelProfile`: recommended
    wrapper(s), candidate recording layers, a provisional map, and warnings.

    Does **no** forward pass and downloads nothing — pure structure inspection.
    """
    meta = _model_meta(model)
    ident = identifier or meta['model_type'] or meta['class']
    groups = find_block_groups(model)
    warnings: List[str] = []
    notes: List[str] = []

    # parameter count + dtype (best-effort; never fail inspection over it)
    n_params, dtype = 0, None
    try:
        params = list(model.parameters())
        n_params = sum(p.numel() for p in params)
        if params:
            dtype = str(params[0].dtype).replace('torch.', '')
    except Exception:
        pass

    recommendations: List[WrapperRecommendation] = []

    if not groups:
        warnings.append(
            "No integer-indexed block list found. Specify recording layers "
            "manually (inspect model.named_modules()).")
    else:
        per_tower = _largest_group_per_tower(groups)
        # A model is only genuinely MULTI-tower (multimodal) when its towers span
        # more than one distinct modality. Same-modality block stacks (a ResNet's
        # layer1..layer4, or repeated encoder stages) are ONE backbone, not many
        # towers — classifying them as multimodal broke plain CNNs.
        tower_modality = {
            top: _classify_tower(top, getattr(model, top, None), meta)[0]
            for top in per_tower
        }
        distinct = {_base_modality(m) for m in tower_modality.values()}
        multi = len(distinct) > 1

        if multi:
            ordered = sorted(per_tower.items(), key=lambda kv: -kv[1].size)
            for top, group in ordered:
                modality, reason = _classify_tower(
                    top, getattr(model, top, None), meta)
                rel = [p[len(top) + 1:] if p.startswith(top + '.') else p
                       for p in group.paths]
                wrapper, agg, regions = _MODALITY_WRAPPER[modality]
                recommendations.append(WrapperRecommendation(
                    modality=modality, wrapper=wrapper, block_layers=rel,
                    submodule_path=top, layer_aggregation=agg,
                    regions=regions, reason=reason))
            notes.append(
                "Multi-tower model: each tower is wrapped separately because "
                "the full forward() typically requires every modality's input "
                "at once (CLIP / BLIP-2 / VLM pattern). Layer paths are "
                "relative to each tower's sub-module.")
        else:
            # single-modality model: one recommendation over the largest stack
            modality, reason = _classify_global(meta)
            wrapper, agg, regions = _MODALITY_WRAPPER[modality]
            largest = max(per_tower.values(), key=lambda g: g.size)
            recommendations.append(WrapperRecommendation(
                modality=modality, wrapper=wrapper, block_layers=largest.paths,
                submodule_path=None, layer_aggregation=agg,
                regions=regions, reason=reason))

    # sizing / precision / gating warnings
    if n_params >= 1_000_000_000:
        warnings.append(
            f"Large model (~{n_params/1e9:.1f}B params). Flattened ViT/encoder "
            "features can exceed sklearn's regressor budget — use "
            "LayerPCA(n_components=1000) or spatial mean-pooling at scoring.")
    if dtype in ('float16', 'bfloat16'):
        warnings.append(
            f"Weights are {dtype}. Move every wrapped sub-module to the same "
            "device explicitly (multi-component models need a final .to(device) "
            "for end-to-end generate()).")
    if any(r.modality == 'video' for r in recommendations):
        notes.append(
            "Video backbones usually interleave time and space in the hook "
            "output — set VideoWrapper(post_hook_fn=...) to reshape to "
            "(B, T, S, H) → mean-over-space → (B, T, H).")
    if any(r.modality == 'vision_flat' for r in recommendations):
        notes.append(
            "Flattened-patch VLM vision: VLMVisionWrapper needs "
            "patch_count_fn + forward_kwargs_map (e.g. grid_thw) to segment "
            "patches back to images. See qwen25_vl_3b for a worked example.")

    return ModelProfile(identifier=ident, n_parameters=n_params, dtype=dtype,
                        recommendations=recommendations,
                        warnings=warnings, notes=notes)


# ── registration scaffold (emit a real plugin skeleton) ──────────────────────

def scaffold_registration(profile: ModelProfile, hf_id: Optional[str] = None,
                          registry_key: Optional[str] = None) -> str:
    """Emit a ``model.py`` registration skeleton matching the existing plugin
    pattern (CLIP / gpt2). The builder fills in the loader + preprocessing and
    refines the layer map with the layer-mapping explorer.
    """
    key = registry_key or profile.identifier
    hf = hf_id or '<huggingface-id>'
    wrappers = sorted({r.wrapper for r in profile.recommendations})
    region_map = profile.provisional_region_layer_map()
    map_lines = "\n".join(f"    {k!r}: {v!r}," for k, v in region_map.items())
    primary = profile.primary

    imports = ["from brainscore import model_registry",
               "from brainscore_core.model_interface import BrainScoreModel"]
    if 'PytorchWrapper' in wrappers:
        imports.append("from brainscore_vision.model_helpers.activations.pytorch "
                       "import PytorchWrapper")
    for w in wrappers:
        if w == 'PytorchWrapper':
            continue
        snake = {'TextWrapper': 'text_wrapper', 'VLMVisionWrapper': 'vlm_vision_wrapper',
                 'VideoWrapper': 'video_wrapper', 'AudioWrapper': 'audio_wrapper'}[w]
        imports.append(f"from brainscore.model_helpers.{snake} import {w}")

    req = profile.required_modalities()
    req_line = (f"        required_modalities={req!r},\n" if req else "")
    agg = primary.layer_aggregation if primary else None
    agg_hint = (f"\n    # wrapper aggregation: layer_aggregation={agg!r}"
                if agg else "")

    warns = "\n".join(f"#   ⚠ {w}" for w in profile.warnings)
    warns_block = (f"\n# Auto-register warnings:\n{warns}\n" if warns else "")

    return textwrap.dedent(f'''\
"""Auto-generated registration scaffold for {key!r}.

Generated by brainscore.tools.auto_register. Fill in the model loader +
preprocessing, then refine REGION_LAYER_MAP with the layer-mapping explorer:

    from brainscore.tools.layer_mapping import sweep_model
    result, approaches, selector = sweep_model(
        wrapper, stimulus_set, brain_target, brain_stimulus_ids,
        layers={primary.block_layers if primary else []})
    print(result.best_layer)   # → assign to the region you are mapping
"""
{chr(10).join(imports)}
{warns_block}
# Provisional map (evenly spaced over detected blocks — REFINE with the
# layer-mapping explorer; this is only a runnable starting point).{agg_hint}
REGION_LAYER_MAP = {{
{map_lines}
}}


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == {key!r}
    # TODO: load weights + processor, e.g.
    #   from transformers import AutoModel, AutoProcessor
    #   hf_model = AutoModel.from_pretrained({hf!r})
    #   processor = AutoProcessor.from_pretrained({hf!r})
    raise NotImplementedError("fill in the loader for {key!r}")


model_registry[{key!r}] = lambda: get_model({key!r})
''')


# ── live construction (vision + text towers) ─────────────────────────────────

def auto_register(model: Any,
                  identifier: str,
                  preprocessing: Optional[Callable] = None,
                  tokenizer: Any = None,
                  processor: Any = None,
                  regions: Optional[Sequence[str]] = None,
                  region_layer_map: Optional[Dict[str, str]] = None,
                  visual_degrees: int = 8,
                  **brainscore_model_kwargs) -> "Any":
    """Build a runnable :class:`BrainScoreModel` from a single-tower model.

    Supports the two towers that need no processor-specific configuration to
    wire automatically — **vision** (``PytorchWrapper``, needs ``preprocessing``)
    and **text** (``TextWrapper``, needs ``tokenizer``). For audio / video /
    flattened-patch VLM towers, or multi-tower models, call
    :func:`inspect_model` + :func:`scaffold_registration` and fill in the
    wrapper config (those wrappers need processor-specific args this helper
    cannot guess).

    :param region_layer_map: override the provisional map. When omitted, an
        evenly-spaced provisional map over the detected blocks is used — runnable
        immediately, refine later with the layer-mapping explorer.
    """
    from brainscore_core.model_interface import BrainScoreModel

    profile = inspect_model(model, processor=processor, identifier=identifier)
    rec = profile.primary
    if rec is None:
        raise ValueError(
            f"Could not find recording layers in {identifier!r}. Inspect "
            "model.named_modules() and register manually.")
    if profile.is_multimodal:
        raise NotImplementedError(
            f"{identifier!r} is multi-tower ({[r.modality for r in profile.recommendations]}). "
            "Use inspect_model() + scaffold_registration() and wire each tower's "
            "wrapper by hand — auto_register builds single-tower models only.")

    if region_layer_map is None:
        regs = tuple(regions) if regions else rec.regions
        region_layer_map = space_layers(rec.block_layers, regs)

    base_modality = _base_modality(rec.modality)

    if rec.modality in ('vision',):
        if preprocessing is None:
            raise ValueError(
                "vision model needs a `preprocessing` callable "
                "(image_filepaths -> (B,C,H,W) array/tensor).")
        from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
        activations_model = PytorchWrapper(
            identifier=identifier, model=model, preprocessing=preprocessing)
        preprocessors = {'vision': preprocessing}
        return BrainScoreModel(
            identifier=identifier, model=model,
            region_layer_map=region_layer_map, preprocessors=preprocessors,
            activations_model=activations_model, visual_degrees=visual_degrees,
            required_modalities={base_modality}, **brainscore_model_kwargs)

    if rec.modality in ('text_causal', 'text_encoder'):
        if tokenizer is None:
            raise ValueError("text model needs a `tokenizer`.")
        from brainscore.model_helpers.text_wrapper import TextWrapper
        text_wrapper = TextWrapper(
            model=model, tokenizer=tokenizer, identifier=identifier,
            layer_aggregation=rec.layer_aggregation or 'last_token')
        return BrainScoreModel(
            identifier=identifier, model=model,
            region_layer_map=region_layer_map,
            preprocessors={'text': text_wrapper},
            required_modalities={base_modality}, **brainscore_model_kwargs)

    raise NotImplementedError(
        f"auto_register cannot yet auto-wire a {rec.modality!r} tower "
        f"({rec.wrapper}). Use inspect_model() + scaffold_registration(); the "
        f"wrapper needs processor-specific args (see {rec.wrapper} docstring).")
