"""One small causal LM for generation, log probabilities, and neural recording."""
import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brainscore import BrainScoreModel

HF_IDENTIFIER = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
IDENTIFIER = 'qwen2.5-0.5b-instruct'


def get_model() -> 'BrainScoreModel':
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from brainscore.model_helpers._device import select_device
    from brainscore.model_helpers.causal_lm import build_causal_lm_subject

    checkpoint = os.environ.get('BRAINSCORE_QWEN25_05B_PATH')
    if checkpoint:
        source = str(Path(checkpoint).expanduser().resolve())
        settings = {'local_files_only': True}
    else:
        from brainscore.models._downloads import hf_preflight
        source = HF_IDENTIFIER
        settings = {**hf_preflight(IDENTIFIER, HF_IDENTIFIER, 1.0), 'revision': REVISION}
    device = select_device().type
    dtype = torch.float16 if device != 'cpu' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(source, **settings)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype, **settings).to(device).eval()
    provenance = {'checkpoint': HF_IDENTIFIER, 'revision': REVISION, 'device': device,
                  'dtype': str(dtype), 'torch': torch.__version__,
                  'region_layer_map_status': 'provisional last decoder layer; no layer search'}
    if checkpoint:
        digest = hashlib.sha256()
        for weights in sorted(Path(source).glob('*.safetensors')):
            with weights.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
        provenance['weights_sha256'] = digest.hexdigest()
    return build_causal_lm_subject(IDENTIFIER, model, tokenizer,
                                   region_layer_map={'language_system': 'model.layers.23'},
                                   layer_aggregation='last_token', provenance=provenance)
