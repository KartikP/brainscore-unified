"""An explicit Qwen video subject for this demonstration; existing registrations stay unchanged."""

CHECKPOINT = 'Qwen/Qwen2.5-VL-3B-Instruct'
REVISION = '66285546d2b821cf421d4f5eb2576359d3770cd3'


def build_subject():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from brainscore_core import BrainScoreModel
    from brainscore_core.selection import LayerSelector
    from brainscore.model_helpers.replies import chat_parts
    from brainscore.model_helpers.vlm_video_wrapper import VLMVideoWrapper
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CHECKPOINT, revision=REVISION, local_files_only=True, dtype=torch.float16,
        device_map={'': 'cuda:0'}, attn_implementation='sdpa')
    model.eval()
    processor = AutoProcessor.from_pretrained(CHECKPOINT, revision=REVISION, local_files_only=True)
    # Model-side sampling is part of this subject's identity. Half-second clips
    # need all sixteen frames; the standard 2-fps sampling would miss their motion.
    processor.video_processor.fps = 30
    processor.video_processor.min_frames = 16
    processor.video_processor.max_frames = 16
    processor.video_processor.size = {'shortest_edge': 128 * 28 * 28,
                                      'longest_edge': 256 * 28 * 28}
    video = VLMVideoWrapper(model.model.visual, processor, identifier='qwen25-video-30fps')
    reply, _ = chat_parts(model, processor, max_new_tokens=96)
    subject = BrainScoreModel(
        identifier='qwen2.5-vl-3b-video-30fps-sun-demo',
        extractor={'video': video}, reply=reply,
        region_layer_map={'V1': LayerSelector('blocks.2', modality='video')})
    provenance = {'checkpoint': CHECKPOINT, 'revision': REVISION, 'dtype': 'float16',
                  'device': 'cuda:0', 'attention': 'sdpa', 'video_fps': 30,
                  'video_min_frames': 16, 'video_max_frames': 16,
                  'video_size': processor.video_processor.size, 'max_new_tokens': 96,
                  'region_mapping': {'V1': 'visual.blocks.2'},
                  'mapping_status': 'illustrative reuse of registered mapping, not validated neural alignment'}
    return subject, provenance
