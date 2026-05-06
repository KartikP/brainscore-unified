"""Re-extract CLIP video features using the existing benchmark pipeline,
save to disk as a single .npz so the multimodal scorer can reuse without
re-running.

Output: /home/ubuntu/.brainio/algonauts2025/video_features_clip.npz
"""
import sys
import time
import numpy as np

sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')

t0 = time.time()
def log(msg):
    print(f'[{time.time()-t0:7.1f}s] {msg}', flush=True)

log('imports...')
import brainscore

log('loading model + benchmark...')
model = brainscore.load_model('clip-vit-b-32')
model._region_layer_map_dict['IT'] = 'post_layernorm'
b = brainscore.load_benchmark('Algonauts2025-friends-sub01')

log('expanding stim_set + extracting features (cached)...')
frame_stim_set = b._expand_to_per_TR_frames()
features, frame_ids = b._extract_per_TR_features(model, frame_stim_set)
log(f'features: {features.shape}')

log('aligning to assembly...')
X = b._align_features_to_assembly(features, frame_ids, frame_stim_set)
log(f'aligned X: {X.shape}')

out_path = '/home/ubuntu/.brainio/algonauts2025/video_features_clip.npz'
np.savez_compressed(out_path, X=X.astype(np.float32))
log(f'saved {out_path}')
