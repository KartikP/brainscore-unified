"""Extract CLIP per-TR video features for held-out S7 + OOD, aligned to the
held-out stub assemblies. Saves video_features_clip_<split>.npz {X, extracted}.
Subject-independent (held-out TR grid is shared)."""
import sys, time, numpy as np
sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
t0=time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)
import brainscore
BR='/home/ubuntu/.brainio/algonauts2025'
log('load CLIP...')
model = brainscore.load_model('clip-vit-b-32')
model._region_layer_map_dict['IT'] = 'post_layernorm'
for split, bid in [('friends_s7','Algonauts2025-friends-s7-sub01'),
                   ('ood','Algonauts2025-ood-sub01')]:
    log(f'=== {split}: {bid} ===')
    b = brainscore.load_benchmark(bid)
    fss = b._expand_to_per_TR_frames()
    log(f'  {len(fss)} frames')
    feats, fids = b._extract_per_TR_features(model, fss)
    X = b._align_features_to_assembly(feats, fids, fss)
    out=f'{BR}/video_features_clip_{split}.npz'
    np.savez_compressed(out, X=np.asarray(X).astype(np.float32))
    log(f'  saved {out}  X={np.asarray(X).shape}')
log('VIDEO_DONE')
