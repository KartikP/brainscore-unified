import torch as _t
if not hasattr(_t,"float8_e8m0fnu"): _t.float8_e8m0fnu=_t.bfloat16   # ponytail shim: transformers5.10 wants torch>=2.7 dtype, unused here
import os, json, glob
import numpy as np
os.environ["DATAPATH"]="/tmp/tribe_data"; os.environ["SAVEPATH"]="/tmp/tribe_save"
res={}
# (A) DEFINITIVE Llama unblock: load the ungated mirror directly
try:
    from transformers import AutoModelForCausalLM
    llama = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-3B", torch_dtype=_t.bfloat16)
    res["llama_mirror_load_ok"]=True
    res["llama_params_B"]=round(sum(p.numel() for p in llama.parameters())/1e9,2)
    print("LLAMA_MIRROR_OK params(B)=", res["llama_params_B"], flush=True)
    del llama; _t.cuda.empty_cache()
except Exception as e:
    res["llama_mirror_load_ok"]=False; res["llama_err"]=f"{type(e).__name__}: {e}"
    print("LLAMA_MIRROR_FAIL", res["llama_err"], flush=True)
# (B) TRIBEv2 predict on Lahner BOLDMoments clips -> (20484,) per clip
from tribev2.demo_utils import TribeModel
m = TribeModel.from_pretrained("facebook/tribev2", cache_folder="/tmp/tribe_cache")
clips = sorted(glob.glob("/tmp/lahner_nv/stimulus_*.mp4"))[:120]
print(f"predicting {len(clips)} Lahner clips", flush=True)
preds={}
for k,c in enumerate(clips):
    sid=os.path.basename(c)[:-4]   # stimulus_0520
    try:
        ev=m.get_events_dataframe(video_path=c)
        p,_=m.predict(ev, verbose=False)
        if p.shape[0]>0: preds[sid]=p.mean(axis=0).astype("float32")
    except Exception as e:
        print(f"  clip {sid} fail: {type(e).__name__}: {e}", flush=True)
    if (k+1)%20==0: print(f"  {k+1}/{len(clips)}", flush=True)
np.savez("/tmp/tribe_lahner_preds.npz", **preds)
res["n_clips_predicted"]=len(preds)
res["pred_dim"]=int(next(iter(preds.values())).shape[0]) if preds else 0
json.dump(res, open("/tmp/tribe_lahner_step1.json","w"), indent=2)
print("STEP1", json.dumps(res), flush=True)
