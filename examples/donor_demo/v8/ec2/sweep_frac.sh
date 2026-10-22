#!/bin/bash
export HF_HUB_OFFLINE=1 RESULTCACHING_HOME=/opt/dlami/nvme/donor2/result_caching CUDA_VISIBLE_DEVICES=$1
for f in 0.05 0.10 0.40 0.50; do
  ~/miniconda3/envs/brainscore-unified/bin/python donor_lesion_protocol.py --models $2 --temps 0 --seeds 0 --frac $f --out /opt/dlami/nvme/donor2/out_frac/$3_f$f
done
echo SWEEP_DONE
