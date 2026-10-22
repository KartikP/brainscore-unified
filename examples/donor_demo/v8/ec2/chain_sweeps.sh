#!/bin/bash
cd /opt/dlami/nvme/donor2
while ! grep -q "s\] DONE" lesion.log; do sleep 30; done
while ! grep -q SWEEP_DONE frac_3b.log; do sleep 30; done
./sweep_frac.sh 0,1 Qwen/Qwen2.5-VL-7B-Instruct 7b > frac_7b.log 2>&1
./sweep_frac.sh 0,1,2,3 Qwen/Qwen2.5-VL-32B-Instruct 32b > frac_32b.log 2>&1
echo ALL_SWEEPS_DONE > sweeps_done.flag
