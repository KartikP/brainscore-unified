#!/bin/bash
# v8 assembly: lesion analysis -> notes -> two-slide deck -> rasters -> MP4/GIF/PNG -> docs. Usage: OUT_ROOT=<root> finalize_v8.sh <lesion_data_dir>
set -euo pipefail
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/analytics-2026/bin/python
SCI=/opt/anaconda3/envs/brainscore-unified-v2/bin/python
ROOT=${OUT_ROOT:?set OUT_ROOT}
LES=${1:?lesion data dir}; DOSE=${2:-}
P=$ROOT/presentation; E=$ROOT/experiment; V=$ROOT/visualization-data
mkdir -p $P $E $V/lesion_protocol $V/lahner2024_vjepa1_wav2vec2/replay/friends $V/lahner2024_vjepa1_wav2vec2/replay/boldmoments_per_scan probe
TABLE="/Users/kartik/Brain-Score Unified/unified/scripts/ec2_results_2026_07_31/task7d_cis.json"
ALGO=../work/stage_algo/results_algonauts.json
$SCI analyze_lesion.py --data $LES --out $LES/analysis ${DOSE:+--dose_dir $DOSE} >/dev/null
$SCI make_lesion_notes.py $LES/analysis/lesion_summary.json $LES/analysis/lesion_notes.txt
[ -f lesion_interpretation.txt ] && { echo; cat lesion_interpretation.txt; } >> $LES/analysis/lesion_notes.txt
$PY make_notes_v8.py ../work/stage_full $ALGO >/dev/null
BIG=$($SCI -c "import json;S=json.load(open('$LES/analysis/lesion_summary.json'));print(max(S['models'],key=lambda m:sum(S['models'][m]['population']['targeted_minus_random_deficit']['per_agent'])))")
$PY build_poster_v8.py --data ../work/stage_full --replay replay_main/replay.mp4 --poster_frame ../work/replay/frames/f00000.png \
    --out $P/Brain-Score-TV-Poster-v8.pptx --table_json "$TABLE" --algonauts $ALGO \
    --lesion_summary $LES/analysis/lesion_summary.json --raster_png $LES/analysis/raster_clean_$BIG.png --lesion_notes $LES/analysis/lesion_notes.txt
$PY render_slide_png.py $P/Brain-Score-TV-Poster-v8.pptx probe/v8_hole.png --hole >/dev/null
$PY render_slide_png.py $P/Brain-Score-TV-Poster-v8.pptx probe/v8_slide2.png --slide 2 >/dev/null
bash compose_outputs_v8.sh probe/v8_hole.png probe/v8_hole.layout.json replay_main/replay.mp4 probe/v8_slide2.png $P/Brain-Score-TV-Poster-v8 25
{ cat ../work/stage_full/poster_notes.txt; echo; echo "===== SLIDE 2 ====="; echo; cat $LES/analysis/lesion_notes.txt; } > $P/Brain-Score-TV-Poster-v8-speaker-notes.txt
$SCI write_results_doc_v8.py $E/02_results_and_methods.md replay_friends/display_params.json out_tr/results_timeresolved.json $LES/analysis/lesion_summary.json $E/02_results_and_methods.md
# data package
cp -r replay_friends/replay_friends.mp4 replay_friends/replay_friends_with_sound.mp4 replay_friends/frame_map.csv replay_friends/display_params.json friends_run_candidates_sub01.json $V/lahner2024_vjepa1_wav2vec2/replay/friends/ 2>/dev/null || true
cp -r friends_demo $V/lahner2024_vjepa1_wav2vec2/replay/friends/ 
cp replay_main/replay.mp4 $V/lahner2024_vjepa1_wav2vec2/replay/replay_v8_combined.mp4
cp -r replay_tr/replay_tr.mp4 replay_tr/frame_map.csv replay_tr/display_params.json $V/lahner2024_vjepa1_wav2vec2/replay/boldmoments_per_scan/
cp -r out_tr $V/lahner2024_vjepa1_wav2vec2/replay/boldmoments_per_scan/encoding_outputs
rsync -a --exclude analysis $LES/ $V/lesion_protocol/
cp -r $LES/analysis $V/lesion_protocol/analysis
echo finalized v8
