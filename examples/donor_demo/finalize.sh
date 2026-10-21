#!/bin/bash
# Final assembly: notes -> deck -> raster -> MP4/GIF/PNG -> docs. Usage: finalize.sh [algonauts_json]
set -euo pipefail
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/analytics-2026/bin/python
P=/Volumes/Samsung9100/Brain-Score-Donor-Demo/presentation
E=/Volumes/Samsung9100/Brain-Score-Donor-Demo/experiment
V=/Volumes/Samsung9100/Brain-Score-Donor-Demo/visualization-data/lahner2024_vjepa1_wav2vec2
ALGO=${1:-}
TABLE="/Users/kartik/Brain-Score Unified/unified/scripts/ec2_results_2026_07_31/task7d_cis.json"
$PY make_notes.py stage_full $ALGO >/dev/null
$PY write_results_doc.py stage_full $E/02_results_and_methods.md $ALGO
$PY build_poster.py --data stage_full --replay replay/replay.mp4 --poster_frame replay/frames/f00000.png \
    --out $P/Brain-Score-TV-Poster-v7.pptx --table_json "$TABLE" ${ALGO:+--algonauts $ALGO}
$PY render_slide_png.py $P/Brain-Score-TV-Poster-v7.pptx probe/v7_hole.png --hole
bash compose_outputs.sh probe/v7_hole.png probe/v7_hole.layout.json replay/replay.mp4 $P/Brain-Score-TV-Poster-v7
cp stage_full/poster_notes.txt $P/Brain-Score-TV-Poster-v7-speaker-notes.txt
cp stage_full/poster_notes.txt $V/poster_notes.txt
echo finalized
