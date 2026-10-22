#!/bin/bash
# Compose the looping poster MP4 (slide 1 with the replay overlaid, then slide 2 held still) and previews.
# usage: compose_outputs_v8.sh <slide1_hole.png> <slide1.layout.json> <replay.mp4> <slide2.png> <out_prefix> [slide2_hold_s]
set -euo pipefail
BASE=$1; LAYOUT=$2; REPLAY=$3; S2=$4; OUT=$5; HOLD=${6:-25}
X=$(python3 -c "import json;d=json.load(open('$LAYOUT'))['movie'];print(d['x'])")
Y=$(python3 -c "import json;d=json.load(open('$LAYOUT'))['movie'];print(d['y'])")
W=$(python3 -c "import json;d=json.load(open('$LAYOUT'))['movie'];print(d['w'] - d['w']%2)")
H=$(python3 -c "import json;d=json.load(open('$LAYOUT'))['movie'];print(d['h'] - d['h']%2)")
T=$(mktemp -d)
ffmpeg -y -loglevel error -loop 1 -i "$BASE" -i "$REPLAY" \
  -filter_complex "[1:v]scale=${W}:${H}:flags=lanczos[v];[0:v][v]overlay=${X}:${Y}:shortest=1,format=yuv420p" \
  -c:v libx264 -crf 18 -r 30 -movflags +faststart "$T/s1.mp4"
ffmpeg -y -loglevel error -loop 1 -t "$HOLD" -i "$S2" -vf "format=yuv420p" -c:v libx264 -crf 18 -r 30 "$T/s2.mp4"
printf "file '%s'\nfile '%s'\n" "$T/s1.mp4" "$T/s2.mp4" > "$T/list.txt"
ffmpeg -y -loglevel error -f concat -safe 0 -i "$T/list.txt" -c copy -movflags +faststart "${OUT}.mp4"
# preview GIF: first 40 s of slide 1 then 4 s of slide 2, 12 fps, 960 px
ffmpeg -y -loglevel error -i "$T/s1.mp4" -t 40 -vf "fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3" "${OUT}-preview.gif"
ffmpeg -y -loglevel error -i "$T/s1.mp4" -frames:v 1 "${OUT}.png"
cp "$S2" "${OUT}-slide2.png"
rm -rf "$T"
echo "wrote ${OUT}.mp4 ${OUT}-preview.gif ${OUT}.png ${OUT}-slide2.png"
