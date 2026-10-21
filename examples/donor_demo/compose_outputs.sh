#!/bin/bash
# Compose the looping poster MP4 and previews from the rasterized slide and the replay video.
# usage: compose_outputs.sh <slide_hole.png> <slide.layout.json> <replay.mp4> <out_prefix>
set -euo pipefail
BASE=$1; LAYOUT=$2; REPLAY=$3; OUT=$4
X=$(python3 -c "import json;d=json.load(open('$LAYOUT'))['movie'];print(d['x'])")
Y=$(python3 -c "import json;d=json.load(open('$LAYOUT'))['movie'];print(d['y'])")
W=$(python3 -c "import json;d=json.load(open('$LAYOUT'))['movie'];print(d['w'] - d['w']%2)")
H=$(python3 -c "import json;d=json.load(open('$LAYOUT'))['movie'];print(d['h'] - d['h']%2)")
# 1920x1080 H.264, the replay scaled into the movie rectangle; duration = replay duration (one loop)
ffmpeg -y -loglevel error -loop 1 -i "$BASE" -i "$REPLAY" \
  -filter_complex "[1:v]scale=${W}:${H}:flags=lanczos[v];[0:v][v]overlay=${X}:${Y}:shortest=1,format=yuv420p" \
  -c:v libx264 -crf 18 -r 30 -movflags +faststart "${OUT}.mp4"
# preview GIF: 12 fps, 960 px wide, first loop
ffmpeg -y -loglevel error -i "${OUT}.mp4" -vf "fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3" "${OUT}-preview.gif"
# still preview with the first replay frame in place
ffmpeg -y -loglevel error -i "${OUT}.mp4" -frames:v 1 "${OUT}.png"
echo "wrote ${OUT}.mp4 ${OUT}-preview.gif ${OUT}.png"
