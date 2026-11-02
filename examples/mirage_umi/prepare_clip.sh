#!/usr/bin/env bash
# Short audiovisual input for fidelity checks; no human response measurements.
set -euo pipefail
MIRAGE_RUN_ROOT="${MIRAGE_RUN_ROOT:-/opt/mirage-run}"
mkdir -p "${MIRAGE_RUN_ROOT}/stimuli"
curl --fail --location --retry 2 \
  'https://raw.githubusercontent.com/bower-media-samples/big-buck-bunny-1080p-30s/master/video.mp4' \
  --output "${MIRAGE_RUN_ROOT}/stimuli/source.mp4"
ffmpeg -y -loglevel error -ss 10 \
  -i "${MIRAGE_RUN_ROOT}/stimuli/source.mp4" -t 4 \
  -vf 'scale=320:-2' -c:v libx264 -crf 23 -c:a aac \
  "${MIRAGE_RUN_ROOT}/stimuli/clip.mp4"
printf '[]\n' > "${MIRAGE_RUN_ROOT}/stimuli/transcript.json"
ffprobe -v error -show_entries format=duration:stream=codec_type \
  -of json "${MIRAGE_RUN_ROOT}/stimuli/clip.mp4" \
  > "${MIRAGE_RUN_ROOT}/stimuli/media-info.json"
