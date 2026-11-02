"""Create an explicit synthetic audiovisual/text input, without human data."""

import argparse
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    transcript = []
    onset = 0.0
    for word in ('green', 'square'):
        path = args.out / f'{word}.wav'
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
            '-i', f'flite=text={word}:voice=slt', str(path),
        ], check=True)
        duration = float(subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(path),
        ], text=True))
        transcript.append({'word': word, 'onset': onset, 'duration': duration})
        onset += duration
    duration = max(4.0, onset)
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'lavfi', '-i', f'color=c=black:s=320x240:r=8:d={duration}',
        '-i', str(args.out / 'green.wav'), '-i', str(args.out / 'square.wav'),
        '-filter_complex',
        '[0:v]drawbox=x=110:y=70:w=100:h=100:color=green:t=fill[v];'
        '[1:a][2:a]concat=n=2:v=0:a=1,apad[a]',
        '-map', '[v]', '-map', '[a]', '-t', str(duration),
        '-c:v', 'libx264', '-crf', '23', '-c:a', 'aac', str(args.out / 'clip.mp4'),
    ], check=True)
    (args.out / 'transcript.json').write_text(json.dumps(transcript, indent=2) + '\n')
    (args.out / 'description.json').write_text(json.dumps({
        'stimulus': 'synthetic green square with synthesized words green and square',
        'timing': 'word-audio segment boundaries, not manually estimated phoneme onsets',
        'human_data': False,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
