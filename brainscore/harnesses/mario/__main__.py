"""Each Super Mario Bros. agent plays one level once, its cortex layers recorded; its progress is printed beside
the participants' plays of the same level.

    python -m brainscore.harnesses.mario --rom <your Super Mario Bros. (NES) ROM file> [--level Level1-1] [--out <folder>]

The ROM is checked by its SHA1 before any work, read where it is, and never copied.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from brainscore import load_model
from brainscore.harnesses.mario import check_rom, participants_progress, play_level

AGENTS = ('mario-ppo', 'mario-ppo-untrained')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rom', required=True)
    parser.add_argument('--level', default='Level1-1')
    parser.add_argument('--max-steps', type=int, default=3000)
    parser.add_argument('--out', default='mario-closed-loop')
    args = parser.parse_args(argv)
    rom, out = check_rom(args.rom), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for identifier in AGENTS:
        start = time.time()
        progress, session = play_level(load_model(identifier), rom, args.level,
                                       outputs=('motor', 'neural:cortex'), max_steps=args.max_steps)
        recorded = session.collect('neural:cortex')
        layers = np.asarray(recorded['layer'].values).astype(str)
        np.save(out / f'{identifier}_cortex.npy', np.asarray(recorded.values, dtype=np.float32))
        results[identifier] = {**progress, 'wall_seconds': round(time.time() - start, 1),
                               'recorded_steps': int(recorded.sizes['presentation']),
                               'recorded_units': {layer: int((layers == layer).sum()) for layer in sorted(set(layers))}}
        print(identifier, json.dumps(results[identifier]), flush=True)
    results['participants'] = participants_progress(args.level)
    for participant, row in results['participants'].items():
        print(participant, json.dumps(row))
    (out / 'closed_loop.json').write_text(json.dumps(results, indent=1))
    print(f'written: {out / "closed_loop.json"} and the recorded layers ({out}/<agent>_cortex.npy)')


if __name__ == '__main__':
    main()
