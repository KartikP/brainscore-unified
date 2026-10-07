"""Build the Fitts pointing site with nodekit. Run in the nodekit environment:

    python build_site.py [outdir]

nodekit needs Python >= 3.12, so this file is not imported by brainscore. Pin:
nodekit @ git+https://github.com/intelligence-observatory/nodekit@3a13ac4
(0.3.0.dev8; pointer logging is not in the v0.2.8 release). The output (the
static site plus ``trials.json``) is committed so scoring needs no nodekit.

Each trial is a home button at the centre and then one square target, as in
nodekit's ``examples/fitts-law``. Widths x distances x 4 directions are fully
crossed and shuffled with a fixed seed.
"""
import json
import random
import sys
from pathlib import Path

import nodekit as nk

WIDTHS = (32, 64, 128)
DISTANCES = (128, 256, 384)
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
HOME_SIZE = 50
CARD_COLOR = '#b8b8b8'


def _button(name, x, y, size):
    return nk.Node(board_color='#ffffff', card=None, sensor=nk.sensors.SelectSensor(choices={
        name: nk.cards.TextCard(region=nk.Region(x=x, y=y, w=size, h=size), text='',
                                background_color=CARD_COLOR)}))


def trial_graph(width, x, y):
    return nk.Graph(start='home', nodes={'home': _button('home', 0, 0, HOME_SIZE),
                                         'target': _button('target', x, y, width)},
                    transitions={'home': nk.transitions.Go(to='target'),
                                 'target': nk.transitions.End()})


def main(outdir):
    trials = [dict(width=w, distance=d, x=d * dx, y=d * dy)
              for w in WIDTHS for d in DISTANCES for dx, dy in DIRECTIONS]
    random.Random(0).shuffle(trials)
    graph = nk.concat([trial_graph(t['width'], t['x'], t['y']) for t in trials])
    result = nk.build_site(graph, outdir, slug='fitts')
    (Path(outdir) / 'trials.json').write_text(json.dumps(
        {'home_size': HOME_SIZE, 'entrypoint': str(result.entrypoint), 'trials': trials}, indent=1))
    print(result.site_root / result.entrypoint)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / 'site'))
