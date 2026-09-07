"""``python -m brainscore.data`` — what you must supply yourself.

Lists every asset a benchmark needs but Brain-Score cannot ship, and whether
it is present. Pass a name for how to obtain and convert that one.
"""

import sys

from . import local


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(local.format_status())
        return 0
    name = argv[0]
    if name not in local.REGISTRY:
        print(f'unknown asset {name!r}; known: {sorted(local.REGISTRY)}')
        return 1
    print(local.REGISTRY[name].instructions())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
