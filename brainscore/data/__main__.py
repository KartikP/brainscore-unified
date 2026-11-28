"""Package a data plugin from its original source, or list the plugins and local assets.

    python -m brainscore.data                       each plugin's source place, and the local assets
    python -m brainscore.data package <name> [--source <folder>] [--output <new folder>] [--part <part>]
    python -m brainscore.data <asset name>          how to obtain one local asset
"""
import argparse
import sys

from . import local
from .packaging import listing, package_data


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(listing())
        print()
        print(local.format_status())
        return 0
    if argv[0] not in ('package', '-h', '--help'):
        if argv[0] not in local.REGISTRY:
            print(f'unknown asset {argv[0]!r}; known: {sorted(local.REGISTRY)}')
            return 1
        print(local.REGISTRY[argv[0]].instructions())
        return 0
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    package = commands.add_parser('package', help='Package a data plugin from its original source')
    package.add_argument('name')
    package.add_argument('--source', help='the folder of the original source data')
    package.add_argument('--output', help='a new folder in place of the place where the benchmarks read the data')
    package.add_argument('--part', help='a part of a large dataset, when its plugin can package one (its README says how)')
    args = parser.parse_args(argv)
    try:
        package_data(args.name, source=args.source, output=args.output, part=args.part)
    except (ValueError, FileNotFoundError, FileExistsError, PermissionError, ImportError,
            NotImplementedError, RuntimeError) as error:
        parser.exit(2, f'{error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
