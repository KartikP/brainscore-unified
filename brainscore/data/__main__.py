"""Build a dataset from native files or an author-supported access identifier."""
import argparse
import sys
from . import local
from .preparation import data_builder_registry, prepare_dataset


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # Preserve the existing asset-status and named-asset commands.
    if not argv:
        print(local.format_status())
        return 0
    if argv[0] not in ('prepare', 'list', '-h', '--help'):
        if argv[0] not in local.REGISTRY:
            print(f'unknown asset {argv[0]!r}; known: {sorted(local.REGISTRY)}')
            return 1
        print(local.REGISTRY[argv[0]].instructions())
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='List available local data builders')
    prepare = commands.add_parser('prepare', help='Build and validate a local dataset')
    prepare.add_argument('dataset')
    location = prepare.add_mutually_exclusive_group(required=True)
    location.add_argument('--source', help='Path to the authors’ native data')
    location.add_argument('--request-id', help='Identifier supported by this data provider')
    prepare.add_argument('--output', required=True, help='New local output directory')
    args = parser.parse_args(argv)
    if args.command == 'list':
        print('\n'.join(sorted(data_builder_registry)))
        return 0
    try:
        result = prepare_dataset(
            args.dataset, source=args.source, request_id=args.request_id,
            output=args.output,
        )
    except (ValueError, FileNotFoundError, FileExistsError, PermissionError, ImportError) as error:
        parser.exit(2, f'{error}\n')
    print(f'Prepared {args.dataset} at {result}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
