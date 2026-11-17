"""Fetch the authors' CogGym repository at a pinned revision: the checkout the runner reads.

    python -m brainscore.data.coggym.data_packaging.fetch --revision <commit> --out <new folder>

``python -m brainscore.data package coggym`` runs it with the registered revision.
"""
import argparse
import subprocess

REPOSITORY = 'https://github.com/lance-ying/coggym.git'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--revision', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    # Clone into the (empty) output folder, then pin the revision the integration was checked against.
    subprocess.run(['git', 'clone', '--quiet', REPOSITORY, args.out], check=True)
    subprocess.run(['git', '-C', args.out, 'checkout', '--quiet', args.revision], check=True)


if __name__ == '__main__':
    main()
