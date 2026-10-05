"""Generate a build-only repair for the pinned upstream LIBERO Dockerfile.

Keep the upstream checkout clean so the evaluator can verify its revision.
The repair pins source-package build tools and installs missing transitive
dependencies while preserving upstream's explicitly pinned runtime versions.
"""
import argparse
from pathlib import Path


def prepare(openpi, output):
    source = (Path(openpi) / 'examples/libero/Dockerfile').read_text()
    original = ('RUN uv pip sync /tmp/requirements.txt /tmp/requirements-libero.txt '
                '/tmp/openpi-client/pyproject.toml --extra-index-url '
                'https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match')
    if source.count(original) != 1:
        raise ValueError('Unexpected upstream Dockerfile; review before adapting')
    replacement = (
        'RUN uv pip install setuptools==75.3.0 wheel==0.45.1 cmake==3.31.6\n'
        'RUN uv pip install --no-build-isolation -r /tmp/requirements.txt '
        '-r /tmp/requirements-libero.txt -r /tmp/openpi-client/pyproject.toml '
        '--extra-index-url https://download.pytorch.org/whl/cu113 '
        '--index-strategy=unsafe-best-match')
    Path(output).write_text(source.replace(original, replacement))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--openpi', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    prepare(args.openpi, args.out)
