"""Serve a trained OpenPI JAX policy with request-scoped experiment tools.

Run in the pinned OpenPI environment with the four UMI repositories on
PYTHONPATH. The checkpoint and its normalization files must already be local.
"""
import argparse
import hashlib
import json
from pathlib import Path


def checkpoint_digest(directory):
    """Hash file names and content so the run identifies the actual weights."""
    digest = hashlib.sha256()
    count = 0
    for path in sorted(directory.rglob('*')):
        if not path.is_file():
            continue
        file_digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                file_digest.update(block)
        digest.update(path.relative_to(directory).as_posix().encode() + b'\0')
        digest.update(file_digest.digest())
        count += 1
    if count == 0:
        raise ValueError('Checkpoint directory is empty')
    return digest.hexdigest()


def load_policy(config_name, checkpoint, *, num_steps=10):
    # OpenPI retains responsibility for model loading and observation/action transforms.
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config
    return create_trained_policy(
        get_config(config_name),
        checkpoint,
        sample_kwargs={'num_steps': num_steps},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='pi05_libero')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--steps', nargs='+', type=int, help='Denoising iterations to inspect/change')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    if not checkpoint.is_dir():
        parser.error('--checkpoint must be a local directory')
    if args.num_steps <= 0 or (args.steps and max(args.steps) >= args.num_steps):
        parser.error('steps must fit within positive num-steps')
    identity = f'{args.config}:sha256:{checkpoint_digest(checkpoint)}'
    policy = load_policy(args.config, checkpoint, num_steps=args.num_steps)
    from brainscore.experiments import OpenPIInstrumentation, OpenPIToolServer
    instrumentation = OpenPIInstrumentation(policy, checkpoint=identity, steps=args.steps)
    print(json.dumps(instrumentation.describe(), indent=2), flush=True)
    # Keep this private. Reach a remote GPU through an SSH tunnel.
    OpenPIToolServer(policy, instrumentation).serve(port=args.port)


if __name__ == '__main__':
    main()
