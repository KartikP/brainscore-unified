"""Download pinned public assets into the explicitly selected task directory."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download, snapshot_download
import yaml


ENCODER_REVISION = 'b38a252c28e5416336423f4eda0c42d7e1df4e9c'
ENCODER_SHA256 = '8ee898b4d83466a72192577a492e89bececdd563d0b2fd82f1679dfd1dbce69d'
BACKBONE = 'Qwen/Qwen3-Omni-30B-A3B-Thinking'
BACKBONE_REVISION = '2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', type=Path, required=True)
    parser.add_argument('--backbone-revision', default=BACKBONE_REVISION,
                        help='Pinned Qwen commit; override only for an explicit configuration comparison')
    args = parser.parse_args()
    root = args.asset_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    encoder = root / 'encoder'
    encoder.mkdir(exist_ok=True)
    for name in ('model.safetensors', 'config.yaml', 'README.md'):
        hf_hub_download('epfl-neuroai/mirage', name, revision=ENCODER_REVISION, local_dir=encoder)
    weights = encoder / 'model.safetensors'
    digest = hashlib.sha256()
    with weights.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != ENCODER_SHA256:
        raise ValueError('Public encoder checkpoint checksum mismatch')
    revision = args.backbone_revision
    (root / 'backbone-revision.txt').write_text(revision + '\n')
    snapshot_download(
        repo_id=BACKBONE, revision=revision,
        allow_patterns=['*.json', '*.txt', '*.model', '*.tiktoken', '*.safetensors', '*.jinja'],
        max_workers=8,
    )
    original = encoder / 'config.yaml'
    shutil.copyfile(original, encoder / 'config.original.yaml')
    config = yaml.safe_load(original.read_text())
    for modality in ('text', 'audio', 'vision'):
        config['data'][modality]['revision'] = revision
    original.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {
        'encoder_revision': ENCODER_REVISION, 'checkpoint_sha256': digest.hexdigest(),
        'backbone_model': BACKBONE, 'backbone_revision': revision,
        'original_config_sha256': hashlib.sha256((encoder / 'config.original.yaml').read_bytes()).hexdigest(),
        'resolved_config_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
        'config_changes': 'Pin the same explicit Qwen revision for all three modalities',
    }
    (root / 'assets.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
