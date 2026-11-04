"""Measure country/capital retrieval from images of words, using a DINO encoder."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re

import numpy as np

from brainscore.benchmarks.rendered_word_semantics import RenderedWordRetrieval
from brainscore.benchmarks.rendered_word_semantics.benchmark import file_sha256
from brainscore.benchmarks.rendered_word_semantics.data import AUTHOR_REVISION, load_author_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--author-repo', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', default='facebook/dinov3-vith16plus-pretrain-lvd1689m')
    parser.add_argument('--revision', required=True, help='Immutable Hugging Face commit SHA')
    parser.add_argument('--slot-limit', type=int, help='Pilot: use fewer of the 96 backgrounds')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        parser.error('--revision must be a full immutable commit SHA')
    if args.output_dir.exists():
        parser.error('--output-dir must be a new directory')

    # Validate the data and output location before loading any model weights.
    from brainscore_core.preflight import check_cache_directory
    check_cache_directory()
    manifest, pairs = load_author_data(
        author_repo=args.author_repo,
        data_root=args.data_root,
        slot_limit=args.slot_limit,
    )
    benchmark = RenderedWordRetrieval(
        manifest=manifest,
        pairs=pairs,
        batch_size=args.batch_size,
    )
    args.output_dir.mkdir(parents=True)
    # Keep input checksums with the result, so a later run can identify edits.
    input_record = benchmark.rows.copy()
    input_record['sha256'] = benchmark.image_hashes
    input_record.to_csv(args.output_dir / 'inputs.csv', index=False)
    from brainscore.benchmarks import rendered_word_semantics
    implementation = Path(rendered_word_semantics.__file__).parent
    record = {
        'status': 'running',
        'scope': 'country-to-capital retrieval only',
        'paper_replication_verified': False,
        'pilot': len(benchmark.slots) != 96,
        'author_revision': AUTHOR_REVISION,
        'checkpoint': args.checkpoint,
        'checkpoint_revision': args.revision,
        'benchmark_identifier': benchmark.identifier,
        'data_sha256': benchmark.data_sha256,
        'words': len(benchmark.words),
        'slots': benchmark.slots,
        'queries': len(pairs),
        'batch_size': args.batch_size,
        'implementation_sha256': {
            name: file_sha256(implementation / name)
            for name in ['benchmark.py', 'data.py', 'model.py']
        },
        'runner_sha256': file_sha256(Path(__file__)),
        'packages': {
            name: importlib.metadata.version(name)
            for name in ['brainscore', 'torch', 'transformers', 'numpy', 'Pillow']
        },
    }
    record_path = args.output_dir / 'result.json'
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    subject = None
    try:
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModel
        from brainscore.benchmarks.rendered_word_semantics.model import build_vision_subject

        config = AutoConfig.from_pretrained(
            args.checkpoint,
            revision=args.revision,
            local_files_only=not args.allow_download,
        )
        if config.model_type not in {'dinov2', 'dinov3_vit'}:
            raise ValueError('Choose a DINOv2 or DINOv3 ViT checkpoint')
        processor = AutoImageProcessor.from_pretrained(
            args.checkpoint,
            revision=args.revision,
            local_files_only=not args.allow_download,
        )
        encoder = AutoModel.from_pretrained(
            args.checkpoint,
            revision=args.revision,
            local_files_only=not args.allow_download,
            dtype=torch.float32,
        )
        subject = build_vision_subject(
            encoder=encoder,
            processor=processor,
            identifier=f'{args.checkpoint}@{args.revision}-final-cls-fp32',
            batch_size=args.batch_size,
        )
        score = benchmark(subject)
        np.savez_compressed(
            args.output_dir / 'word_directions.npz',
            words=np.asarray(score.attrs['words']),
            directions=score.attrs['directions'],
        )
        record.update(
            status='complete',
            score=float(score),
            chance=score.attrs['chance'],
            leaderboard_eligible=False,
            device=str(next(encoder.parameters()).device),
            model_dtype=str(next(encoder.parameters()).dtype),
            processor_class=type(processor).__name__,
            processor_config=processor.to_dict(),
            feature_storage_dtype='float16',
            queries=[
                {'query': a, 'partner': b, 'rank': rank + 1, 'top10': rank < 10}
                for (a, b), rank in zip(pairs, score.attrs['ranks'])
            ],
        )
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if subject is not None:
            subject.reset()
        record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(f"Country/capital top-10 accuracy: {record['score']:.3f}")
    print(f'Results: {record_path}')


if __name__ == '__main__':
    main()
