"""Read the authors' local rendering outputs without redistributing images."""
from pathlib import Path, PurePosixPath
import subprocess

import pandas as pd

AUTHOR_REVISION = '0651a533d6fe5cb19c007ce9fea9afb9822582ca'


def load_author_data(author_repo: Path, data_root: Path, *,
                     slot_limit: int | None = None) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Load all vocabulary words and country/capital pairs from a pinned checkout.

    Run the authors' backgrounds, vocabulary, and render scripts first. Paths
    recorded on another computer are relocated under ``data_root/renders``.
    A slot limit keeps the full candidate vocabulary but changes the experiment.
    """
    author_repo = Path(author_repo).resolve()
    data_root = Path(data_root).resolve()
    revision = subprocess.check_output(
        ['git', '-C', str(author_repo), 'rev-parse', 'HEAD'], text=True,
    ).strip()
    if revision != AUTHOR_REVISION:
        raise ValueError(f'Use author revision {AUTHOR_REVISION}')
    changes = subprocess.check_output(
        ['git', '-C', str(author_repo), 'status', '--porcelain', '--untracked-files=no'],
        text=True,
    )
    if changes.strip():
        raise ValueError('The authors\' tracked files must be unchanged')
    vocabulary = pd.read_csv(author_repo / 'lexicon/vocabulary.tsv', sep='\t')
    words = set(vocabulary['word'])
    relation = vocabulary[vocabulary['list'] == 'country_capital'].pivot(
        index='item', columns='item_role', values='word',
    )
    pairs = list(zip(relation['country'], relation['capital']))
    if len(words) != 1561 or len(pairs) != 60:
        raise ValueError('Expected the paper\'s 1,561 words and 60 country/capital pairs')
    manifest = pd.read_parquet(data_root / 'lexicon/manifest.parquet')
    manifest = manifest[manifest['condition'] != 'clean'].copy()
    if set(manifest['word']) != words or set(manifest['slot']) != set(range(96)):
        raise ValueError('Expected the full vocabulary rendered on slots 0–95')
    if slot_limit is not None:
        if not isinstance(slot_limit, int) or not 1 <= slot_limit <= 96:
            raise ValueError('slot_limit must be an integer from 1 to 96')
        manifest = manifest[manifest['slot'] < slot_limit].copy()

    def relocate(recorded: str) -> str:
        parts = PurePosixPath(recorded).parts
        if 'renders' not in parts:
            raise ValueError(f'Expected an author renders/ path: {recorded}')
        path = (data_root / Path(*parts[parts.index('renders'):])).resolve()
        if not path.is_relative_to(data_root / 'renders'):
            raise ValueError('Rendered image path escapes the data directory')
        return str(path)

    manifest['image_path'] = manifest['image_path'].map(relocate)
    return manifest, pairs
