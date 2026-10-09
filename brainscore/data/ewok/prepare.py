"""Convert native EWoK CSV/Parquet or the authors' paper archive locally."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

REFERENCE_REVISION = '9e40d30e242925866ee50448f80a13bcdf971318'
HF_REPOSITORY = 'ewok-core/ewok-core-1.0'
HF_REVISION = '34d912a608066c92e2990a0328ffc3bd9a716042'
TEXT_COLUMNS = ('Context1', 'Context2', 'Target1', 'Target2')
KEY_COLUMNS = ('Domain', 'MetaTemplateID', 'TemplateID', 'Version')
MAX_TABLE_BYTES = 32 * 1024 * 1024


def resolve(request_id: str) -> tuple[Path, dict]:
    """Use existing Hugging Face access; never accept terms or log credentials."""
    if request_id != HF_REPOSITORY:
        raise ValueError(
            'EWoK supports the dataset ID ewok-core/ewok-core-1.0. '
            'Author approval codes are not download IDs; alternatively use --source.'
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError('Install brainscore[ewok-data] for hosted EWoK access') from None
    try:
        path = snapshot_download(
            repo_id=HF_REPOSITORY, repo_type='dataset', revision=HF_REVISION,
            allow_patterns=['data/test/ewok-core-1.0.parquet'],
        )
    except Exception:
        # Upstream HTTP errors can include credentials or private access URLs.
        raise PermissionError(
            'Could not obtain EWoK. Request access from the authors on Hugging Face, '
            'authenticate with hf auth login or HF_TOKEN, and retry. '
            'You can also use --source with already downloaded files.'
        ) from None
    return Path(path) / 'data/test/ewok-core-1.0.parquet', {
        'method': 'huggingface', 'repository': HF_REPOSITORY, 'revision': HF_REVISION,
    }


def _read_table(raw: bytes, suffix: str):
    import pandas as pd
    if len(raw) > MAX_TABLE_BYTES:
        raise ValueError('EWoK table exceeds the 32 MB input limit')
    if suffix == '.parquet':
        try:
            return pd.read_parquet(io.BytesIO(raw))
        except ImportError:
            raise ImportError('Install brainscore[ewok-data] to read native Parquet') from None
    return pd.read_csv(io.BytesIO(raw), comment='#', dtype=str, keep_default_na=False)


def _archive_table(path: Path, member: str):
    try:
        import pyzipper
    except ImportError:
        raise ImportError('Install brainscore[ewok-data] to read protected EWoK ZIP files') from None
    password = os.environ.get('EWOK_ARCHIVE_PASSWORD')
    if not password:
        raise ValueError('Set EWOK_ARCHIVE_PASSWORD to the password in the authors’ terms of use')
    with pyzipper.AESZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.file_size > MAX_TABLE_BYTES:
            raise ValueError('EWoK archive member exceeds the 32 MB limit')
        archive.setpassword(password.encode())
        try:
            raw = archive.read(member)
        except RuntimeError:
            raise ValueError('Could not decrypt EWoK archive; check EWOK_ARCHIVE_PASSWORD') from None
    return _read_table(raw, '.csv')


def normalize(frame: "pd.DataFrame", *, version: str | None = None) -> list[dict]:
    """Preserve author labels and family IDs; do not guess missing answer keys."""
    # Native exports differ only in column casing/underscores across providers.
    names = {re.sub(r'[^a-z0-9]', '', c.lower()): c for c in frame.columns}
    for column in (*KEY_COLUMNS, *TEXT_COLUMNS):
        original = names.get(column.lower())
        if original is not None and original != column:
            frame = frame.rename(columns={original: column})
    if 'Version' not in frame and version is not None:
        frame = frame.assign(Version=version)
    missing = set((*KEY_COLUMNS, *TEXT_COLUMNS)) - set(frame.columns)
    if missing:
        raise ValueError(f'Native EWoK table lacks required columns: {sorted(missing)}')
    items = []
    for row in frame.to_dict('records'):
        clean = {}
        for column in (*KEY_COLUMNS, *TEXT_COLUMNS):
            value = row[column]
            if value is None or str(value).strip().lower() in ('', 'nan', 'none'):
                raise ValueError(f'Empty native EWoK field: {column}')
            clean[column] = str(value) if column in TEXT_COLUMNS else str(value).strip()
        if clean['Context1'] == clean['Context2'] or clean['Target1'] == clean['Target2']:
            raise ValueError('EWoK contexts and targets must be contrasting pairs')
        key = json.dumps([clean[c] for c in KEY_COLUMNS], ensure_ascii=False)
        clean['item_id'] = hashlib.sha256(key.encode()).hexdigest()[:24]
        clean['family'] = f"{clean['Domain']}:{clean['MetaTemplateID']}"
        items.append(clean)
    if not items:
        raise ValueError('EWoK source contains no items')
    if len({r['item_id'] for r in items}) != len(items):
        raise ValueError('Duplicate EWoK item IDs; use a single native export, not evaluation outputs')
    return sorted(items, key=lambda r: tuple(r[c] for c in KEY_COLUMNS))


def build(source: Path, destination: Path) -> dict:
    """Write items and checksummed provenance, never redistribute upstream text."""
    import pandas as pd
    files = []
    scope = 'local_author_export'
    exclusions = 0
    # Paper inclusion table has the authors' target reversals already applied.
    # Read only named members, never extract the archive or execute upstream code.
    if source.is_dir() and (source / 'analyses/data.zip').is_file():
        archive = source / 'analyses/data.zip'
        config = source / 'config.zip'
        included = _archive_table(archive, 'data/items_in_results.csv')
        removed = _archive_table(config, 'config/utils/remove_from_results.csv')
        columns = ['Domain', *TEXT_COLUMNS]
        excluded = set(map(tuple, removed[columns].values.tolist()))
        mask = included[columns].apply(tuple, axis=1).isin(excluded)
        exclusions = int(mask.sum())
        frames = [included.loc[~mask]]
        files = [archive, config]
        scope = 'paper_inclusion_table_with_final_exclusions'
    else:
        files = [source] if source.is_file() else sorted(
            p for p in source.rglob('*') if p.suffix in ('.csv', '.parquet')
        )
        if not files:
            raise ValueError('Provide native EWoK CSV/Parquet files or the paper checkout')
        frames = []
        for file in files:
            if file.suffix not in ('.csv', '.parquet'):
                raise ValueError('Expected native CSV or Parquet, or a paper checkout directory')
            if file.stat().st_size > MAX_TABLE_BYTES:
                raise ValueError('EWoK table exceeds the 32 MB input limit')
            frame = _read_table(file.read_bytes(), file.suffix)
            match = re.search(r'vers(?:ion)?=(\d+)', str(file))
            rows = normalize(frame, version=match.group(1) if match else None)
            frames.append(pd.DataFrame(rows))
    items = normalize(pd.concat(frames, ignore_index=True))
    payload = (json.dumps(items, ensure_ascii=False, indent=2) + '\n').encode()
    (destination / 'items.json').write_bytes(payload)
    metadata = {
        'schema_version': 1, 'dataset': 'EWoK-core-1.0', 'scope': scope,
        'items': len(items), 'domains': sorted({r['Domain'] for r in items}),
        'versions': sorted({r['Version'] for r in items}),
        'items_sha256': hashlib.sha256(payload).hexdigest(),
        'sources': [{'name': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
        'final_exclusions': exclusions,
        'answer_rule': 'Context1 matches Target1; Context2 matches Target2',
        'citation': 'https://arxiv.org/abs/2405.09605',
        'terms': 'https://github.com/ewok-core/ewok-paper/blob/main/TERMS_OF_USE.txt',
        'redistribution': 'Do not publish EWoK text or derivatives as plaintext',
        'paper_comparability': 'Not established by conversion; check source, exclusions, versions and evaluation protocol',
    }
    (destination / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata
