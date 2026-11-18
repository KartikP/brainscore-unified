"""Fetch selected source scenes using ZIP ranges, keeping every source byte used.

Run on a compute machine: python -m brainscore.data.sun2025.data_packaging.fetch OUTPUT
Without --scenes it fetches all 40 scenes; --scenes 1 fetches one scene (28 trials).
Nothing is downloaded on import.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

GITHUB_REVISION = 'c5886559b20be313125db0e1ba96baf464eab342'
HF_REVISION = 'a7e83195986131df109e530efda1c4cbb1f21d6d'
ARCHIVE_SIZE = 9878720999
ARCHIVE_URL = ('https://huggingface.co/datasets/sunana/mateiral-controlled-motion-dataset/'
               f'resolve/{HF_REVISION}/human_static.zip')
GITHUB_URL = f'https://raw.githubusercontent.com/anoymized/multi-order-motion-model/{GITHUB_REVISION}/'
# SHA-256 of each file of the authors' repository at GITHUB_REVISION. A changed file is refused.
SOURCE_FILES = {
    'README.md': 'dd519a3182cdc26a4cc731bdd5b46e6ec8ec5c53a03d9cc8cf87b156a3f8fbf4',
    'LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
    'second-order-exp/FormalExp.py': 'e2e05e75ac00fc2ad3d793acb8a4112215aa1be0f915e75c316c397a60c4bb32',
    'second-order-exp/ini.py': '02dde45f44ce3b198f300bf4e1d11ef138a494bc8faca68c5995b801180dd808',
    'second-order-exp/configure.py': '60054f59a9efd558552905fa2a6cd2e77b3ef3c2fccfe5cf6670c6a9b4d333d6',
    'second-order-exp/GenerateTable.py': '14032db1a07b6ffc0b59743d1ff7345646bb6aa0dc6d7984fcf2a62d783b199e',
    'second-order-exp/Static_AllTrails_Final_shuffled.xlsx':
        '581d6a06f56e29eaa9d0f73ef868a977f4b8256d8a674fc92441a89ce945ca71',
    'second-order-exp/dataviz/humandata_prepross.ipynb':
        '26ccc42b51de272fb2d91c54c985d7928e76e0aafd0300083a75e01a4a5885ce',
    'second-order-exp/data/participant1/data.csv': '07b89d0534fd1e1f2265323e82a245b0799326c03f1fd6b076d511cf712f3fa0',
    'second-order-exp/data/participant2/data.csv': 'e0234eb02ddcaa20a2e351055ce104a56172c12c9ff830c669756949df40be46',
    'second-order-exp/data/participant3/data.csv': '0118afd9ae16ad0e641a998350509548294a2337504e4a5a76c19c59d35c88b2',
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RangeReader(io.RawIOBase):
    """A seekable public ZIP with a bounded cache and strict HTTP range checks."""
    def __init__(self):
        self.position, self.cache_start, self.cache = 0, 0, b''

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError('Invalid seek mode')
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else ARCHIVE_SIZE)
        if self.position < 0:
            raise ValueError('Negative archive position')
        return self.position

    @staticmethod
    def fetch(start, count):
        if not 0 < count <= 128 * 1024**2:
            raise ValueError('Archive reads must be between 1 byte and 128 MiB')
        end = start + count - 1
        request = urllib.request.Request(ARCHIVE_URL + f'?range={start}-{end}',
                                         headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(request, timeout=120) as response:
            expected = f'bytes {start}-{end}/{ARCHIVE_SIZE}'
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                raise ValueError('Archive server did not honor the requested byte range')
            data = response.read(count + 1)
        if len(data) != count:
            raise ValueError('Archive range has the wrong length')
        return data

    def prefetch(self, start, end):
        self.cache = self.fetch(start, end - start)
        self.cache_start = start

    def read(self, count=-1):
        count = min(ARCHIVE_SIZE - self.position, count if count >= 0 else ARCHIVE_SIZE)
        if count <= 0:
            return b''
        offset = self.position - self.cache_start
        if 0 <= offset and offset + count <= len(self.cache):
            data = self.cache[offset:offset + count]
        else:
            data = self.fetch(self.position, count)
        self.position += count
        return data


def fetch_source(destination: Path, scenes=range(1, 41)) -> dict:
    scenes = sorted(set(scenes))
    if not scenes or any(type(scene) is not int or not 1 <= scene <= 40 for scene in scenes):
        raise ValueError('Scenes must be integers 1..40')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = {}

    def save(name, data, **details):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[name] = {'sha256': digest(data), 'bytes': len(data), **details}

    for name, expected in SOURCE_FILES.items():
        with urllib.request.urlopen(GITHUB_URL + name, timeout=60) as response:
            data = response.read()
        if digest(data) != expected:
            raise ValueError(f'{name} at revision {GITHUB_REVISION} has a changed checksum')
        save(name, data, url=GITHUB_URL + name)
    reader = RangeReader()
    with zipfile.ZipFile(reader) as archive:
        index = {entry.filename: entry for entry in archive.infolist()}
        for scene in scenes:
            prefix = f'human_static/image/Scene_{scene}/'
            name = prefix + 'NewConvention_Final.xlsx'
            save(name, archive.read(name), archive_crc32=index[name].CRC)
            for modulation in range(1, 8):
                names = [prefix + f'type_{modulation}/frame_{frame:04d}.png' for frame in range(16)]
                entries = [index[name] for name in names]
                start = min(entry.header_offset for entry in entries)
                last = max(entries, key=lambda entry: entry.header_offset)
                # Local header, name and extra fields fit in this conservative allowance.
                end = min(ARCHIVE_SIZE, last.header_offset + last.compress_size + 65536)
                reader.prefetch(start, end)
                for name in names:
                    save(name, archive.read(name), archive_crc32=index[name].CRC)
            print(f'Fetched scene {scene}', flush=True)
    manifest = {'schema_version': 1, 'github_revision': GITHUB_REVISION,
                'hf_revision': HF_REVISION, 'archive_url': ARCHIVE_URL,
                'archive_size_bytes': ARCHIVE_SIZE, 'scenes': scenes, 'files': files}
    (destination / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--scenes', type=int, nargs='+', default=list(range(1, 41)))
    args = parser.parse_args()
    fetch_source(args.output, args.scenes)
