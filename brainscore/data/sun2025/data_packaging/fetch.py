"""Fetch selected source scenes using ZIP ranges, keeping every source byte used.

Run remotely: python -m brainscore.data.sun2025.data_packaging.fetch OUTPUT --scenes 1
Omitting --scenes downloads all 40 scenes. No model or data download happens on import.
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
SOURCE_FILES = (
    'README.md', 'LICENSE', 'second-order-exp/FormalExp.py', 'second-order-exp/ini.py',
    'second-order-exp/configure.py', 'second-order-exp/GenerateTable.py',
    'second-order-exp/Static_AllTrails_Final_shuffled.xlsx',
    'second-order-exp/dataviz/humandata_prepross.ipynb',
    *(f'second-order-exp/data/participant{i}/data.csv' for i in (1, 2, 3)),
)


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

    for name in SOURCE_FILES:
        with urllib.request.urlopen(GITHUB_URL + name, timeout=60) as response:
            save(name, response.read(), url=GITHUB_URL + name)
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
