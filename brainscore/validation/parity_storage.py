"""Temporary activation snapshots with bounded reads and writes."""
from dataclasses import dataclass
from collections.abc import Iterator
import hashlib
from pathlib import Path
import shutil

import numpy as np


CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class ActivationSnapshot:
    path: Path
    shape: tuple[int, ...]
    dtype: np.dtype
    sha256: str

    def chunks(self, elements: int) -> Iterator[np.ndarray]:
        """Read in C order without mapping the whole file into memory."""
        remaining = int(np.prod(self.shape))
        if self.path.stat().st_size != remaining * self.dtype.itemsize:
            raise IOError('Activation snapshot size changed')
        with self.path.open('rb') as handle:
            while remaining:
                count = min(elements, remaining)
                values = np.fromfile(handle, dtype=self.dtype, count=count)
                if len(values) != count:
                    raise IOError('Incomplete activation snapshot')
                remaining -= count
                yield values


def snapshot(
    values: np.ndarray, path: Path, *, chunk_bytes: int = CHUNK_BYTES,
) -> ActivationSnapshot:
    """Save every value and hash the same C-order bytes used by parity reports.

    Buffered iteration also bounds copies for transposed or sliced arrays.
    The caller owns the temporary directory and its cleanup.
    """
    values, path = np.asarray(values), Path(path)
    if values.dtype.kind not in 'biufc':
        raise TypeError('Activation snapshots require numeric arrays')
    if chunk_bytes < values.dtype.itemsize:
        raise ValueError('Chunk budget must fit at least one activation')
    if shutil.disk_usage(path.parent).free < values.nbytes + CHUNK_BYTES:
        raise OSError('Insufficient scratch space for activation snapshot')
    digest = hashlib.sha256()
    with path.open('xb') as handle:
        chunks = np.nditer(
            values, flags=['external_loop', 'buffered', 'zerosize_ok'],
            op_flags=[['readonly', 'contig']], order='C',
            buffersize=chunk_bytes // values.dtype.itemsize,
        )
        for chunk in chunks:
            data = memoryview(chunk).cast('B')
            digest.update(data)
            handle.write(data)
    return ActivationSnapshot(path, values.shape, values.dtype, digest.hexdigest())
