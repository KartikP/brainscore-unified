"""Shared output checks for experiments and external bridges."""
from pathlib import Path
import tempfile


def check_output_directory(directory: Path) -> None:
    """Check a new output location without creating it or reserving disk space."""
    if directory.exists():
        raise FileExistsError(directory)
    parent = directory.absolute().parent
    while not parent.exists():
        parent = parent.parent
    try:
        with tempfile.TemporaryFile(dir=parent) as probe:
            probe.write(b'UMI output check')
            probe.flush()
    except OSError as error:
        raise type(error)(error.errno, f'Cannot write experiment output: {error}',
                          str(directory)) from error
