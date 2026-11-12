"""Scoped observation of download callbacks in the lab worker only."""
from contextlib import contextmanager
import math
from pathlib import Path
import threading
import time
from urllib.parse import unquote, urlsplit
import urllib.request


class DownloadProgress:
    def __init__(self, send):
        self.send = send
        self.lock = threading.RLock()
        self.last_sent = -float('inf')
        self.completed = 0
        self.streams = {}
        self.sequence = 0
        self.latest = {'phase': 'preparing'}

    def emit(self, value, force=False):
        with self.lock:
            self.latest = {**value, 'files_completed': self.completed, 'active_files': len(self.streams)}
            now = time.monotonic()
            if force or now - self.last_sent >= .25:
                self.last_sent = now
                self.send({'download': self.latest})

    def start(self, name, initial=0, total=None):
        with self.lock:
            self.sequence += 1
            key = self.sequence
            # Never publish URLs, query tokens or full local cache paths.
            name = unquote(urlsplit(str(name)).path).replace('\\', '/').rsplit('/', 1)[-1]
            self.streams[key] = {'phase': 'downloading', 'file': name[:120] or 'model weights',
                                 'downloaded_bytes': max(0, int(initial)),
                                 'total_bytes': int(total) if total is not None and math.isfinite(total) and total > 0 else None}
            self.emit(dict(self.streams[key]), True)
            return key

    def update(self, key, received, total=None):
        with self.lock:
            if key not in self.streams:
                return
            value = self.streams[key]
            if total is not None and math.isfinite(total) and total > 0:
                value['total_bytes'] = int(total)
            value['downloaded_bytes'] = max(0, int(received))
            if value['total_bytes'] is not None:
                value['downloaded_bytes'] = min(value['downloaded_bytes'], value['total_bytes'])
            self.emit(dict(value))

    def finish(self, key, success):
        with self.lock:
            value = self.streams.pop(key, None)
            if value is None:
                return
            if success:
                self.completed += 1
            current = dict(next(iter(self.streams.values()))) if self.streams else {**value, 'phase': 'loading'}
            self.emit(current, True)


@contextmanager
def observe_downloads(send):
    """Observe urlretrieve and byte-oriented tqdm bars; restore all hooks on exit.

    No changes to transport, caches, credentials, offline settings or model calls.
    Counters include a tqdm bar's resume offset, even when terminal output is off.
    """
    from tqdm.std import tqdm

    progress = DownloadProgress(send)
    retrieve = urllib.request.urlretrieve
    initialize, update, close = tqdm.__init__, tqdm.update, tqdm.close
    bars = {}

    def urlretrieve(url, filename=None, reporthook=None, data=None):
        key = progress.start(url)
        def report(count, size, total):
            if reporthook is not None:
                reporthook(count, size, total)
            progress.update(key, count * size, total)
        try:
            result = retrieve(url, filename, report, data)
            progress.update(key, Path(result[0]).stat().st_size)
        except BaseException:
            progress.finish(key, False)
            raise
        progress.finish(key, True)
        return result

    def init_bar(bar, *args, **kwargs):
        result = initialize(bar, *args, **kwargs)
        if kwargs.get('unit') == 'B':
            initial = kwargs.get('initial', 0)
            key = progress.start(kwargs.get('desc', 'model weights'), initial, kwargs.get('total'))
            bars[id(bar)] = {'key': key, 'received': initial, 'total': kwargs.get('total')}
        return result

    def update_bar(bar, n=1):
        result = update(bar, n)
        with progress.lock:
            record = bars.get(id(bar))
            if record is not None:
                record['received'] += n
                progress.update(record['key'], record['received'], record['total'])
        return result

    def close_bar(bar):
        with progress.lock:
            record = bars.pop(id(bar), None)
            if record is not None:
                progress.update(record['key'], record['received'], record['total'])
                total = record['total']
                progress.finish(record['key'], total is not None and record['received'] >= total)
        return close(bar)

    urllib.request.urlretrieve = urlretrieve
    tqdm.__init__, tqdm.update, tqdm.close = init_bar, update_bar, close_bar
    try:
        progress.emit({'phase': 'preparing'}, True)
        yield progress
    except BaseException:
        progress.emit({**progress.latest, 'phase': 'failed'}, True)
        raise
    finally:
        urllib.request.urlretrieve = retrieve
        tqdm.__init__, tqdm.update, tqdm.close = initialize, update, close
