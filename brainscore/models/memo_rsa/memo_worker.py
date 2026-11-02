"""Python-compatible bootstrap for the isolated memo worker process."""

import json
import sys


def _write(response):
    sys.stdout.write(json.dumps(response, allow_nan=False) + '\n')
    sys.stdout.flush()


def _run():
    if sys.version_info < (3, 12):
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('op') == 'shutdown':
                return
            _write({'ok': False, 'error': 'memo worker requires Python >=3.12'})
            return
        return

    import jax
    import jaxlib
    from memo import __version__ as memo_version
    from memo_model import predict

    runtime = {
        'python': sys.version.split()[0],
        'python_major_minor': list(sys.version_info[:2]),
        'memo_version': memo_version,
        'jax_version': jax.__version__,
        'jaxlib_version': jaxlib.__version__,
        'jax_enable_x64': bool(jax.config.jax_enable_x64),
        'jax_default_matmul_precision': jax.config.jax_default_matmul_precision,
        'devices': [{'platform': device.platform, 'device_kind': device.device_kind,
                     'id': device.id} for device in jax.devices()],
        'backend': jax.default_backend(),
    }
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.get('op')
            if operation == 'shutdown':
                return
            if operation == 'hello':
                response = {'ok': True, **runtime}
            elif operation == 'predict':
                rows, inference_dtype = predict(
                    depth=request['depth'], beta=request['beta'],
                    utterances=request['utterances'],
                )
                response = {'ok': True, 'runtime': {**runtime, 'inference_dtype': inference_dtype},
                            'probabilities': rows}
            else:
                raise ValueError(f'unknown operation {operation!r}')
        except Exception as error:
            response = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
        _write(response)


if __name__ == '__main__':
    _run()
