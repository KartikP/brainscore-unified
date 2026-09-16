"""Execution controls for the standalone, single-process parity runner."""
from contextlib import contextmanager, ExitStack
import hashlib
from importlib import metadata
from pathlib import Path
import platform
from unittest.mock import patch

# Serialized reference inputs used for the first CPU calibration. Re-saving or
# changing them requires a separately recorded qualification, not silent reuse.
CPU_GPT2_SHA256 = {
    'model.safetensors': '248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707',
    'config.json': '0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4',
    'tokenizer_config.json': '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
    'tokenizer.json': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
    'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
    'vocab.json': '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783',
}


def require_cpu_reference(manifest):
    """Require the serialized GPT-2 inputs shared by the CPU and L4 profiles."""
    actual = {name.removeprefix('gpt2/'): value['sha256']
              for name, value in manifest.items() if name.startswith('gpt2/')}
    if actual != CPU_GPT2_SHA256:
        raise ValueError('The fixed FP32 policy requires the calibrated GPT-2 checkpoint/tokenizer file set')


def checkpoint_manifest(resnet18, gpt2):
    """Hash explicit local inputs, following snapshot symlinks to file contents."""
    resnet18, gpt2 = Path(resnet18), Path(gpt2)
    if not resnet18.is_file() or not gpt2.is_dir():
        raise FileNotFoundError('Stage the ResNet18 file and GPT-2 directory before qualification')
    files = {'resnet18': resnet18}
    files.update({f'gpt2/{path.relative_to(gpt2)}': path
                  for path in sorted(gpt2.rglob('*')) if path.is_file()})
    if len(files) == 1:
        raise FileNotFoundError('The GPT-2 checkpoint directory is empty')
    result = {}
    for name, path in files.items():
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        result[name] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    return result


@contextmanager
def execution(device, threads):
    """Pin automatic wrapper placement and refuse unstaged data downloads.

    Availability overrides are confined to this standalone harness, not the
    model API. An explicit CUDA request must pass the real availability check.
    """
    import torch
    from threadpoolctl import threadpool_limits, threadpool_info
    from brainscore_core.supported_data_standards.brainio.fetch import BotoFetcher
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError(f'Unsupported parity device: {device}')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA qualification requires an available CUDA device; no CPU fallback')
    if threads < 1:
        raise ValueError('Thread count must be positive')
    previous_threads = torch.get_num_threads()
    previous_matmul, previous_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    def missing_data(fetcher):
        raise FileNotFoundError(f'Parity input was not staged: {fetcher.output_filename}')
    try:
        torch.set_num_threads(threads)
        if device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        with ExitStack() as stack:
            stack.enter_context(threadpool_limits(limits=threads))
            stack.enter_context(patch.object(BotoFetcher, 'download_boto', missing_data))
            if device in ('cpu', 'cuda'):
                stack.enter_context(patch('torch.backends.mps.is_available', return_value=False))
            if device == 'cpu':
                stack.enter_context(patch('torch.cuda.is_available', return_value=False))
            yield {
                'requested_device': device,
                'platform': platform.platform(), 'machine': platform.machine(),
                'python': platform.python_version(),
                'dependencies': {name: metadata.version(name) for name in
                    ('torch', 'torchvision', 'transformers', 'numpy', 'scipy', 'scikit-learn', 'xarray')},
                'torch_threads': torch.get_num_threads(),
                'torch_interop_threads': torch.get_num_interop_threads(),
                'threadpools': threadpool_info(), 'cuda_version': torch.version.cuda,
                'cuda_name': torch.cuda.get_device_name() if device == 'cuda' else None,
                'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                'persistent_result_cache': 'disabled', 'data_downloads': 'refused',
            }
    finally:
        torch.set_num_threads(previous_threads)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn
