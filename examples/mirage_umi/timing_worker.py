"""Benchmark-only timers around the original native CLI or qualified worker."""

import argparse
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import time


def main() -> None:
    entered = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timings', required=True, type=Path)
    parser.add_argument('target', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    target = args.target
    if target[0] == '--':
        target = target[1:]
    measurements = {'status': 'failed'}
    try:
        imported = time.perf_counter()
        import torch
        from brain_enc.eval import infer_fmri
        measurements['imports_s'] = time.perf_counter() - imported
        original_inference = infer_fmri.run_inference

        def inference(request):
            sync_start = time.perf_counter()
            if request.device.startswith('cuda'):
                torch.cuda.synchronize()
            measurements['initial_cuda_sync_s'] = time.perf_counter() - sync_start
            start = time.perf_counter()
            values = original_inference(request)
            if request.device.startswith('cuda'):
                torch.cuda.synchronize()
            measurements['native_inference_s'] = time.perf_counter() - start
            return values

        infer_fmri.run_inference = inference
        sys.argv = target
        if target[:2] == ['-m', 'brain_enc.cli.infer_fmri']:
            sys.argv = [target[1], *target[2:]]
            runpy.run_module(target[1], run_name='__main__')
        else:
            spec = importlib.util.spec_from_file_location('qualified_mirage_worker', target[0])
            worker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(worker)
            original_sha256 = worker.sha256

            def checksum(path):
                start = time.perf_counter()
                result = original_sha256(path)
                if path.name == 'model.safetensors':
                    measurements['checkpoint_verification_s'] = time.perf_counter() - start
                return result

            worker.sha256 = checksum
            worker.main()
        measurements['status'] = 'complete'
    finally:
        measurements['worker_body_s'] = time.perf_counter() - entered
        args.timings.write_text(json.dumps(measurements, indent=2) + '\n')


if __name__ == '__main__':
    main()
