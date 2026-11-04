"""Execute the Sun notebook unchanged in fresh kernels on the remote worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--saved-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=('run', 'replay'), default=['run', 'replay'])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    source = repo / 'notebooks/experiments/15_sun2025_motion_and_activity_replay.ipynb'
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for mode in args.modes:
        notebook = nbformat.read(source, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs, cell.execution_count = [], None
        output = args.output / mode
        output.mkdir()
        env = dict(os.environ, BRAINSCORE_SUN2025_ROOT=str(args.data),
                   SUN2025_NOTEBOOK_MODE=mode, SUN2025_RUN=str(args.saved_run),
                   SUN2025_NOTEBOOK_OUTPUT=str(output / 'new-run'),
                   RESULTCACHING_DISABLE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_HUB_DISABLE_PROGRESS_BARS='1', TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MPLBACKEND='Agg')
        manager = KernelManager(kernel_name='python3')
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        client = NotebookClient(notebook, km=manager, timeout=600, allow_errors=False,
                                resources={'metadata': {'path': str(repo)}})
        record = {'mode': mode, 'passed': False}
        try:
            client.execute(env=env, cleanup_kc=True)
            last = next(cell for cell in reversed(notebook.cells) if cell.cell_type == 'code')
            result = next(item for item in last.outputs if item.output_type == 'execute_result')
            # The notebook's final expression is the receipt also written beside the HTML.
            import ast
            receipt = ast.literal_eval(result.data['text/plain'])
            if receipt['trial_count'] != 4 or receipt['neural_shape'][:2] != [4, 8]:
                raise ValueError('Unexpected demonstration selection or recording dimensions')
            if not Path(receipt['replay_html']).is_file():
                raise FileNotFoundError('Notebook did not export the replay HTML')
            record.update(passed=True, receipt=receipt)
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            nbformat.write(notebook, output / source.name)
            records.append(record)
            (args.output / 'notebook-validation.json').write_text(json.dumps({
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'python': sys.version, 'results': records,
                'passed': len(records) == len(args.modes) and all(row['passed'] for row in records),
            }, indent=2) + '\n')
        print(f'PASS: {mode}', flush=True)


if __name__ == '__main__':
    main()
