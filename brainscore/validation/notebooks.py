"""Execute the offline tutorials with fresh kernels and retain failure evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


# Downloads, GPU runs and full scientific reruns have separate qualification requirements.
EXCLUDED = {
    'experiments/02_which_layer_matches_which_region.ipynb': 'Pretrained ResNet-50 download',
    'experiments/09_recording_and_silencing_resnet18.ipynb': 'Pretrained ResNet-18 download',
    'experiments/03_predicting_it_responses.ipynb': 'MajajHong2015 data and CLIP download; GPU recommended',
    'experiments/06_spatial_organization_of_units.ipynb': 'TDANN checkpoints and positions download (280 MB)',
    'experiments/10_story_listening_fmri.ipynb': 'LeBel 2023 assembly (3 GB) and GPT-2 download',
    'experiments/13_passing_outputs_between_subjects.ipynb': 'GPT-2 and DistilGPT-2 download',
    'experiments/14_reading_instructions_and_words.ipynb': 'Qwen2.5-VL-7B download and GPU run',
}
SELECTED = (
    'lessons/01_your_first_subject.ipynb', 'lessons/02_your_first_benchmark.ipynb',
    'lessons/03_wrap_a_pytorch_model.ipynb', 'lessons/04_bring_an_existing_model.ipynb',
    'lessons/05_layers_and_regions.ipynb', 'lessons/06_behavior.ipynb',
    'lessons/07_perturb_a_model.ipynb', 'lessons/08_choose_units.ipynb',
    'lessons/09_closed_loop.ipynb', 'lessons/10_run_an_experiment.ipynb',
    'lessons/11_interventions_in_an_experiment.ipynb',
    'experiments/04_silencing_units_in_a_digit_classifier.ipynb',
    'experiments/08_learning_in_a_closed_loop.ipynb',
)
# Folders shared by every notebook; experiments reference them as ../assets etc.
SHARED_FOLDERS = ('assets', 'data', 'figure_sources')


def execute(source: Path, destination: Path, timeout: int) -> None:
    """Run unchanged tutorial cells using this interpreter, outside the checkout."""
    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient

    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            images = re.findall(r'!\[[^\]]*\]\(([^\s)]+)\)', cell.source)
            images += re.findall(r'<img\b[^>]*src=[\"\x27]([^\"\x27]+)', cell.source)
            for image in images:
                if '://' not in image and not image.startswith('data:'):
                    if not (source.parent / image).is_file():
                        raise FileNotFoundError(f'Missing notebook image: {image}')
    # Never mistake saved output for execution evidence.
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    # Match the CPU pytest profile without changing the tutorial source.
    notebook.cells.insert(0, nbformat.v4.new_code_cell(
        'import torch\n'
        'torch.cuda.is_available = lambda: False\n'
        'torch.backends.mps.is_available = lambda: False\n'
        'torch.set_num_threads(1)\n',
        metadata={'tags': ['umi-validation-setup']},
    ))
    with tempfile.TemporaryDirectory(prefix='umi-notebook-') as temporary:
        work = Path(temporary)
        # Mirror the notebooks folder: shared folders at the root, the notebook
        # one level below, so both ./assets and ../assets resolve.
        run_dir = work / 'notebook'
        run_dir.mkdir()
        for folder in SHARED_FOLDERS:
            if (source.parent / folder).exists():
                shutil.copytree(source.parent / folder, run_dir / folder)
            if (source.parent.parent / folder).exists():
                shutil.copytree(source.parent.parent / folder, work / folder)
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env.update({
            'RESULTCACHING_DISABLE': '1', 'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1', 'MPLBACKEND': 'Agg',
            'RESULTCACHING_HOME': str(work / 'cache'),
            'HF_HOME': str(work / 'huggingface'), 'TORCH_HOME': str(work / 'torch'),
            'MPLCONFIGDIR': str(work / 'matplotlib'),
            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        })
        manager = KernelManager(kernel_name='python3')
        manager.kernel_spec.argv = [
            sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}',
        ]
        client = NotebookClient(
            notebook, km=manager, timeout=timeout, allow_errors=False,
            resources={'metadata': {'path': str(run_dir)}},
        )
        try:
            client.execute(env=env, cleanup_kc=True)
        finally:
            # Keep partial output on failure, without modifying the teaching file.
            nbformat.write(notebook, destination)


def run(source: Path, output: Path, timeout: int = 180) -> dict:
    """Run the explicit selection; missing, new or failing notebooks fail the check."""
    if source.resolve() == output.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError('Execution output must not overwrite source notebooks')
    output.mkdir(parents=True, exist_ok=True)
    actual = {
        path.relative_to(source).as_posix() for path in source.rglob('*.ipynb')
        if not {'archive', '.ipynb_checkpoints'} & set(path.relative_to(source).parts)
    }
    expected = set(SELECTED) | set(EXCLUDED)
    report = {
        'python': sys.version, 'executable': sys.executable,
        'device_profile': 'cpu',
        'excluded': EXCLUDED, 'missing': sorted(expected - actual),
        'unclassified': sorted(actual - expected), 'results': [], 'passed': False,
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    # Saved Markdown figures must still open after the temporary kernels exit.
    for folder in SHARED_FOLDERS:
        if (source / folder).exists():
            shutil.copytree(source / folder, output / folder, dirs_exist_ok=True)
    for name in SELECTED:
        path = source / name
        result = {'notebook': name, 'passed': False}
        try:
            result['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            execute(path, output / name, timeout)
            result['passed'] = True
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
        report['results'].append(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {name}", flush=True)
        report['passed'] = (
            not report['missing'] and not report['unclassified']
            and len(report['results']) == len(SELECTED)
            and all(row['passed'] for row in report['results'])
        )
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=180, help='Seconds per cell')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    raise SystemExit(0 if run(args.source, args.out, args.timeout)['passed'] else 1)


if __name__ == '__main__':
    main()
