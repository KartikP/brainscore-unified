# Install

UMI is four Python packages that are installed together: `core` (the interface), `vision` and `language` (classic Brain-Score models and benchmarks), and `unified` (wrappers, new models and benchmarks, experiment tools). It needs Python 3.11 and conda.

## On a new machine

```bash
mkdir brainscore-umi && cd brainscore-umi
base=https://raw.githubusercontent.com/KartikP/brainscore-unified/unified-model-interface-v2/install
curl -fsSLO "$base/setup.sh"
curl -fsSLO "$base/environment-unified.yml"
bash setup.sh                       # clones the four repositories and builds the environment
conda activate brainscore-unified
```

## With the four repositories already side by side

```bash
conda create -y -n brainscore-unified python=3.11
conda activate brainscore-unified
python -m pip install -c unified/install/v2-constraints.txt \
    -e ./core -e ./vision -e ./language -e "./unified[notebooks,test]"
```

## Check the installation

```bash
python -m brainscore.doctor
```

This checks package versions and that the result cache is writable. If the cache points to a drive that is not mounted, set `RESULTCACHING_HOME=/some/writable/folder` before starting Python, or `RESULTCACHING_DISABLE=1` to turn caching off.

More detail, including common conda mistakes, is in [install/README.md](../../install/README.md).

**Next:** [Lesson 1](../../notebooks/lessons/01_your_first_subject.ipynb).
