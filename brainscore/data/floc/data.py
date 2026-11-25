"""The fLoc stimuli as a ``StimulusSet``, read from the authors' repository, cloned at first use."""
import re
import subprocess
from pathlib import Path
from typing import Optional

# The authors' repository (Stanford VPNL) at a pinned commit; Brain-Score hosts no copy.
REPOSITORY = 'https://github.com/VPNL/fLoc.git'
COMMIT = 'de6a26cc269a2c7075461a4c839bfd628f225c95'
# The five stimulus conditions of the localizer and their two subcategories (the authors' README).
# The repository's ``scrambled`` folder is the oddball task's image, not a condition: it is not read.
CATEGORIES = {'adult': 'faces', 'child': 'faces', 'body': 'bodies', 'limb': 'bodies', 'word': 'characters',
              'number': 'characters', 'car': 'objects', 'instrument': 'objects', 'house': 'places',
              'corridor': 'places'}
IMAGES_PER_SUBCATEGORY = 144


def source() -> Path:
    """The repository at ``COMMIT`` in ``<data home>/sources/floc``, cloned at the first use."""
    from brainscore.data.local import data_home
    clone = data_home() / 'sources' / 'floc'

    def git(*arguments, check=True):
        return subprocess.run(['git', '-C', str(clone), *arguments], capture_output=True, text=True, check=check)
    try:
        if not (clone / '.git').exists():
            clone.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--quiet', '--no-checkout', REPOSITORY, str(clone)],
                           check=True, capture_output=True, text=True)
        if git('rev-parse', 'HEAD', check=False).stdout.strip() != COMMIT or not (clone / 'stimuli').is_dir():
            git('-c', 'core.hooksPath=/dev/null', 'checkout', '--quiet', COMMIT)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f'Brain-Score could not clone the fLoc repository {REPOSITORY} at {COMMIT} into {clone}: '
            f'{(error.stderr or "").strip()[-500:]}. Next step: check the network and run again, or clone it '
            f'there yourself at that commit.') from error
    return clone


def load_stimulus_set(root: Optional[Path] = None):
    """The 1,440 images of the ten subcategories: ``stimulus_id`` (the file name without its suffix),
    ``subcategory``, ``category`` (the condition) and ``number`` (1 to 144) and ``image_path``. ``root`` is a checkout of the
    repository; by default ``source()``."""
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    folder = Path(root or source()) / 'stimuli'
    rows, paths = [], {}
    for subcategory, category in CATEGORIES.items():
        files = sorted((folder / subcategory).glob(f'{subcategory}-*.jpg'))
        if len(files) != IMAGES_PER_SUBCATEGORY:
            raise ValueError(f'{folder / subcategory} holds {len(files)} images, not the {IMAGES_PER_SUBCATEGORY} '
                             f'of fLoc at {COMMIT}. Next step: delete the folder and load again.')
        for file in files:
            number = int(re.fullmatch(rf'{subcategory}-(\d+)', file.stem).group(1))
            rows.append({'stimulus_id': file.stem, 'subcategory': subcategory, 'category': category,
                         'number': number, 'image_path': str(file)})
            paths[file.stem] = str(file)
    stimuli = StimulusSet(rows)
    stimuli.stimulus_paths = paths
    stimuli.identifier = f'fLoc-{COMMIT[:8]}'
    return stimuli
