"""Data plugin for the Rajalingham2018 2-AFC scoring library (``brainscore.tools.rajalingham2018_2afc``).

``trials.csv`` holds the (image, distractor) conditions and ``tokens.csv`` the object token images.
The human responses come from the hosted ``Rajalingham2018.public`` dataset. This plugin registers nothing.
"""
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
TRIALS_CSV = FOLDER / 'trials.csv'
TOKENS_CSV = FOLDER / 'tokens.csv'
