"""Registry identifiers for the V-JEPA v1 video + Wav2Vec2 A+V combos.

Torch-free so the plugin __init__ enumerates variants without importing
the heavy model.py (criterion-2: keep `import brainscore` light)."""
SUPPORTED_IDENTIFIERS = (
    'vjepa1-wav2vec2',
    'random-vjepa1-wav2vec2',
    'vjepa1-random-wav2vec2',
    'random-vjepa1-random-wav2vec2',
)
