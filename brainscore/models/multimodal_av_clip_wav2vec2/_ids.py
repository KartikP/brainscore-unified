"""Registry identifiers for the CLIP frames + Wav2Vec2 A+V combos.

Torch-free so the plugin __init__ enumerates variants without importing
the heavy model.py (criterion-2: keep `import brainscore` light)."""
SUPPORTED_IDENTIFIERS = (
    'clip-wav2vec2',
    'random-clip-wav2vec2',
    'clip-random-wav2vec2',
    'random-clip-random-wav2vec2',
)
