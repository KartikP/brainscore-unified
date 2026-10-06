# Saved notebook figures

Keep experiment setup in notebooks and longer plotting code here. Images are displayed directly from `notebooks/assets/`, so readers can see them without running an evaluation.

Run from the repository root with the notebook dependencies installed:

```bash
# Run notebook 15's deterministic examples and render their measured timings.
python notebooks/figure_sources/render_streaming_figures.py

# Render notebook 16's stored real-data scores. No model inference required.
python notebooks/figure_sources/render_whole_brain_figures.py

# Optional: repeat the full scientific evaluation before rendering.
export BRAINSCORE_LEBEL_PICKLE=/path/to/assembly_lebel_uts03.pkl
python notebooks/figure_sources/measure_whole_brain.py
python notebooks/figure_sources/render_whole_brain_figures.py
```

The LeBel evaluation requires the [public source assembly](https://github.com/GT-LIT-Lab/litcoder_core#2-quick-setup-from-a-prepackaged-assembly-lebel), GPT-2 weights, and sufficient memory. The figure renderer may download the standard cortical surface on first use.
