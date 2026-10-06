# Measured story-listening predictions

These assets show GPT-2 predicting one participant's (UTS03) fMRI responses during 25 stories. They contain measured benchmark scores, not illustrative activity.

- `cortical_predictions.png`: held-out Pearson correlation at all 20,484 cortical locations, shown from four viewpoints.
- `score_distribution.png`: the distribution of those correlations.
- `gpt2_lebel_scores.npz`: per-location correlations and their original target indices.
- `measurements.json`: result, dataset checksum, model/layer, environment, timing, and source revision.

## Saved run

October 6, 2026: median r **0.1013**, mean r 0.1134; all 20,484 scores are finite. GPT-2 inference used Apple MPS; regression ran locally on the CPU. Evaluation and export took 203 seconds after downloading the inputs.

## Protocol

Registered model `gpt2`, layer `h.11`; registered benchmark `LeBel2023-UTS03-encoding`. Word features use up to 32 words of context, are resampled onto the two-second fMRI grid, and enter ridge regression with four time delays. Five folds hold out complete stories; each fold selects its penalty using training stories only. Scores are raw correlations, without a noise ceiling or an anatomical mask.

The maps use the documented fsaverage5 surface order: 10,242 left-hemisphere locations followed by 10,242 right-hemisphere locations. All views share a symmetric color scale. Negative correlations are retained, and values are not averaged into anatomical parcels. The surface renderer shades mesh faces from their vertices.

One participant cannot establish generalization across people. A high score means agreement between predicted and measured patterns; it does not establish a region's function.

## Reproduce

Obtain the approximately 3 GB assembly from the [source repository's download instructions](https://github.com/GT-LIT-Lab/litcoder_core#2-quick-setup-from-a-prepackaged-assembly-lebel). The raw recordings and model weights are not included here.

From the repository root, in the coordinated notebook environment:

```bash
export BRAINSCORE_LEBEL_PICKLE=/path/to/assembly_lebel_uts03.pkl
python notebooks/figure_sources/measure_whole_brain.py
python notebooks/figure_sources/render_whole_brain_figures.py
```

To redraw the bundled measurements, run only the second command. The notebook's live smoke example scores a 2,000-location subset and does not overwrite these full-cortex figures.
