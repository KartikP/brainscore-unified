# L4 FP32 qualification profile

The user chose FP32 performance with a documented numerical error budget.
Keep `historical-v1` and the CPU profile intact. Add a separate profile for the
observed NVIDIA L4 stack; do not label other CUDA devices qualified by it.

The four fixed GPT-2/Pereira cases use the same recorded activations for each
experiment, with different linear/ridge metrics. Apply the existing 16 effective
FP32 ULP activation rule to both protocols on L4. This keeps the historical
linear activation budget; it is below the prospective 32-ULP review ceiling
recorded before observing L4 language results. Use the already declared 1e-5
normalized/raw-scaled score target and 5e-6 mean signed target per experiment
pair. Adapter checks and vision limits remain unchanged.

Require explicit CUDA FP32, the exact calibrated GPT-2/tokenizer file set, and
actual device name NVIDIA L4. Keep the original defaults. A passing profile is
scoped to these fixed cases and recorded software/driver configuration, not a
universal error bound for new experiments or an approval of general availability.

Validation: retain the historical run; audit all token/position/mask inputs,
repeatability, and worst-passage precision replays; compare diagnostic activation
hashes with independently scored metric inputs. Run the four real language cases
under the new profile in a separate installed environment. Test device rejection,
activation/score boundaries, paired checks and adapter invariants. Reuse unchanged
vision case evidence with explicit per-case provenance rather than recomputing
expensive fits after a validation-only change.
