# Qualification before the interpretability tools PR

Qualify mechanics on pinned, pretrained GPT-2 and CLIP text encoders, using
cached weights and CPU float32. Verify each loader file against the pinned
Hugging Face listing. Reuse the shipped TextWrapper, BrainScoreModel sessions,
tool loader, experiments and direct attachment.

Independent native forwards supply expected activations. Test recording-only
identity, selected-unit ablation/scaling, exact donor patches, signed steering,
LEACE/reference parity, probe/reference parity, composition and restoration.
Check cleanup after interruption, unchanged parameters/RNG, record integrity,
public-call observation and explicit unavailability of reasoning from encoders.
Save fitted artifacts, records and a report that also survives failures.

Fit on explicit discovery sentences and evaluate on disjoint sentences. Report
held-out color-label probe accuracy and a shuffled-label control descriptively;
do not gate on accuracy, steering semantics or held-out erasure effectiveness.
These checks qualify numerical behavior and integration, not a true circuit,
human validity, benchmark parity or the faithfulness of generated explanations.

Keep the two trained checks opt-in and fail if staged assets are absent. Add
fast fixture tests of the qualification procedure, including deliberately wrong
outputs and failed reports, to the normal offline suite. Include reproducible
commands, pinned identities and observed numerical results in the PR evidence.
