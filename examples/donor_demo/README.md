# Donor demo (2026-10-21): watch, record, predict, compare

Scripts behind the "Neuroscientific Toolbox for AI" TV poster. Experiment A runs the
registered `vjepa1-wav2vec2` model through `BrainScoreModel.process()` on BOLDMoments
(Lahner 2024) and compares video, soundtrack and combined ridge readouts under one nested-CV
protocol; Experiment B does the video / audio / text comparison on the Friends recordings
(Algonauts 2025 caches). The remaining scripts render the measured replay and build the deck.

EC2 only for the two experiment scripts (see docstrings). Outputs and the methods document
live outside the repository (`/Volumes/Samsung9100/Brain-Score-Donor-Demo`).

## v8 (2026-10-22): scan-by-scan replay and the lesion-protocol slide

`v8/` holds the second iteration: `ec2/donor_timeresolved_replay.py` (per-scan encoding on the BOLDMoments single-trial series), `ec2/donor_lesion_protocol.py` with `ec2/sweep_frac.sh` and `ec2/chain_sweeps.sh` (five-session lesion protocol on Qwen2.5-VL 3B/7B/32B through `BrainScoreModel.process()`, `process(StateChange)` and `reset()`, seven decoding agents per size, lesion-size sweep), the two replay renderers (`render_friends_replay.py`, `render_tr_replay.py`), `analyze_lesion.py`, and the two-slide deck pipeline (`build_poster_v8.py`, `render_slide_png.py`, `compose_outputs_v8.sh`, `make_notes_v8.py`, `write_results_doc_v8.py`, `finalize_v8.sh`). `lesion_interpretation.txt` is the written reading of the lesion results, appended to the slide notes at build time. Results and data live with the deliverable, not in this repository.
