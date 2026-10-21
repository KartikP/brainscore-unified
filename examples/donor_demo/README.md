# Donor demo (2026-10-21): watch, record, predict, compare

Scripts behind the "Neuroscientific Toolbox for AI" TV poster. Experiment A runs the
registered `vjepa1-wav2vec2` model through `BrainScoreModel.process()` on BOLDMoments
(Lahner 2024) and compares video, soundtrack and combined ridge readouts under one nested-CV
protocol; Experiment B does the video / audio / text comparison on the Friends recordings
(Algonauts 2025 caches). The remaining scripts render the measured replay and build the deck.

EC2 only for the two experiment scripts (see docstrings). Outputs and the methods document
live outside the repository (`/Volumes/Samsung9100/Brain-Score-Donor-Demo`).
