# Source and packaging receipts

These files describe the scene-1 pilot; they do not stand in for the full dataset.

- source-manifest.json: exact upstream revisions, URLs, byte sizes, SHA-256s
  and archive CRCs for every retained source file used by the pilot.
- prepared-manifest.json: transformation settings and hashes for 28 lossless
  clips, stimulus metadata and the human-response table.
- source-validation.json: all 40 archived GT vectors matched to their scene
  identities for each of three participants, and the pilot rebuild check.
- [pilot-preview.png](pilot-preview.png): inspected frames 4 and 8 from the
  seven unflipped pilot clips, resized only for this visual preview.

The rebuilt pilot has the same prepared manifest, including every clip/table
hash, after the renderer's source-edge background handling was made explicit.
The selected scene does not reach a source edge; a synthetic edge-case test
checks gray-background rendering for the remaining source locations.

Model execution, response-format policy and test receipts are in
examples/sun2025/validation. Source acquisition and conversion scripts are in
the parent data_packaging directory; the dataset registry belongs to this data
plugin, independently of any experiment or benchmark.
