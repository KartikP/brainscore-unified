# Rajalingham2018 2-AFC trial tables

`trials.csv` holds the (test image, distractor) conditions with the human choice
rate, and `tokens.csv` the token image of each object. The scoring library
`brainscore.tools.rajalingham2018_2afc` reads them; the human responses and the
images come from the vision datasets `Rajalingham2018.public` and
`objectome.public`.

## Rebuild

### rajalingham2018-2afc

- Source: the Brain-Score vision datasets Rajalingham2018.public (human 2-AFC trials of Rajalingham et al. 2018) and objectome.public (the images); licence as the vision plugins state it.
- Access: public, no account: the hosted vision files download with no account.
- Needs: brainscore_vision; a small download; under a minute.
- Command: `python -m brainscore.data package rajalingham2018-2afc`. Source folder: none: the script loads the two vision datasets through brainscore_vision.
- Writes: `trials.csv` (one row per test image and distractor) and `tokens.csv` (one token image per object). Use them: the scoring tool reads only `trials.csv` and `tokens.csv` in the plugin folder `brainscore/data/rajalingham2018_2afc/`; equal sha1 values mean the rebuilt files are those files, so there is nothing to put elsewhere.
- Check: sha1 of `trials.csv` is `71d7317315c7eb310bf25489d81cdebfdd2cdafd`; sha1 of `tokens.csv` is `18f219fd4aa96567377b196b8f165b5c6996bc59`
