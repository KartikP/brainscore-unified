# Package contents

Keep resources users can run with documented setup and their own authorized
inputs. Examples, notebooks, and dataset builders are not excluded by file type.
Archive machine-specific exploration before removing it.

## Decisions in this review

| Package | Kept | Changed |
| --- | --- | --- |
| Core | Runtime modules and plugin shell scripts | Internal BrainIO tests remain in source, outside the wheel |
| Vision | Runtime plugins, builders, licenses, and reusable notebooks | Exclude top-level tests/scripts; archive 14 historical notebooks and four obsolete scripts; replace personal paths in retained notebooks |
| Language | Both scientific notebooks and runtime resources | Use installed imports, write ceiling results to the current directory, clear saved outputs |
| Unified | Runtime assets and V-JEPA backbone | Include its LICENSE and NOTICE; use a portable Lahner checkpoint default |

All four wheels exclude Python bytecode. Pre-review source is preserved in each
repository on `archive/pre-packaging-review-2026-10-21`. Archiving does not erase
Git history. See vision's `docs/package_contents.md` for the archived file list.

## Checks

After building all four wheels, run:

```sh
python unified/brainscore/validation/package_contents.py \
    --wheels wheels \
    --out qualification/package-contents.json
```

The check rejects unexpected top-level packages, Python bytecode, missing
required notices/resources, and saved execution state or personal paths in
packaged notebooks. It records wheel hashes. Both CI build workflows run this
check before installation. Installed-package checks and tutorial execution follow.
Notebook inspection is a portability check, not a substitute for executing the
scientific workflow with its dataset.

## Licensing boundary

Core, vision, and language retain their MIT notices. V-JEPA v1 retains its
CC BY-NC 4.0 license and separate attribution. UMI-owned licensing and exact
third-party provenance remain tracked in #67/#8; file-presence checks do not
certify redistribution rights. The unified package has no blanket MIT declaration
until that decision is recorded. Model-submission screening is future task #68.

Package name and candidate versions are unchanged. Nothing is published.
