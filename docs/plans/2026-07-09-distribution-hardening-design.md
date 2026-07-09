# Distribution hardening design

## Objective and boundary

Make a cold laptop install reproducible and give a scientist one public route
from checkout to first result. Distribution work is restricted to shell,
package metadata, environment configuration, Markdown, and notebooks. Python
source remains in the concurrent correctness/API lane.

## Installation architecture

The distribution root owns environment-unified.yml and setup.sh. The script
clones or verifies all four repositories at unified-model-interface-v2, creates
a new named environment without mutating an existing one, installs the sibling
repositories editably, and runs pip/import/cache-API gates. Package metadata
shares Python 3.11, NumPy below 2, xarray 2022.3.0, sklearn 1.5.x, and
Transformers 4.57.x ranges.

## Documentation architecture

The root README is the front door. It links installation, getting started,
extension templates, notebooks, architecture, and the UMI cookbook. Legacy
domain READMEs receive migration banners. EXTENDING distinguishes model-level
constructor callables from core framework Capability classes.

## Notebook policy

The numbered 01 through 11 sequence is the only active notebook on-ramp.
Legacy and paid-API notebooks move to archive/. Laptop notebooks must be
self-contained or explicitly guard optional downloads. Illustrative notebooks
must label placeholder or descriptive scope; committed tracebacks and stale
execution order are not allowed.

## Verification

Create bs-verify from environment-unified.yml, run pip check, import brainscore,
assert DynamicCache.to_legacy_cache exists, and execute every active
laptop-safe notebook. Remove the throwaway environment afterward. Commit only
the allowed file types and never include concurrent Python-source changes.
