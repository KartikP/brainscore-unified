# Brain-Score UMI manual

The Unified Model Interface (UMI) is how Brain-Score runs experiments on models. This manual is short on purpose. Read the first three pages in order; use the rest when you need them.

## Start here

1. [What UMI is](01_what_is_umi.md): the "lab for models" idea, in one page and one diagram.
2. [Core concepts](02_core_concepts.md): the eight terms everything else is built from, each with a few lines of code.
3. [Class map](03_class_map.md): which class to use, which method to call, and how the abstract ideas turn into concrete code.

Then work through the [lessons](../notebooks/README.md#lessons), one concept per notebook.

## Guides

Each guide answers one "how do I" question.

| I want to... | Guide |
| --- | --- |
| Install UMI | [Install](guides/install.md) |
| Score a model on a benchmark | [Score a model](guides/score_a_model.md) |
| Connect my own network | [Wrap your model](guides/wrap_your_model.md) |
| Use a model already in Brain-Score | [Existing models](guides/existing_models.md) |
| Write a benchmark | [Write a benchmark](guides/write_a_benchmark.md) |
| Get choices or answers from a model | [Behavior](guides/behavior.md) |
| Silence or change units, or choose units by function | [Perturb and select units](guides/perturb_and_select_units.md) |
| Run a game, simulator or robot task | [Closed loop](guides/closed_loop.md) |
| Compare conditions, record activity, replay a run | [Experiments](guides/experiments.md) |
| Add my own channel, tool or capability | [Extend UMI](guides/extend.md) |

## Reference

- [Glossary](04_glossary.md): every term, one line each.
- [Feature status](reference/features.md): what is core, what is advanced, and what is still experimental; old names and their replacements.
- [Experiments](../notebooks/README.md#experiments): worked studies that use the concepts.

Older documentation, including qualification records and audits, is kept unchanged in [archive/](archive/).
