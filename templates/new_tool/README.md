# Template: a new tool

A tool is equipment attached to an experiment: it observes the run (inputs, outputs,
layer activity) or changes the subject through a declared channel. This template observes
the outputs and returns the mean response of each recorded region. See
`docs/guides/extend.md` ("A new tool") for the event keys and `docs/guides/experiments.md`
for experiments.

## Use it

1. Copy `tool.py` to `brainscore/tools/<your_name>.py`.
2. Edit it: rename `YourTool`, replace `on_event` with your analysis.
3. Add one line to `_TOOLS` in `brainscore/tools/_registry.py`:
   `'<your-id>': ('brainscore.tools.<your_name>', 'YourTool', False)`.
   The last value says whether `load_tool(..., artifact=...)` loads a saved, fitted tool.
   The module is imported only when `load_tool` asks for it.
4. Move `test_tool.py` to `tests/test_<your_name>.py`, adapt it and run it.

A tool in your own package (outside Brain-Score) is added with
`register_tool('<your-id>', YourTool)` from `brainscore.tools`, before `load_tool`.

From the `templates/` folder:

```python
import os
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from brainscore import BrainScoreModel
from brainscore.tools import load_tool, register_tool
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from new_tool import tool

register_tool('your-tool', tool.YourTool)          # the template is not in the tool table
print(type(load_tool('your-tool')).__name__)        # YourTool

torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU()).eval()
subject = BrainScoreModel(identifier='tiny-cnn', model=net, region_layer_map={'IT': '1'},
                          extractor={'vision': PytorchWrapper(
                              identifier='tiny-cnn', model=net,
                              preprocessing=lambda paths: load_preprocess_images(paths, image_size=16))})
workspace = tempfile.TemporaryDirectory()      # removed when the script ends
folder = workspace.name
paths = {f's{i}': os.path.join(folder, f'{i}.png') for i in range(4)}
for i, path in enumerate(paths.values()):
    Image.fromarray(np.random.RandomState(i).randint(0, 256, (16, 16, 3), dtype=np.uint8)).save(path)
stimuli = StimulusSet(pd.DataFrame({'stimulus_id': list(paths), 'image_path': list(paths.values())}))
stimuli.stimulus_paths = paths
stimuli.identifier = 'tiny-images'
print(sorted(tool.region_means(subject, stimuli, ['IT'])))   # ['IT']
```
