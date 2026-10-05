# A recording room for models

Run five CPU demonstrations with the same experiment API: images, text,
image-plus-text, a feedback-controlled toy environment, and a recurrent subject
emitting both activity and behavior. These are synthetic, untrained examples.

From the unified repository, in the coordinated UMI environment:

```sh
python examples/experiment_toolbox/run.py --out /tmp/my-toolbox-demo
```

Choose a new output folder. Each example records a baseline trial and a trial
with a module ablated, then replays the same inputs with the same intervention.
`summary.json` reports exact output agreement. The robotics example has an actual
feedback loop, but is not LIBERO or a robot-performance benchmark.

Read [Set up an experiment](../../docs/experiment_toolbox.md) for configuration,
outputs, replay, limits, and external evaluator integration. Copy
[partner_tool.py](partner_tool.py) to your own package to add a tool using only
public APIs.
