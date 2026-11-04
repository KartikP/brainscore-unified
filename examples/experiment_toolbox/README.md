# A recording room for models

The five CPU demonstrations are part of the separately installed
[example package](../partner_tool/README.md):

```sh
python -m pip install ./examples/partner_tool
python -I -m umi_partner_tool.toolbox --out ./my-toolbox-demo
```

After installation, the second command works from any directory. Choose a new
output folder. Images, text, image-plus-text, a feedback environment, and a
recurrent subject all use the same recording and intervention tools.

Each case runs baseline, silenced, and restored conditions, then reruns saved
inputs. `summary.json` checks intervention, recovery, and replay. These are
synthetic, untrained examples; the feedback environment is not a robotics
benchmark. Replay uses saved observations and does not restart that environment.

Read [Set up an experiment](../../docs/guides/experiments.md) for configuration,
outputs, replay, and external evaluators. Copy [partner_tool.py](partner_tool.py)
to your own package to add a tool using public APIs.

## Record reasoning

The [reasoning example](reasoning.py) records two synthetic reasoning fragments
and a final answer. It uses `build_trace_subject`, `SessionProtocol`, and the
same record format; no model download or paid API call is needed.

```sh
python examples/experiment_toolbox/reasoning.py --out ./my-reasoning-demo
```

See [reasoning recording](../../docs/archive/reasoning_recording.md) for provider adapters
and the distinction between visible reasoning and hidden model computation.
