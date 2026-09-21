# Independent UMI tool package

Install the coordinated candidate wheel set first, then install this package:

```sh
python -m pip install ./examples/partner_tool
python -m umi_partner_tool --out /tmp/new-partner-tool-run
```

Run the second command from any directory. The demo registers new input/output
channels and a capability, attaches an independent observer, saves/replays the
measurement and verifies reset. Its synthetic amplitude calculation is an
authoring example, not a scientific benchmark.

Copy this package under a new distribution/module name for your integration.
Replace its payload validators, units and capability computation, add meaningful
conformance tests, and document the supported model/data profile. The package
uses only public imports; no edits to any Brain-Score contract are needed.
