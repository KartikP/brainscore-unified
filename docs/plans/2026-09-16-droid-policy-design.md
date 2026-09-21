# Connect a DROID policy without changing the model contract

Use the existing action provider extension to connect an OpenPI-style policy.
The provider lives in the robotics harness; core continues to receive an
EnvironmentStep and return an EnvironmentResponse.

A single action per observation is the default for recorded-trajectory analysis.
A caller may explicitly choose a longer action horizon. Validate the whole
predicted chunk before keeping it, clear queued actions at episode boundaries,
and require an explicit policy reset callback. Do not silently clip actions,
binarize the gripper, infer units, or claim that resetting a buffer resets a
stochastic model. The caller declares these choices and checkpoint provenance.

The official OpenPI DROID example uses seven joint velocity commands plus one
gripper position, at 15 Hz. This differs from the seven-value Cartesian action
in the sampled RLDS data. Select named action_dict fields explicitly. Retain
three cameras in the recorded observation but send the policy its declared
external camera and wrist camera. Keep demonstration targets outside the request.

Validate the mapping on an actual public DROID episode, with records and offline
replay. A deterministic test policy establishes transport and lifecycle behavior
only. A trained checkpoint run is a separate gate. OpenPI currently documents
Ubuntu/NVIDIA inference; no new paid GPU run is authorized.

Alternatives: a robot-control loop would add hardware obligations; training a new
policy would test a different research question. Neither is needed to verify this
integration. A separate policy process can later isolate OpenPI dependencies from
the scientific scoring environment.

Sources:
- https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py
- https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/droid_policy.py
- https://droid-dataset.github.io/droid/the-droid-dataset.html
