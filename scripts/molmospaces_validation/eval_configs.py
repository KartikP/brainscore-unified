"""Custom eval configs that route π0-FAST through our EnvironmentStep schema.

Used by `molmo_spaces.evaluation.eval_main` via the dotted module path:

    python molmo_spaces/evaluation/eval_main.py \\
        molmospaces_validation.eval_configs:UnifiedInterfaceEvalConfig \\
        --benchmark_dir <FrankaPickDroidMiniBench-path> \\
        --checkpoint_path <pi0_fast_droid_jointpos-path> \\
        --task_horizon_steps 500

Both this config and the stock PiPolicyEvalConfig share the SAME π0 server
(default localhost:8080); only the data path between obs and the server
differs. Three-way comparison: published leaderboard / stock-config eval /
this-config eval.
"""

from molmo_spaces.configs.policy_configs_baselines import PiPolicyConfig
from molmo_spaces.evaluation.configs.evaluation_configs import PiPolicyEvalConfig

# Local import — wrapper_adapter.py must be importable from this module's
# parent directory. We add it via PYTHONPATH at eval invocation time.
from wrapper_adapter import UnifiedInterfacePolicy


class UnifiedInterfacePolicyConfig(PiPolicyConfig):
    """Drop-in PiPolicyConfig that swaps PI_Policy → UnifiedInterfacePolicy.

    Same checkpoint path, same chunk size, same grasping config — the only
    difference is which policy class wraps the inference call. Inheriting
    rather than reimplementing keeps the config behavior identical except
    for the policy class itself.
    """
    policy_cls: type = None  # set in model_post_init

    def model_post_init(self, __context) -> None:
        # NOTE: deliberately bypass parent's model_post_init since that one
        # would set policy_cls = PI_Policy. Call grandparent's instead.
        super(PiPolicyConfig, self).model_post_init(__context)
        self.policy_cls = UnifiedInterfacePolicy


class UnifiedInterfaceEvalConfig(PiPolicyEvalConfig):
    """Eval config that runs the same eval as PiPolicyEvalConfig but uses
    UnifiedInterfacePolicy instead of PI_Policy."""
    policy_config: UnifiedInterfacePolicyConfig = UnifiedInterfacePolicyConfig()
