from __future__ import annotations

from pathlib import Path

from isaaclab_rl.rsl_rl.rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab.utils.io import load_yaml  # Isaac Lab helper used across configs

def _to_policy_cfg(d: dict) -> RslRlPpoActorCriticCfg:
    return RslRlPpoActorCriticCfg(**d)


def _to_algo_cfg(d: dict) -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(**d)


def make_cfg(path) -> RslRlOnPolicyRunnerCfg:
    raw = load_yaml(path)

    # build nested config objects expected by RSL-RL runner
    raw["policy"] = _to_policy_cfg(raw["policy"])
    raw["algorithm"] = _to_algo_cfg(raw["algorithm"])

    return RslRlOnPolicyRunnerCfg(**raw)


# Expose an INSTANCE (gym entry points can be string/class/instance) :contentReference[oaicite:3]{index=3}
PPO_WAREHOUSE_JETBOT_CFG = make_cfg(path = Path(__file__).with_name("ppo_warehouse_jetbot.yaml"))
