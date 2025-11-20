# policies/policy_registry.py

"""Policy registry for managing and instantiating different policies."""

from __future__ import annotations

from typing import Any, Dict, Type, Optional, Callable
import torch

from .base_policy import BasePolicy, PolicyConfig
from .scripted_policies import (
    ForwardPolicy,
    CirclePolicy,
    RandomPolicy,
    WaypointPolicy,
    ObstacleAvoidancePolicy,
    ScriptedPolicyConfig,
    WaypointPolicyConfig,
    ObstacleAvoidancePolicyConfig,
)
from .learned_policies import (
    RslRlPolicy,
    Sb3Policy,
    HybridPolicy,
    LearnedPolicyConfig,
    HybridPolicyConfig,
)


class PolicyRegistry:
    """
    Registry for managing navigation policies.

    Provides easy access to different policy types and configurations.
    """

    _policies: Dict[str, tuple[Type[BasePolicy], Type[PolicyConfig]]] = {}

    @classmethod
    def register(cls, name: str, policy_class: Type[BasePolicy],
                 config_class: Type[PolicyConfig]) -> None:
        """
        Register a policy with the registry.

        Args:
            name: Unique name for the policy.
            policy_class: Policy class to register.
            config_class: Configuration class for the policy.
        """
        if name in cls._policies:
            print(f"[PolicyRegistry] Warning: Overwriting existing policy '{name}'")

        cls._policies[name] = (policy_class, config_class)
        print(f"[PolicyRegistry] Registered policy: {name}")

    @classmethod
    def get(cls, name: str, config: Optional[PolicyConfig] = None,
            num_envs: int = 1, observation_space: Any = None,
            action_space: Any = None, **kwargs) -> BasePolicy:
        """
        Get a policy instance by name.

        Args:
            name: Name of the policy to instantiate.
            config: Optional configuration. If None, creates default config.
            num_envs: Number of parallel environments.
            observation_space: Environment observation space.
            action_space: Environment action space.
            **kwargs: Additional arguments passed to config constructor.

        Returns:
            Instantiated policy.

        Raises:
            KeyError: If policy name not found in registry.
        """
        if name not in cls._policies:
            raise KeyError(f"Policy '{name}' not found in registry. "
                          f"Available: {list(cls._policies.keys())}")

        policy_class, config_class = cls._policies[name]

        # Create config if not provided
        if config is None:
            config = config_class(name=name, **kwargs)

        # Instantiate policy
        policy = policy_class(config, num_envs, observation_space, action_space)

        return policy

    @classmethod
    def list_policies(cls) -> list[str]:
        """Get list of registered policy names."""
        return list(cls._policies.keys())

    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        """Get information about a registered policy."""
        if name not in cls._policies:
            raise KeyError(f"Policy '{name}' not found")

        policy_class, config_class = cls._policies[name]

        return {
            "name": name,
            "policy_class": policy_class.__name__,
            "config_class": config_class.__name__,
            "description": policy_class.__doc__ or "No description",
        }


# Register all built-in policies
def _register_builtin_policies():
    """Register all built-in policies with the registry."""
    registry = PolicyRegistry

    # Scripted policies
    registry.register("forward", ForwardPolicy, ScriptedPolicyConfig)
    registry.register("circle", CirclePolicy, ScriptedPolicyConfig)
    registry.register("random", RandomPolicy, ScriptedPolicyConfig)
    registry.register("waypoint", WaypointPolicy, WaypointPolicyConfig)
    registry.register("obstacle_avoidance", ObstacleAvoidancePolicy, ObstacleAvoidancePolicyConfig)

    # Learned policies
    registry.register("rsl_rl", RslRlPolicy, LearnedPolicyConfig)
    registry.register("sb3", Sb3Policy, LearnedPolicyConfig)
    registry.register("hybrid", HybridPolicy, HybridPolicyConfig)


# Initialize built-in policies
_register_builtin_policies()


# Convenience functions
def register_policy(name: str, policy_class: Type[BasePolicy],
                   config_class: Type[PolicyConfig]) -> None:
    """Register a custom policy (convenience function)."""
    PolicyRegistry.register(name, policy_class, config_class)


def get_policy(name: str, config: Optional[PolicyConfig] = None,
               num_envs: int = 1, observation_space: Any = None,
               action_space: Any = None, **kwargs) -> BasePolicy:
    """Get a policy instance (convenience function)."""
    return PolicyRegistry.get(name, config, num_envs, observation_space, action_space, **kwargs)


def list_policies() -> list[str]:
    """List all registered policies (convenience function)."""
    return PolicyRegistry.list_policies()
