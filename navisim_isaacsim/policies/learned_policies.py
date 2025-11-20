# policies/learned_policies.py

"""Wrappers for learned policies from different RL frameworks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path

import torch
import gymnasium as gym

from .base_policy import BasePolicy, PolicyConfig


@dataclass
class LearnedPolicyConfig(PolicyConfig):
    """Configuration for learned policies."""

    checkpoint_path: str = ""  # Path to model checkpoint
    deterministic: bool = True  # Use deterministic actions


class RslRlPolicy(BasePolicy):
    """
    Wrapper for RSL-RL trained policies.

    Loads and runs policies trained with RSL-RL framework.
    """

    def __init__(self, config: LearnedPolicyConfig, num_envs: int,
                 observation_space: Any, action_space: Any):
        super().__init__(config, num_envs, observation_space, action_space)

        self.checkpoint_path = config.checkpoint_path
        self.actor = None

        if self.checkpoint_path:
            self.load(self.checkpoint_path)

    def load(self, path: str) -> None:
        """Load RSL-RL policy from checkpoint."""
        from isaaclab_rl.rsl_rl.runners import OnPolicyRunner

        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        print(f"[RslRlPolicy] Loading checkpoint from: {path}")

        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Extract actor network
        if "model_state_dict" in checkpoint:
            self.actor = checkpoint["model_state_dict"]["actor"]
        elif "actor" in checkpoint:
            self.actor = checkpoint["actor"]
        else:
            raise ValueError("Could not find actor in checkpoint")

        # Move to device
        if isinstance(self.actor, torch.nn.Module):
            self.actor.to(self.device)
            self.actor.eval()

        print(f"[RslRlPolicy] Successfully loaded policy")

    def compute_action(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute actions using loaded RSL-RL actor."""
        if self.actor is None:
            raise RuntimeError("Policy not loaded. Call load() first or provide checkpoint_path.")

        obs = observations["policy"]

        with torch.no_grad():
            if isinstance(self.actor, torch.nn.Module):
                actions = self.actor(obs)
            else:
                # If actor is state dict, need to rebuild network
                raise NotImplementedError("Actor state dict loading not yet implemented")

        return actions

    def save(self, path: str) -> None:
        """Save policy checkpoint."""
        if self.actor is not None:
            torch.save({
                "actor": self.actor,
                "config": self.config,
            }, path)
            print(f"[RslRlPolicy] Saved to: {path}")


class Sb3Policy(BasePolicy):
    """
    Wrapper for Stable-Baselines3 trained policies.

    Loads and runs policies trained with SB3 framework.
    """

    def __init__(self, config: LearnedPolicyConfig, num_envs: int,
                 observation_space: Any, action_space: Any):
        super().__init__(config, num_envs, observation_space, action_space)

        self.checkpoint_path = config.checkpoint_path
        self.model = None

        if self.checkpoint_path:
            self.load(self.checkpoint_path)

    def load(self, path: str) -> None:
        """Load SB3 policy from checkpoint."""
        from stable_baselines3 import PPO

        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        print(f"[Sb3Policy] Loading checkpoint from: {path}")

        # Load SB3 model
        self.model = PPO.load(path, device=self.device)
        self.model.policy.eval()

        print(f"[Sb3Policy] Successfully loaded policy")

    def compute_action(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute actions using loaded SB3 policy."""
        if self.model is None:
            raise RuntimeError("Policy not loaded. Call load() first or provide checkpoint_path.")

        # SB3 expects numpy arrays, convert from torch
        obs_numpy = observations["policy"].cpu().numpy()

        # Get actions (deterministic or stochastic)
        actions_numpy, _ = self.model.predict(obs_numpy, deterministic=self.config.deterministic)

        # Convert back to torch tensor
        actions = torch.from_numpy(actions_numpy).to(self.device)

        return actions

    def save(self, path: str) -> None:
        """Save policy checkpoint."""
        if self.model is not None:
            self.model.save(path)
            print(f"[Sb3Policy] Saved to: {path}")


@dataclass
class HybridPolicyConfig(PolicyConfig):
    """Configuration for hybrid policies."""

    learned_policy_path: str = ""
    learned_policy_type: str = "rsl_rl"  # "rsl_rl" or "sb3"
    fallback_to_scripted: bool = True
    confidence_threshold: float = 0.8


class HybridPolicy(BasePolicy):
    """
    Hybrid policy combining learned and scripted behaviors.

    Switches between learned policy and scripted fallback based on confidence.
    """

    def __init__(self, config: HybridPolicyConfig, num_envs: int,
                 observation_space: Any, action_space: Any):
        super().__init__(config, num_envs, observation_space, action_space)

        # Load learned policy
        if config.learned_policy_type == "rsl_rl":
            learned_config = LearnedPolicyConfig(
                name="learned_component",
                checkpoint_path=config.learned_policy_path,
                device=config.device,
                deterministic=config.deterministic,
            )
            self.learned_policy = RslRlPolicy(learned_config, num_envs, observation_space, action_space)
        elif config.learned_policy_type == "sb3":
            learned_config = LearnedPolicyConfig(
                name="learned_component",
                checkpoint_path=config.learned_policy_path,
                device=config.device,
                deterministic=config.deterministic,
            )
            self.learned_policy = Sb3Policy(learned_config, num_envs, observation_space, action_space)
        else:
            raise ValueError(f"Unknown policy type: {config.learned_policy_type}")

        # Scripted fallback
        from .scripted_policies import ForwardPolicy, ScriptedPolicyConfig
        scripted_config = ScriptedPolicyConfig(name="scripted_fallback", device=config.device)
        self.fallback_policy = ForwardPolicy(scripted_config, num_envs, observation_space, action_space)

        self.use_fallback = config.fallback_to_scripted
        self.confidence_threshold = config.confidence_threshold

    def compute_action(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute actions using hybrid approach.

        Uses learned policy by default, falls back to scripted when uncertain.
        """
        try:
            # Try learned policy
            actions = self.learned_policy.compute_action(observations)

            # TODO: Add confidence estimation based on value function or uncertainty
            # For now, always use learned policy if available

            return actions

        except Exception as e:
            if self.use_fallback:
                print(f"[HybridPolicy] Falling back to scripted policy: {e}")
                return self.fallback_policy.compute_action(observations)
            else:
                raise
