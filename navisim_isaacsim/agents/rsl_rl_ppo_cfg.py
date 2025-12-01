# agents/rsl_rl_ppo_cfg.py

"""RSL-RL PPO configuration for NaviSim navigation environment."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.runners import OnPolicyRunnerCfg


@configclass
class NavisimNavigationPPORunnerCfg(OnPolicyRunnerCfg):
    """Configuration for RSL-RL PPO runner for NaviSim navigation."""

    # Environment settings
    num_steps_per_env = 24  # Number of steps per environment per iteration
    max_iterations = 3000   # Total number of training iterations

    # Saving and logging
    save_interval = 50      # Save model every N iterations
    experiment_name = "navisim_navigation"
    run_name = ""           # Can be overridden via CLI

    # Empirical length calculation
    empirical_normalization = False

    # Policy configuration
    policy = OnPolicyRunnerCfg.PolicyCfg(
        # Network architecture
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],   # Actor MLP layers
        critic_hidden_dims=[256, 256, 128],  # Critic MLP layers
        activation="elu",                     # Activation function
    )

    # PPO algorithm parameters
    algorithm = OnPolicyRunnerCfg.AlgorithmCfg(
        # Training hyperparameters
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,              # PPO clipping parameter
        entropy_coef=0.005,          # Entropy bonus coefficient
        num_learning_epochs=5,       # Number of epochs per iteration
        num_mini_batches=4,          # Mini-batches per epoch
        learning_rate=3.0e-4,        # Adam learning rate
        schedule="adaptive",         # Learning rate schedule
        gamma=0.99,                  # Discount factor
        lam=0.95,                    # GAE lambda
        desired_kl=0.01,             # Target KL divergence
        max_grad_norm=1.0,           # Gradient clipping
    )
