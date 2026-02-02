# NaviSim Agent Framework

This document describes the refactored agent framework that enables pluggable RL algorithms in NaviSim.

## Overview

The NaviSim agent framework provides a unified interface for training and evaluating navigation agents using different RL libraries (RSL-RL, Stable-Baselines3, skrl, etc.). This architecture enables:

- **Extensibility**: Easy addition of new RL frameworks
- **Consistency**: Unified API across all agents
- **Maintainability**: Clear separation of concerns
- **Flexibility**: Switch between algorithms with minimal code changes

## Architecture

```
navisim_isaacsim/
├── navisim_lab/
│   ├── agents/                    # Agent implementations
│   │   ├── base_agent.py          # Abstract base interface
│   │   ├── __init__.py            # Registry system
│   │   └── rsl_rl/
│   │       ├── rsl_rl_agent.py    # RSL-RL implementation
│   │       └── __init__.py
│   │
│   ├── envs/                      # Environment implementations
│   │   ├── base/
│   │   │   ├── base_nav_env.py    # Base navigation environment
│   │   │   └── __init__.py
│   │   └── warehouse/
│   │       ├── warehouse_env.py   # Warehouse task (inherits from base)
│   │       ├── warehouse_env_cfg.py
│   │       └── warehouse_scene_cfg.py
│   │
│   └── configs/
│       └── rsl_rl/
│           └── ppo_warehouse_jetbot.yaml
│
└── scripts/
    ├── train.py                   # Unified training script
    ├── play.py                    # Unified inference script
    └── rsl_rl/                    # Legacy scripts (backwards compat)
        ├── train.py
        └── play.py
```

## Core Components

### 1. BaseRLAgent Interface

All agent implementations must inherit from `BaseRLAgent` and implement:

```python
from navisim_lab.agents.base_agent import BaseRLAgent

class MyAgent(BaseRLAgent):
    def setup(self, env, config: dict) -> "MyAgent":
        """Initialize agent with environment and config"""
        pass

    def train(self, num_iterations: int, log_dir: Path) -> None:
        """Train the agent"""
        pass

    def load(self, checkpoint_path: Path) -> None:
        """Load checkpoint"""
        pass

    def save(self, save_path: Path) -> None:
        """Save checkpoint"""
        pass

    def get_policy(self, deterministic: bool = True) -> Callable:
        """Get inference policy"""
        pass

    @property
    def requires_env_wrapper(self) -> bool:
        """Whether environment wrapping is needed"""
        pass

    def wrap_env(self, env) -> Any:
        """Apply agent-specific wrapper"""
        pass

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, config: dict) -> "MyAgent":
        """Create agent from checkpoint"""
        pass
```

### 2. Agent Registry

Agents are registered using a decorator:

```python
from navisim_lab.agents import register_agent

@register_agent("my_agent")
class MyAgent(BaseRLAgent):
    ...
```

Create agents using the factory:

```python
from navisim_lab.agents import create_agent, list_agents

# List available agents
print(list_agents())  # ['rsl_rl', 'sb3', ...]

# Create an agent
agent = create_agent("rsl_rl", algorithm="ppo")
```

### 3. BaseNavigationEnv

Navigation environments inherit from `BaseNavigationEnv` which provides:

- Standard scene setup
- Common navigation state management (position tracking, goals)
- Shared termination logic (timeout, collision detection hooks)
- Utility methods for reward computation

**Required implementations:**
- `_compute_navigation_reward()`: Task-specific reward logic
- `_check_termination_conditions()`: Early termination conditions
- `_pre_physics_step()`: Action processing
- `_get_observations()`: Observation computation

**Example:**

```python
from navisim_lab.envs.base import BaseNavigationEnv

class MyEnv(BaseNavigationEnv):
    def _compute_navigation_reward(self) -> torch.Tensor:
        # Compute task-specific reward
        curr_pos = self.robot.data.root_state_w[:, :3]
        progress = self._compute_forward_progress(curr_pos, axis=0)
        return progress * self.cfg.reward_weight

    def _check_termination_conditions(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check for collision or goal reached
        terminated = self.collision_flags | self.goal_reached
        truncated = torch.zeros_like(terminated)
        return terminated, truncated
```

## Usage

### Training

**Unified training script:**

```bash
# Train with RSL-RL PPO
python scripts/train.py \
    --task Navisim-Warehouse-Jetbot-v0 \
    --agent rsl_rl \
    --algorithm ppo \
    --config navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml \
    --num_envs 64 \
    --max_iterations 1000

# Train headless for faster training
python scripts/train.py \
    --task Navisim-Warehouse-Jetbot-v0 \
    --agent rsl_rl \
    --config navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml \
    --headless
```

**Using legacy RSL-RL script (backwards compatible):**

```bash
python scripts/rsl_rl/train.py \
    --task Navisim-Warehouse-Jetbot-v0 \
    --agent_cfg navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml \
    --num_envs 64
```

### Inference/Evaluation

**Unified play script:**

```bash
# Auto-detect agent type from checkpoint
python scripts/play.py \
    --task Navisim-Warehouse-Jetbot-v0 \
    --checkpoint logs/Navisim-Warehouse-Jetbot-v0/rsl_rl_ppo/model_100.pt \
    --config navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml \
    --num_steps 5000

# Specify agent explicitly
python scripts/play.py \
    --task Navisim-Warehouse-Jetbot-v0 \
    --agent rsl_rl \
    --checkpoint logs/model.pt \
    --config navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml \
    --num_envs 1 \
    --headless
```

**Using legacy RSL-RL script:**

```bash
python scripts/rsl_rl/play.py \
    --checkpoint logs/model.pt \
    --agent_cfg navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml
```

## Adding New Agents

### 1. Implement BaseRLAgent

Create a new file: `navisim_lab/agents/my_framework/my_agent.py`

```python
from navisim_lab.agents.base_agent import BaseRLAgent
from navisim_lab.agents import register_agent

@register_agent("my_framework")
class MyFrameworkAgent(BaseRLAgent):
    def __init__(self, algorithm: str = "ppo"):
        self.algorithm = algorithm
        self.model = None
        self.env = None

    def setup(self, env, config: dict) -> "MyFrameworkAgent":
        # Initialize your framework
        self.env = env
        self.model = MyFramework.load(config)
        return self

    def train(self, num_iterations: int, log_dir: Path):
        # Training logic
        self.model.learn(total_timesteps=num_iterations * 1000)
        self.model.save(log_dir / "final_model")

    def load(self, checkpoint_path: Path):
        self.model = MyFramework.load(str(checkpoint_path))

    def save(self, save_path: Path):
        self.model.save(str(save_path))

    def get_policy(self, deterministic: bool = True):
        def policy(obs):
            return self.model.predict(obs, deterministic=deterministic)
        return policy

    @property
    def requires_env_wrapper(self) -> bool:
        return True  # If your framework needs wrapping

    def wrap_env(self, env):
        return MyFrameworkWrapper(env)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, config: dict):
        agent = cls()
        agent.config = config
        agent.load(checkpoint_path)
        return agent
```

### 2. Register in `__init__.py`

Create `navisim_lab/agents/my_framework/__init__.py`:

```python
from .my_agent import MyFrameworkAgent

__all__ = ["MyFrameworkAgent"]
```

### 3. Update main registry

Edit `navisim_lab/agents/__init__.py` to import your agent:

```python
# Add to the imports at the bottom
try:
    from .my_framework import *  # noqa: F401, F403
except ImportError:
    pass  # Framework not installed
```

### 4. Create config template

Create `navisim_lab/configs/my_framework/ppo.yaml`:

```yaml
seed: 42
device: "cuda:0"
max_iterations: 1000

# Framework-specific parameters
learning_rate: 0.0003
batch_size: 64
# ... etc
```

### 5. Use it!

```bash
python scripts/train.py \
    --task Navisim-Warehouse-Jetbot-v0 \
    --agent my_framework \
    --algorithm ppo \
    --config navisim_lab/configs/my_framework/ppo.yaml
```

## Adding New Environments

### 1. Inherit from BaseNavigationEnv

Create a new file: `navisim_lab/envs/my_task/my_env.py`

```python
from navisim_lab.envs.base import BaseNavigationEnv
import torch

class MyEnv(BaseNavigationEnv):
    cfg: MyEnvCfg

    def _initialize_task_state(self):
        super()._initialize_task_state()
        # Add custom state tracking
        self.custom_state = torch.zeros(self.num_envs, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Process actions
        self.robot.set_joint_velocity_target(actions * self.cfg.action_scale)

    def _apply_action(self):
        self.scene.write_data_to_sim()

    def _get_observations(self) -> dict:
        obs = self.robot.data.root_state_w.clone()
        return {"policy": obs}

    def _compute_navigation_reward(self) -> torch.Tensor:
        # Custom reward logic
        curr_pos = self.robot.data.root_state_w[:, :3]
        reward = torch.norm(curr_pos - self.goal_positions, dim=1)
        return -reward  # Negative distance to goal

    def _check_termination_conditions(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Custom termination logic
        goal_reached = torch.norm(
            self.robot.data.root_state_w[:, :3] - self.goal_positions, dim=1
        ) < 0.5
        terminated = goal_reached | self.collision_flags
        truncated = torch.zeros_like(terminated)
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset robot state
        # ... reset logic ...

        # Reset base navigation state
        self._reset_navigation_state(env_ids)

        super()._reset_idx(env_ids)
```

### 2. Register with Gymnasium

Create `navisim_lab/tasks/my_task/__init__.py`:

```python
import gymnasium as gym

gym.register(
    id="Navisim-MyTask-v0",
    entry_point="navisim_lab.envs.my_task.my_env:MyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "navisim_lab.envs.my_task.my_env_cfg:MyEnvCfg",
    },
)
```

### 3. Use it!

```bash
python scripts/train.py --task Navisim-MyTask-v0 --agent rsl_rl --config my_config.yaml
```

## Configuration Structure

All agent configs follow a common structure:

```yaml
# Training parameters
seed: 42
device: "cuda:0"
num_steps_per_env: 32
max_iterations: 1000
save_interval: 50

# Experiment naming
experiment_name: "my_experiment"
run_name: "ppo_v1"

# Observation/critic groups (for RSL-RL)
obs_groups:
  policy: ["policy"]
  critic: ["policy"]

# Policy architecture
policy:
  class_name: "ActorCritic"
  actor_hidden_dims: [256, 256]
  critic_hidden_dims: [256, 256]
  activation: "elu"
  init_noise_std: 1.0

# Algorithm hyperparameters
algorithm:
  class_name: "PPO"
  learning_rate: 0.0003
  num_learning_epochs: 5
  num_mini_batches: 4
  gamma: 0.99
  lam: 0.95
  entropy_coef: 0.01
  # ... algorithm-specific params
```

## Migration Guide

### From Legacy RSL-RL Scripts

**Old way:**
```bash
python scripts/rsl_rl/train.py --task MyTask-v0 --agent_cfg config.yaml
```

**New way (unified):**
```bash
python scripts/train.py --task MyTask-v0 --agent rsl_rl --config config.yaml
```

**Old way:**
```bash
python scripts/rsl_rl/play.py --checkpoint model.pt --agent_cfg config.yaml
```

**New way (unified):**
```bash
python scripts/play.py --task MyTask-v0 --checkpoint model.pt --config config.yaml
```

### Updating Existing Environments

If you have an environment that directly inherits from `DirectRLEnv`:

1. Change inheritance to `BaseNavigationEnv`
2. Rename `_get_rewards()` to `_compute_navigation_reward()`
3. Implement `_check_termination_conditions()` (extract from old `_get_dones()`)
4. Call `self._reset_navigation_state(env_ids)` in `_reset_idx()`

**Before:**
```python
from isaaclab.envs import DirectRLEnv

class MyEnv(DirectRLEnv):
    def _get_rewards(self) -> torch.Tensor:
        return self.compute_reward()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length
        terminated = self.collision_flags
        return terminated, timeout
```

**After:**
```python
from navisim_lab.envs.base import BaseNavigationEnv

class MyEnv(BaseNavigationEnv):
    def _compute_navigation_reward(self) -> torch.Tensor:
        return self.compute_reward()

    def _check_termination_conditions(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self.collision_flags
        truncated = torch.zeros_like(terminated)
        return terminated, truncated  # Timeout handled by base class
```

## Benefits

1. **Extensibility**: Add new RL frameworks by implementing a single interface
2. **Consistency**: All agents follow the same API
3. **Maintainability**: Clear separation between agent, environment, and script logic
4. **Code Reuse**: BaseNavigationEnv reduces duplication across tasks
5. **Backwards Compatibility**: Legacy scripts remain functional
6. **Future-Proof**: Easy to experiment with new algorithms

## Troubleshooting

### Agent not found
```
ValueError: Agent 'my_agent' not found in registry
```
**Solution**: Ensure your agent is imported in `navisim_lab/agents/__init__.py`

### Environment wrapper issues
```
AttributeError: 'MyEnv' object has no attribute 'num_actions'
```
**Solution**: Ensure `agent.requires_env_wrapper` is True and wrapper is applied

### Checkpoint incompatible
```
ValueError: Cannot auto-detect agent type from checkpoint
```
**Solution**: Specify `--agent` explicitly when using `play.py`

## Future Extensions

Planned agent implementations:
- [ ] Stable-Baselines3 (PPO, SAC, TD3)
- [ ] skrl (multi-algorithm support)
- [ ] CleanRL (minimalist implementations)
- [ ] Custom NaviSim algorithms

Planned environment features:
- [ ] Collision detection in BaseNavigationEnv
- [ ] Goal-conditioned navigation base class
- [ ] Multi-robot navigation environment
- [ ] Dynamic obstacle environments
