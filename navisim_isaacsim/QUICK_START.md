# Quick Start Guide - New Agent Framework

## TL;DR

The codebase now supports multiple RL frameworks through a unified interface.

### Old Way (Still Works)
```bash
python scripts/rsl_rl/train.py --task MyTask-v0 --agent_cfg config.yaml
```

### New Way (Recommended)
```bash
python scripts/train.py --task MyTask-v0 --agent rsl_rl --config config.yaml
```

## Quick Commands

### Training
```bash
# Basic training
python scripts/train.py \
    --task Navisim-Warehouse-Jetbot-v0 \
    --agent rsl_rl \
    --config navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml

# Training with custom settings
python scripts/train.py \
    --task Navisim-Warehouse-Jetbot-v0 \
    --agent rsl_rl \
    --config navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml \
    --num_envs 128 \
    --max_iterations 2000 \
    --headless
```

### Inference
```bash
# Play with checkpoint (auto-detects agent)
python scripts/play.py \
    --task Navisim-Warehouse-Jetbot-v0 \
    --checkpoint logs/model.pt \
    --config navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml

# Specify agent explicitly
python scripts/play.py \
    --task Navisim-Warehouse-Jetbot-v0 \
    --agent rsl_rl \
    --checkpoint logs/model.pt \
    --config navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml \
    --num_steps 10000 \
    --headless
```

### Testing
```bash
# Quick syntax check (no Isaac Sim needed)
python test_imports.py

# Full integration test (requires Isaac Sim)
python scripts/smoke_test.py --num_envs 1
```

## What's New

### 1. Pluggable Agents
```python
from navisim_lab.agents import create_agent, list_agents

# See available agents
print(list_agents())  # ['rsl_rl', 'sb3', ...]

# Create any agent
agent = create_agent("rsl_rl", algorithm="ppo")
```

### 2. Base Navigation Environment
```python
from navisim_lab.envs.base import BaseNavigationEnv

class MyEnv(BaseNavigationEnv):
    def _compute_navigation_reward(self) -> torch.Tensor:
        # Your reward logic
        ...
```

### 3. Unified Scripts
- Single `train.py` for all agents
- Single `play.py` for inference
- Auto-detects agent from checkpoint
- Better logging and error handling

## Adding New Agents

### Step 1: Implement BaseRLAgent
```python
# navisim_lab/agents/my_agent/my_agent.py
from navisim_lab.agents import register_agent
from navisim_lab.agents.base_agent import BaseRLAgent

@register_agent("my_agent")
class MyAgent(BaseRLAgent):
    def setup(self, env, config): ...
    def train(self, num_iterations, log_dir): ...
    def load(self, checkpoint_path): ...
    def save(self, save_path): ...
    def get_policy(self, deterministic=True): ...
    def wrap_env(self, env): ...

    @property
    def requires_env_wrapper(self): return True

    @classmethod
    def from_checkpoint(cls, checkpoint_path, config): ...
```

### Step 2: Register
```python
# navisim_lab/agents/my_agent/__init__.py
from .my_agent import MyAgent

# navisim_lab/agents/__init__.py
try:
    from .my_agent import *  # noqa
except ImportError:
    pass
```

### Step 3: Use It
```bash
python scripts/train.py --task MyTask-v0 --agent my_agent --config my_config.yaml
```

## Common Tasks

### Switch Algorithm
```bash
# Change only the config file
python scripts/train.py --task MyTask-v0 --agent rsl_rl --config configs/sac.yaml
```

### Compare Algorithms
```bash
# Train PPO
python scripts/train.py --task MyTask-v0 --agent rsl_rl --config ppo.yaml --run_name ppo_v1

# Train SAC
python scripts/train.py --task MyTask-v0 --agent rsl_rl --config sac.yaml --run_name sac_v1

# Compare logs
ls logs/MyTask-v0/
# -> rsl_rl_ppo_v1/
# -> rsl_rl_sac_v1/
```

### Debug Environment
```bash
# Run for few iterations to test
python scripts/train.py \
    --task MyTask-v0 \
    --agent rsl_rl \
    --config config.yaml \
    --num_envs 4 \
    --max_iterations 5 \
    --headless
```

## File Locations

```
Key Files:
├── scripts/train.py                    # ← Use this for training
├── scripts/play.py                     # ← Use this for inference
├── navisim_lab/agents/                 # ← Agent implementations
├── navisim_lab/envs/base/              # ← Base environment classes
└── navisim_lab/configs/                # ← Algorithm configs

Documentation:
├── AGENT_FRAMEWORK.md                  # ← Complete guide
├── REFACTORING_SUMMARY.md              # ← What changed
└── QUICK_START.md                      # ← This file
```

## Troubleshooting

### "Agent 'X' not found"
```bash
# List available agents
python -c "from navisim_lab.agents import list_agents; print(list_agents())"
```

### "Checkpoint incompatible"
```bash
# Specify agent explicitly
python scripts/play.py --agent rsl_rl --checkpoint model.pt --config config.yaml --task MyTask-v0
```

### "Module not found"
```bash
# Ensure you're in the right directory
cd /path/to/navisim_isaacsim

# Check imports
python test_imports.py
```

## Examples

### Full Training Pipeline
```bash
# 1. Train
python scripts/train.py \
    --task Navisim-Warehouse-Jetbot-v0 \
    --agent rsl_rl \
    --config navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml \
    --num_envs 64 \
    --max_iterations 1000 \
    --experiment_name warehouse_nav \
    --run_name ppo_baseline \
    --headless

# 2. Evaluate
python scripts/play.py \
    --task Navisim-Warehouse-Jetbot-v0 \
    --checkpoint logs/warehouse_nav/ppo_baseline/model_1000.pt \
    --config navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml \
    --num_steps 10000

# 3. Visualize
python scripts/play.py \
    --task Navisim-Warehouse-Jetbot-v0 \
    --checkpoint logs/warehouse_nav/ppo_baseline/model_1000.pt \
    --config navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml \
    --num_envs 1 \
    --num_steps 1000
    # (no --headless for visualization)
```

## Next Steps

1. **Read the docs**: Check `AGENT_FRAMEWORK.md` for complete guide
2. **Try it out**: Run smoke test to verify setup
3. **Experiment**: Try different algorithms/configs
4. **Contribute**: Add new agents or environments

## Support

- Documentation: `AGENT_FRAMEWORK.md`
- Changes: `REFACTORING_SUMMARY.md`
- Tests: `python test_imports.py`
- Issues: GitHub issues with "refactoring" label

---

**Ready to start?**
```bash
python scripts/train.py --task Navisim-Warehouse-Jetbot-v0 --agent rsl_rl --config navisim_lab/configs/rsl_rl/ppo_warehouse_jetbot.yaml
```
