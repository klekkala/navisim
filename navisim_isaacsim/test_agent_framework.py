#!/usr/bin/env python3
"""Test script to verify the agent framework refactoring."""

import sys
from pathlib import Path

# Ensure navisim_lab is in the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_agent_registry():
    """Test that agent registry works correctly."""
    print("=" * 80)
    print("Testing Agent Registry")
    print("=" * 80)

    from navisim_lab.agents import list_agents, create_agent, get_agent_class

    # List available agents
    agents = list_agents()
    print(f"✓ Available agents: {agents}")
    assert "rsl_rl" in agents, "RSL-RL agent should be registered"

    # Get agent class
    AgentClass = get_agent_class("rsl_rl")
    print(f"✓ Got agent class: {AgentClass.__name__}")

    # Create agent instance
    agent = create_agent("rsl_rl", algorithm="ppo")
    print(f"✓ Created agent instance: {agent}")
    assert agent.algorithm == "ppo", "Algorithm should be set correctly"

    print("✓ Agent registry tests passed!\n")


def test_base_navigation_env():
    """Test that BaseNavigationEnv can be imported."""
    print("=" * 80)
    print("Testing BaseNavigationEnv")
    print("=" * 80)

    from navisim_lab.envs.base import BaseNavigationEnv

    print(f"✓ BaseNavigationEnv imported: {BaseNavigationEnv.__name__}")
    print(f"✓ Abstract methods: {BaseNavigationEnv.__abstractmethods__}")

    print("✓ BaseNavigationEnv tests passed!\n")


def test_warehouse_env_import():
    """Test that WarehouseEnv still works after refactoring."""
    print("=" * 80)
    print("Testing WarehouseEnv Import")
    print("=" * 80)

    from navisim_lab.envs.warehouse.warehouse_env import WarehouseEnv
    from navisim_lab.envs.base import BaseNavigationEnv

    print(f"✓ WarehouseEnv imported: {WarehouseEnv.__name__}")

    # Check inheritance
    assert issubclass(
        WarehouseEnv, BaseNavigationEnv
    ), "WarehouseEnv should inherit from BaseNavigationEnv"
    print(f"✓ WarehouseEnv correctly inherits from BaseNavigationEnv")

    print("✓ WarehouseEnv import tests passed!\n")


def test_task_registration():
    """Test that Gym task is still registered."""
    print("=" * 80)
    print("Testing Task Registration")
    print("=" * 80)

    import gymnasium as gym
    import navisim_lab.tasks  # noqa: F401

    # Check if task is registered
    task_id = "Navisim-Warehouse-Jetbot-v0"
    assert task_id in gym.envs.registry, f"Task {task_id} should be registered"
    print(f"✓ Task {task_id} is registered")

    # Get task spec
    spec = gym.spec(task_id)
    print(f"✓ Task spec: {spec}")
    print(f"  Entry point: {spec.entry_point}")
    print(f"  Kwargs: {spec.kwargs}")

    print("✓ Task registration tests passed!\n")


def test_agent_interface():
    """Test that RslRlAgent implements the interface correctly."""
    print("=" * 80)
    print("Testing Agent Interface")
    print("=" * 80)

    from navisim_lab.agents import create_agent
    from navisim_lab.agents.base_agent import BaseRLAgent

    agent = create_agent("rsl_rl", algorithm="ppo")

    # Check that it's an instance of BaseRLAgent
    assert isinstance(
        agent, BaseRLAgent
    ), "Agent should be instance of BaseRLAgent"
    print(f"✓ Agent is instance of BaseRLAgent")

    # Check required methods exist
    required_methods = [
        "setup",
        "train",
        "load",
        "save",
        "get_policy",
        "wrap_env",
        "from_checkpoint",
    ]

    for method in required_methods:
        assert hasattr(agent, method), f"Agent should have method '{method}'"
        print(f"✓ Agent has method: {method}")

    # Check required properties
    assert hasattr(
        agent, "requires_env_wrapper"
    ), "Agent should have 'requires_env_wrapper' property"
    print(f"✓ Agent requires_env_wrapper: {agent.requires_env_wrapper}")

    print("✓ Agent interface tests passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("NaviSim Agent Framework Tests")
    print("=" * 80 + "\n")

    try:
        test_agent_registry()
        test_base_navigation_env()
        test_warehouse_env_import()
        test_task_registration()
        test_agent_interface()

        print("=" * 80)
        print("✓ All tests passed successfully!")
        print("=" * 80)
        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ Tests failed!")
        print("=" * 80)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
