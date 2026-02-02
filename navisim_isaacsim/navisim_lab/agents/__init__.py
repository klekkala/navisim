"""Agent registry for NaviSim RL frameworks.

This module provides a registry system for pluggable RL agent implementations.
New agent frameworks can be added by implementing BaseRLAgent and registering
with the @register_agent decorator.

Example:
    from navisim_lab.agents import create_agent, list_agents

    # Create an agent
    agent = create_agent("rsl_rl", algorithm="ppo")

    # List available agents
    available = list_agents()  # ['rsl_rl', 'sb3', ...]
"""

from typing import Dict, Type, Optional
from .base_agent import BaseRLAgent

_AGENT_REGISTRY: Dict[str, Type[BaseRLAgent]] = {}


def register_agent(name: str):
    """Decorator to register an agent implementation.

    Args:
        name: Unique identifier for the agent (e.g., 'rsl_rl', 'sb3')

    Example:
        @register_agent("rsl_rl")
        class RslRlAgent(BaseRLAgent):
            ...
    """

    def decorator(cls: Type[BaseRLAgent]):
        if name in _AGENT_REGISTRY:
            raise ValueError(f"Agent '{name}' is already registered")
        if not issubclass(cls, BaseRLAgent):
            raise TypeError(f"Agent class must inherit from BaseRLAgent")
        _AGENT_REGISTRY[name] = cls
        return cls

    return decorator


def create_agent(name: str, algorithm: Optional[str] = None, **kwargs) -> BaseRLAgent:
    """Factory function to create agent instances.

    Args:
        name: Agent framework name (e.g., 'rsl_rl', 'sb3')
        algorithm: Algorithm name (e.g., 'ppo', 'sac'), if applicable
        **kwargs: Additional arguments passed to agent constructor

    Returns:
        Initialized agent instance

    Raises:
        ValueError: If agent name is not registered

    Example:
        agent = create_agent("rsl_rl", algorithm="ppo")
    """
    if name not in _AGENT_REGISTRY:
        available = list(_AGENT_REGISTRY.keys())
        raise ValueError(
            f"Agent '{name}' not found in registry. " f"Available agents: {available}"
        )

    AgentClass = _AGENT_REGISTRY[name]

    # Try to pass algorithm if the constructor accepts it
    try:
        if algorithm is not None:
            return AgentClass(algorithm=algorithm, **kwargs)
        else:
            return AgentClass(**kwargs)
    except TypeError:
        # Fallback if constructor doesn't accept algorithm
        return AgentClass(**kwargs)


def get_agent_class(name: str) -> Type[BaseRLAgent]:
    """Get agent class by name without instantiating.

    Args:
        name: Agent framework name

    Returns:
        Agent class

    Raises:
        ValueError: If agent name is not registered
    """
    if name not in _AGENT_REGISTRY:
        available = list(_AGENT_REGISTRY.keys())
        raise ValueError(
            f"Agent '{name}' not found in registry. " f"Available agents: {available}"
        )
    return _AGENT_REGISTRY[name]


def list_agents() -> list[str]:
    """List all registered agent names.

    Returns:
        List of registered agent identifiers
    """
    return list(_AGENT_REGISTRY.keys())


# Auto-import agent implementations to trigger registration
try:
    from .rsl_rl import *  # noqa: F401, F403
except ImportError:
    pass  # RSL-RL not installed or not available

# Future agent frameworks can be imported here
# from .sb3 import *  # noqa: F401, F403
# from .skrl import *  # noqa: F401, F403


__all__ = [
    "BaseRLAgent",
    "register_agent",
    "create_agent",
    "get_agent_class",
    "list_agents",
]
