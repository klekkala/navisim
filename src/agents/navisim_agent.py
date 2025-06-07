from enums.enums import RlPolicy

import numpy as np

class NavisimAgent:
    """
    NavisimAgent class that selects actions based on a specified policy.

    Supported policies:
    - RANDOM: Selects actions randomly from the action space.
    - GREEDY: Moves directly toward the goal position using normalized direction vector.
    """
    
    def __init__(self, action_space):
        """
        Initialize the NavisimAgent.

        Args:
            action_space: Gymnasium-style action space (e.g., Box).
            policy (RlPolicy): Behavior policy used for action selection.
        """
        self.action_space = action_space
    
    def sample_action(self):
        return self.action_space.sample()


    def act(self, policy, observation):
        """
        Select an action based on the current observation and agent policy.

        Args:
            observation (dict): Dictionary containing observation data (e.g., "aux").

        Returns:
            np.ndarray: Action to apply in the environment.
        """
        if policy == RlPolicy.RANDOM:
            return self.action_space.sample()
        elif policy == RlPolicy.GREEDY:
            return self._greedy_toward_goal(observation)
        else:
            raise NotImplementedError(f"Unsupported policy: {self.policy}")

    #TODO : Implement action based on greedy policy
    def _greedy_toward_goal(self, observation):
        raise NotImplementedError("Greedy policy is not implemented yet.")
    
    