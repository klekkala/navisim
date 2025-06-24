from navisim.enums.enums import RlPolicy

class NavisimAgent:
    """
    NavisimAgent class that selects actions based on a specified policy.

    Supported policies:
    - RANDOM: Selects actions randomly from the action space.
    """
    
    def __init__(self, action_space, policy:RlPolicy = RlPolicy.RANDOM):
        """
        Initialize the NavisimAgent.

        Args:
            action_space: Gymnasium-style action space (e.g., Box).
            policy (RlPolicy): Behavior policy used for action selection.
        """
        self.action_space = action_space
        self.policy = policy
    
    def act(self, observation):
        """
        Select an action based on the current observation and agent policy.

        Args:
            observation (dict): Dictionary containing observation data (e.g., "aux").

        Returns:
            np.ndarray: Action to apply in the environment.
        """
        if self.policy == RlPolicy.RANDOM:
            return self.action_space.sample()
        else:
            raise NotImplementedError(f"Unsupported policy: {self.policy}")
    
    