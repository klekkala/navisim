import torch
from isaaclab.controllers import BasePolicy

class SACPolicy(BasePolicy):
    def __init__(self, actor_path: str, obs_dim: int, act_dim: int, device="cuda"):
        super().__init__()
        self.device = torch.device(device)

        # Build actor network structure (must match RL-Games architecture)
        self.actor = self._build_actor(obs_dim, act_dim).to(self.device)

        # Load weights from RL-Games checkpoint
        checkpoint = torch.load(actor_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])

        self.actor.eval()

    def _build_actor(self, obs_dim, act_dim):
        # ← You MUST match RL-Games SAC actor architecture
        return torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_dim),
            torch.nn.Tanh(),
        )

    @torch.no_grad()
    def act(self, obs):
        """
        obs: numpy array [num_envs, obs_dim]
        return: torch or numpy array [num_envs, act_dim]
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = self.actor(obs_t)
        return action.cpu().numpy()

    def reset(self, env_ids=None):
        pass  # SAC doesn't need hidden states
