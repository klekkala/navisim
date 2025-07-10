from typing import Optional, Union

import gymnasium as gym

try:
    from .backends import RenderBackend, SimulationBackend
    from .utils.system.backend import parse_sim_and_render_backend
except ImportError:
    import sys, os

    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )
    from navisim.envs.backends import RenderBackend, SimulationBackend
    from navisim.envs.utils.system.backend import parse_sim_and_render_backend


class BaseEnv(gym.Env):
    def __init__(
        self,
        render_mode: Optional[str] = None,
        sim_backend: SimulationBackend = SimulationBackend.AUTO,
        render_backend: RenderBackend = RenderBackend.GPU,
        num_envs: int = 1,
    ):
        if sim_backend == SimulationBackend.AUTO:
            if num_envs > 1:
                sim_backend = SimulationBackend.CUDA
            else:
                sim_backend = SimulationBackend.CPU

        self.num_envs = num_envs
        self.backend = parse_sim_and_render_backend(sim_backend, render_backend)
        self.device = self.backend.device
        self.render_mode = render_mode
        self._viewer = None
        
        # self._sim_device = self.backend.sim_device
        # self._render_revice = self.backend.render_device
    
    def _load_agent(self, initial_agent_poses : Optional[Union[Pose]])
