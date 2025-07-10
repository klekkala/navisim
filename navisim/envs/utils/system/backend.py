from dataclasses import dataclass
import torch

try:
    from ...backends import RenderBackend, SimulationBackend
except ImportError:
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    from navisim.envs.backends import RenderBackend, SimulationBackend
    
@dataclass
class BackendInfo:
    device : torch.device
    sim_backend : SimulationBackend
    render_backend : RenderBackend


def parse_sim_and_render_backend(sim_backend: SimulationBackend, render_backend: RenderBackend) -> BackendInfo:
    return BackendInfo(
        device = torch.device(sim_backend.name),
        sim_backend = sim_backend,
        render_backend = render_backend
    )