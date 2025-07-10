from enum import Enum

class RenderBackend(Enum):
    GPU = 'gpu'
    CPU = 'cpu'

class SimulationBackend(Enum):
    AUTO = 'auto'
    CPU = 'cpu'
    CUDA = 'cuda'