from dataclasses import dataclass
from typing import Union, Optional, List
import torch

try:
    from ...native.pose import Pose as nPose
except ImportError:
    import os, sys    
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from native.pose import Pose as nPose

@dataclass
class Pose:
    @classmethod
    def create(cls, pose: Union[torch.Tensor, nPose, List[nPose], "Pose"]):
        pass
        
    def __mul__(self, arg0: Union["Pose", nPose]):
        pass
    
    
