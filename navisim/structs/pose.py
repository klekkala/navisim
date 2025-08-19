from dataclasses import dataclass
from typing import Union, Optional, List
import ray

try:
    from ..utils.ray_utils import sync_remote
    from ...native.pose import Pose as nPose
except ImportError:
    import os, sys    
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    
    from navisim.utils.ray_utils import sync_remote
    from native.pose import Pose as nPose
    
ray.init(ignore_reinit_error=True)

@ray.remote(num_cpus=1)
def _translate(p: nPose, dx: float, dy: float, dz: float) -> nPose:
    p.translate(dx, dy, dz)
    return p

@ray.remote(num_cpus=1)
def _inverse(p: nPose) -> nPose:
    p.inverse()
    return p

translate = sync_remote(_translate)
inverse = sync_remote(_inverse)

if __name__ == "__main__":
    # 4) Create a sample and call remote
    samplePose = nPose(0, 0, 0, 0, 0, 0)
    print("before:", samplePose)
    samplePose.translate(1, 1, 1)
    print("local:", samplePose)

    # 5) Send it through Ray
    translated = translate(samplePose, 1.0, 2.0, 3.0)
    print("remote:", translated)
    
    inverse = inverse(samplePose)
    print("remote inverse:", inverse)