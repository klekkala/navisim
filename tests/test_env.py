import sys
import os
import numpy as np

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.beogym.beogym import BeoGym
from src.beogym.sequence_graph.sequence_graph import *
# from src.util.video_helper import images2video

# Sample Sequence Graph
node1 = Node('sec2')
# node1 = Node('sec4')
# node1 = Node('sec5')
sequenceGraph = BeogymSequenceGraph(initial_nodes=[node1])

# Beogym Env
target_location = np.array([10, 0]) #np.zeros(2, dtype = int)
start_location = np.array([-80, 158]) #np.zeros(2, dtype = int)
env = BeoGym(
    config={},
    sequence_graph = sequenceGraph,
    start_location=start_location,
    target_location = target_location,
    render_mode="human"
)

observation, info = env.reset()
env.render()

# -------------------------------------------------------------------------------------
# for i in range(5):
#     action = [1, 1]
#     observation, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         break
#     env.render()

# images2video("./src/assets/gaussian_output/images", "./src/assets/gaussian_output/videos", "output_video", fps=3)
# -------------------------------------------------------------------------------------

# Example: python tests\test_env.py