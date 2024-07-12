import sys
import os
import numpy as np

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from beogym import BeoGym
from util.video_helper import images2video
from pointcloud.pointcloud import PointCloud
from pointcloud.sequence_graph import get_sequence_graph


target_location = np.array([10, 0]) #np.zeros(2, dtype = int)
start_location = np.array([-80, 158]) #np.zeros(2, dtype = int)
env = BeoGym(config={}, sequence_graph = get_sequence_graph(), start_location=start_location, target_location = target_location, render_mode="human")

observation, info = env.reset()
env.render()

# for i in range(5):
#     # print("Taking Action# ", i)
#     # action = env.action_space.sample()
#     # print("Action Value: ", action)
#     action = [1, 1]
    
#     observation, reward, terminated, truncated, info = env.step(action)

    # if terminated or truncated:
    #     break

    # env.render()


# images2video("./src/assets/gaussian_output/images", "./src/assets/gaussian_output/videos", "output_video", fps=3)

