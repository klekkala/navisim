"""Environment wrapper for dynamic scene loading in IsaacLab.

This wrapper integrates DynamicSceneManager with IsaacLab environments,
automatically managing USD section loading as robots navigate.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import numpy as np

from isaaclab.envs import DirectRLEnv
from .scene_graph import SceneGraph
from .dynamic_scene_manager import DynamicSceneManager

logger = logging.getLogger(__name__)


class DynamicSceneEnvWrapper:
    """Wrapper that adds dynamic scene loading to an IsaacLab environment.

    This wrapper intercepts environment steps and automatically loads/unloads
    USD sections based on robot positions.

    Example:
        # Create base environment
        env = gym.make("Navisim-Warehouse-Jetbot-v0", cfg=env_cfg)

        # Wrap with dynamic scene loading
        scene_graph = SceneGraph.from_pickle("path/to/scene_graph.pkl")
        env = DynamicSceneEnvWrapper(
            env,
            scene_graph=scene_graph,
            update_frequency=10  # Update every 10 steps
        )

        # Use normally
        obs, _ = env.reset()
        for _ in range(1000):
            action = policy(obs)
            obs, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        env: DirectRLEnv,
        scene_graph: SceneGraph,
        max_loaded_sections: int = 9,
        load_radius: float = 50.0,
        neighbor_depth: int = 1,
        update_frequency: int = 10,
        robot_index: int = 0,
        preload_sections: Optional[list] = None,
    ):
        """Initialize the dynamic scene wrapper.

        Args:
            env: Base IsaacLab environment to wrap
            scene_graph: SceneGraph with section definitions
            max_loaded_sections: Maximum sections to keep loaded
            load_radius: Radius around robot to load (meters)
            neighbor_depth: Graph neighbor depth to include
            update_frequency: How often to update (in steps)
            robot_index: Which robot to track (for multi-robot envs)
            preload_sections: Optional sections to preload initially
        """
        self.env = env
        self.scene_graph = scene_graph
        self.update_frequency = update_frequency
        self.robot_index = robot_index

        # Get USD stage from environment
        stage = env.sim.stage

        # Create scene manager
        self.scene_manager = DynamicSceneManager(
            scene_graph=scene_graph,
            stage=stage,
            max_loaded_sections=max_loaded_sections,
            load_radius=load_radius,
            neighbor_depth=neighbor_depth,
            preload_sections=preload_sections,
        )

        # Track steps for update frequency
        self.step_count = 0

        # Track current section for each environment
        self.current_sections = ["" for _ in range(env.num_envs)]

        logger.info(
            f"Initialized DynamicSceneEnvWrapper: "
            f"update_freq={update_frequency}, "
            f"robot_idx={robot_index}"
        )

    def reset(self, **kwargs):
        """Reset environment and scene manager."""
        obs, info = self.env.reset(**kwargs)

        # Get robot position and update scene
        robot_pos = self._get_robot_position()
        if robot_pos is not None:
            self.scene_manager.update(robot_pos)

        self.step_count = 0

        # Add scene info to info dict
        if isinstance(info, dict):
            info["scene_stats"] = self.scene_manager.get_stats()
            info["current_section"] = self.scene_manager.get_current_section(robot_pos)

        return obs, info

    def step(self, action):
        """Step environment and update scene if needed."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.step_count += 1

        # Update scene at specified frequency
        if self.step_count % self.update_frequency == 0:
            robot_pos = self._get_robot_position()
            if robot_pos is not None:
                update_result = self.scene_manager.update(robot_pos)

                # Add scene info to info dict
                if isinstance(info, dict):
                    info["scene_update"] = update_result
                    info["scene_stats"] = self.scene_manager.get_stats()
                    info["current_section"] = self.scene_manager.get_current_section(
                        robot_pos
                    )

                # Check for section transitions
                current_section = self.scene_manager.get_current_section(robot_pos)
                if (
                    current_section
                    and current_section != self.current_sections[self.robot_index]
                ):
                    logger.info(
                        f"Robot transitioned to section: {current_section} "
                        f"(from {self.current_sections[self.robot_index]})"
                    )
                    self.current_sections[self.robot_index] = current_section

                    if isinstance(info, dict):
                        info["section_transition"] = {
                            "from": self.current_sections[self.robot_index],
                            "to": current_section,
                        }

        return obs, reward, terminated, truncated, info

    def _get_robot_position(self) -> Optional[np.ndarray]:
        """Get current robot position from environment.

        Returns:
            Robot position [x, y, z] or None if not available
        """
        try:
            # Access robot through scene
            # Assuming env has 'scene' with robot articulation
            if hasattr(self.env, "scene") and hasattr(self.env.scene, "data"):
                # For environments with named robots (e.g., "jetbot")
                robot_names = ["jetbot", "robot", "agent"]
                for name in robot_names:
                    if name in self.env.scene:
                        robot = self.env.scene[name]
                        if hasattr(robot, "data") and hasattr(
                            robot.data, "root_state_w"
                        ):
                            # Get position of first environment's robot
                            pos = robot.data.root_state_w[self.robot_index, :3]
                            if isinstance(pos, torch.Tensor):
                                pos = pos.cpu().numpy()
                            return np.array(pos)

            logger.warning("Could not extract robot position from environment")
            return None

        except Exception as e:
            logger.error(f"Error getting robot position: {e}")
            return None

    def close(self):
        """Close environment and clean up scene manager."""
        logger.info("Closing DynamicSceneEnvWrapper")
        self.scene_manager.reset()
        self.env.close()

    def preload_path(self, path_section_ids: list):
        """Preload sections along a path.

        Args:
            path_section_ids: List of section IDs to preload
        """
        self.scene_manager.preload_path(path_section_ids)

    def get_scene_stats(self) -> Dict:
        """Get scene loading statistics.

        Returns:
            Dictionary with statistics
        """
        return self.scene_manager.get_stats()

    def visualize_scene_graph(self, output_path: Optional[Path] = None):
        """Visualize the scene graph.

        Args:
            output_path: Optional path to save visualization
        """
        self.scene_graph.visualize(output_path)

    # Forward all other attributes to wrapped environment
    def __getattr__(self, name):
        """Forward attribute access to wrapped environment."""
        return getattr(self.env, name)

    def __repr__(self) -> str:
        return f"DynamicSceneEnvWrapper({self.env})"
