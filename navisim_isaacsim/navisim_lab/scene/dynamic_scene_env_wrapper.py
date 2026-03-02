"""Environment wrapper for dynamic scene loading in IsaacLab.

This wrapper integrates DynamicSceneManager with IsaacLab environments,
automatically managing USD section loading as robots navigate.

Sector transition logic:
- Transition detection runs every step (cheap XY-AABB check).
- On detection: robot is immediately teleported to the new sector's center
  (preserving heading, zeroing velocity) via BaseNavigationEnv.teleport_robot().
- USD load/unload streaming runs every `update_frequency` steps (stage ops are
  more expensive and do not need to be synchronous with the transition).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import numpy as np

from isaaclab.envs import DirectRLEnv
from .scene_graph import SceneGraph
from .dynamic_scene_manager import DynamicSceneManager

logger = logging.getLogger(__name__)


class DynamicSceneEnvWrapper:
    """Wrapper that adds dynamic scene loading and sector transitions to an IsaacLab env.

    On each step the wrapper:
    1. Detects whether the robot has crossed into a new sector (XY-AABB, every step).
    2. On transition: swaps the loaded USD immediately and teleports the robot to
       the new sector's center (heading preserved, velocity zeroed, no episode reset).
    3. Updates the USD streaming window (load/unload by radius) every N steps.

    Example::

        env = gym.make("Navisim-Outdoor-Jetbot", cfg=env_cfg)
        scene_graph = build_scene_graph(...)   # SceneGraph with your USDZ sectors
        env = DynamicSceneEnvWrapper(env, scene_graph=scene_graph)

        obs, _ = env.reset()
        for _ in range(10000):
            obs, reward, done, truncated, info = env.step(policy(obs))
            if "section_transition" in info:
                print(info["section_transition"])
    """

    def __init__(
        self,
        env: DirectRLEnv,
        scene_graph: SceneGraph,
        max_loaded_sections: int = 3,
        load_radius: float = 20.0,
        neighbor_depth: int = 1,
        update_frequency: int = 10,
        robot_index: int = 0,
        preload_sections: Optional[List[str]] = None,
    ):
        """Initialize the dynamic scene wrapper.

        Args:
            env: Base IsaacLab environment to wrap.
            scene_graph: SceneGraph describing all sectors and their adjacency.
            max_loaded_sections: Maximum USD sections to keep resident in the stage.
            load_radius: Radius (metres) around the robot for pre-loading neighbours.
            neighbor_depth: Graph-hop depth to include when computing the load window.
            update_frequency: Steps between USD streaming updates (load/unload calls).
                Transition detection and teleportation always run every step.
            robot_index: Which parallel env index to use for single-robot tracking.
            preload_sections: Optional section IDs to load at construction time.
        """
        self.env = env
        self.scene_graph = scene_graph
        self.update_frequency = update_frequency
        self.robot_index = robot_index

        self.scene_manager = DynamicSceneManager(
            scene_graph=scene_graph,
            stage=env.sim.stage,
            max_loaded_sections=max_loaded_sections,
            load_radius=load_radius,
            neighbor_depth=neighbor_depth,
            preload_sections=preload_sections,
        )

        self.step_count = 0
        # Per-env current section tracking (empty string = unknown)
        self.current_sections: List[str] = ["" for _ in range(env.num_envs)]

        logger.info(
            f"DynamicSceneEnvWrapper: update_freq={update_frequency}, "
            f"max_sections={max_loaded_sections}, load_radius={load_radius}m"
        )

    # ------------------------------------------------------------------
    # Public Gymnasium-compatible interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        robot_pos = self._get_robot_position()
        if robot_pos is not None:
            # Seed the current section so the first step doesn't misfire
            section_id = self.scene_graph.find_section_containing_point(robot_pos)
            self.current_sections[self.robot_index] = section_id or ""
            self.scene_manager.update(robot_pos)

        self.step_count = 0

        if isinstance(info, dict):
            info["scene_stats"] = self.scene_manager.get_stats()
            info["current_section"] = self.current_sections[self.robot_index]

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        robot_pos = self._get_robot_position()
        if robot_pos is None:
            return obs, reward, terminated, truncated, info

        # --- Transition detection: every step (cheap) ---
        current_section = self.scene_graph.find_section_containing_point(robot_pos)
        if (
            current_section is not None
            and current_section != self.current_sections[self.robot_index]
        ):
            reward = self._handle_transition(
                current_section, robot_pos, reward, info
            )

        # --- USD streaming: every N steps (expensive stage ops) ---
        if self.step_count % self.update_frequency == 0:
            update_result = self.scene_manager.update(robot_pos)
            if isinstance(info, dict):
                info["scene_update"] = update_result
                info["scene_stats"] = self.scene_manager.get_stats()

        return obs, reward, terminated, truncated, info

    def close(self):
        self.scene_manager.reset()
        self.env.close()

    # ------------------------------------------------------------------
    # Transition handling
    # ------------------------------------------------------------------

    def _handle_transition(
        self,
        new_section_id: str,
        robot_pos: np.ndarray,
        reward: Any,
        info: dict,
    ) -> Any:
        """Handle a detected sector boundary crossing.

        Steps:
        1. Swap the loaded USD immediately (unload old, load new + neighbours).
        2. Teleport the robot to the new sector's XY centre, keeping its current Z.
        3. Zero the reward for this step to avoid the position-jump spike.
        4. Record the transition in info.

        Args:
            new_section_id: The section the robot just entered.
            robot_pos: Robot world position at the moment of detection.
            reward: Current step reward tensor (will be zeroed).
            info: Step info dict to annotate.

        Returns:
            Zeroed reward tensor.
        """
        prev_section_id = self.current_sections[self.robot_index]
        self.current_sections[self.robot_index] = new_section_id

        new_section = self.scene_graph.get_section(new_section_id)
        target: Optional[np.ndarray] = None

        if new_section is not None:
            # Teleport to sector XY centre; keep robot's live Z (wheel height)
            target = new_section.center.copy()
            target[2] = robot_pos[2]

            # Swap USD immediately so the visual matches the new position
            self.scene_manager.load_section(new_section_id)
            if prev_section_id:
                self.scene_manager.unload_section(prev_section_id)

            # Teleport robot (no episode reset)
            if hasattr(self.env, "teleport_robot"):
                self.env.teleport_robot(target)
            else:
                logger.warning(
                    "env does not implement teleport_robot(); "
                    "sector transition will be visual-only."
                )

        # Zero reward on the transition step to avoid the position-jump spike
        if isinstance(reward, torch.Tensor):
            reward = torch.zeros_like(reward)
        else:
            reward = 0.0

        logger.info(
            f"Sector transition: '{prev_section_id}' -> '{new_section_id}' | "
            f"teleport target: {target}"
        )

        if isinstance(info, dict):
            info["section_transition"] = {
                "from": prev_section_id,
                "to": new_section_id,
                "teleport_target": target.tolist() if target is not None else None,
            }

        return reward

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_robot_position(self) -> Optional[np.ndarray]:
        """Return world-space XYZ of the tracked robot, or None on failure."""
        try:
            for name in ("jetbot", "robot", "agent"):
                if name in self.env.scene:
                    pos = self.env.scene[name].data.root_state_w[self.robot_index, :3]
                    return pos.cpu().numpy() if isinstance(pos, torch.Tensor) else np.array(pos)
        except Exception as e:
            logger.error(f"Error getting robot position: {e}")
        return None

    def preload_path(self, path_section_ids: List[str]):
        """Preload sections along a planned route."""
        self.scene_manager.preload_path(path_section_ids)

    def get_scene_stats(self) -> Dict:
        return self.scene_manager.get_stats()

    def visualize_scene_graph(self, output_path: Optional[Path] = None):
        self.scene_graph.visualize(output_path)

    # Forward all other attribute access to the wrapped env
    def __getattr__(self, name):
        return getattr(self.env, name)

    def __repr__(self) -> str:
        return f"DynamicSceneEnvWrapper({self.env})"
