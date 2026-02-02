"""Dynamic scene manager for streaming USD sections in Isaac Sim.

This manager handles loading/unloading USD sections as the robot navigates,
working with Isaac Sim's USD stage to efficiently stream large environments.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np

from .scene_graph import SceneGraph, SceneSection

logger = logging.getLogger(__name__)


class DynamicSceneManager:
    """Manages dynamic loading/unloading of USD scene sections in Isaac Sim.

    This class bridges the SceneGraph (spatial organization) with Isaac Sim's
    USD stage, enabling efficient streaming of large environments.
    """

    def __init__(
        self,
        scene_graph: SceneGraph,
        stage,  # Isaac Sim USD stage
        max_loaded_sections: int = 9,
        load_radius: float = 50.0,
        neighbor_depth: int = 1,
        preload_sections: Optional[List[str]] = None,
    ):
        """Initialize the dynamic scene manager.

        Args:
            scene_graph: SceneGraph with section definitions
            stage: Isaac Sim USD stage (from simulation_app.context.get_stage())
            max_loaded_sections: Maximum sections to keep loaded
            load_radius: Radius around robot to keep loaded (meters)
            neighbor_depth: Graph hops to include as neighbors
            preload_sections: Optional list of sections to preload
        """
        self.scene_graph = scene_graph
        self.stage = stage
        self.max_loaded_sections = max_loaded_sections
        self.load_radius = load_radius
        self.neighbor_depth = neighbor_depth

        # Track USD prims for each section
        self.section_prims: Dict[str, str] = {}  # section_id -> prim_path

        # Track loading state
        self.loading_in_progress: Set[str] = set()
        self.unloading_in_progress: Set[str] = set()

        # Statistics
        self.total_loads = 0
        self.total_unloads = 0

        logger.info(
            f"Initialized DynamicSceneManager: "
            f"max_sections={max_loaded_sections}, "
            f"load_radius={load_radius}m, "
            f"neighbor_depth={neighbor_depth}"
        )

        # Preload initial sections
        if preload_sections:
            for section_id in preload_sections:
                self.load_section(section_id)

    def load_section(self, section_id: str) -> bool:
        """Load a USD section into the stage.

        Args:
            section_id: Section to load

        Returns:
            True if loaded successfully
        """
        section = self.scene_graph.get_section(section_id)
        if section is None:
            logger.error(f"Section {section_id} not found in scene graph")
            return False

        if section.is_loaded:
            logger.debug(f"Section {section_id} already loaded")
            return True

        if section_id in self.loading_in_progress:
            logger.debug(f"Section {section_id} loading in progress")
            return False

        try:
            self.loading_in_progress.add(section_id)
            logger.info(f"Loading section: {section_id} from {section.usd_path}")

            # Create unique prim path for this section
            prim_path = f"/World/Sections/{section_id}"

            # Load USD as reference
            # This uses Isaac Sim's USD API to add the section
            from pxr import Sdf, Usd

            # Create a reference to the USD file
            prim = self.stage.DefinePrim(prim_path)
            references = prim.GetReferences()
            references.AddReference(str(section.usd_path))

            # Store prim path for later unloading
            self.section_prims[section_id] = prim_path

            # Mark as loaded
            self.scene_graph.mark_loaded(section_id)
            self.total_loads += 1

            logger.info(
                f"Loaded section {section_id} at {prim_path} "
                f"(total loaded: {len(self.scene_graph.loaded_sections)})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load section {section_id}: {e}")
            return False

        finally:
            self.loading_in_progress.discard(section_id)

    def unload_section(self, section_id: str) -> bool:
        """Unload a USD section from the stage.

        Args:
            section_id: Section to unload

        Returns:
            True if unloaded successfully
        """
        section = self.scene_graph.get_section(section_id)
        if section is None:
            logger.error(f"Section {section_id} not found")
            return False

        if not section.is_loaded:
            logger.debug(f"Section {section_id} not loaded")
            return True

        if section_id in self.unloading_in_progress:
            logger.debug(f"Section {section_id} unloading in progress")
            return False

        try:
            self.unloading_in_progress.add(section_id)
            logger.info(f"Unloading section: {section_id}")

            # Get prim path
            prim_path = self.section_prims.get(section_id)
            if prim_path is None:
                logger.warning(f"No prim path found for section {section_id}")
                return False

            # Remove prim from stage
            prim = self.stage.GetPrimAtPath(prim_path)
            if prim:
                self.stage.RemovePrim(prim_path)

            # Clean up tracking
            del self.section_prims[section_id]
            self.scene_graph.mark_unloaded(section_id)
            self.total_unloads += 1

            logger.info(
                f"Unloaded section {section_id} "
                f"(total loaded: {len(self.scene_graph.loaded_sections)})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to unload section {section_id}: {e}")
            return False

        finally:
            self.unloading_in_progress.discard(section_id)

    def update(self, robot_position: np.ndarray) -> Dict[str, List[str]]:
        """Update loaded sections based on robot position.

        This should be called every frame or at regular intervals.

        Args:
            robot_position: Current robot position [x, y, z]

        Returns:
            Dict with 'loaded' and 'unloaded' section lists
        """
        # Compute what should be loaded/unloaded
        to_load, to_unload = self.scene_graph.compute_loading_strategy(
            robot_position,
            max_loaded_sections=self.max_loaded_sections,
            load_radius=self.load_radius,
            neighbor_depth=self.neighbor_depth,
        )

        loaded = []
        unloaded = []

        # Unload first to free memory
        for section_id in to_unload:
            if self.unload_section(section_id):
                unloaded.append(section_id)

        # Then load new sections
        for section_id in to_load:
            if self.load_section(section_id):
                loaded.append(section_id)

        if loaded or unloaded:
            logger.info(
                f"Scene update: loaded {len(loaded)}, unloaded {len(unloaded)} "
                f"(total: {len(self.scene_graph.loaded_sections)})"
            )

        return {"loaded": loaded, "unloaded": unloaded}

    def preload_path(self, section_ids: List[str]):
        """Preload sections along a path.

        Useful for preloading a planned route.

        Args:
            section_ids: List of section IDs to preload
        """
        logger.info(f"Preloading {len(section_ids)} sections along path")

        for section_id in section_ids:
            self.load_section(section_id)

    def get_current_section(self, robot_position: np.ndarray) -> Optional[str]:
        """Get the section ID containing the robot.

        Args:
            robot_position: Robot position [x, y, z]

        Returns:
            Section ID or None
        """
        return self.scene_graph.find_section_containing_point(robot_position)

    def get_loaded_section_ids(self) -> List[str]:
        """Get list of currently loaded sections.

        Returns:
            List of loaded section IDs
        """
        return list(self.scene_graph.loaded_sections)

    def get_stats(self) -> Dict:
        """Get loading statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_sections": len(self.scene_graph),
            "loaded_sections": len(self.scene_graph.loaded_sections),
            "total_loads": self.total_loads,
            "total_unloads": self.total_unloads,
            "loading_in_progress": len(self.loading_in_progress),
            "unloading_in_progress": len(self.unloading_in_progress),
        }

    def reset(self, initial_position: Optional[np.ndarray] = None):
        """Reset the scene manager, unloading all sections.

        Args:
            initial_position: Optional position to preload around
        """
        logger.info("Resetting DynamicSceneManager")

        # Unload all sections
        for section_id in list(self.scene_graph.loaded_sections):
            self.unload_section(section_id)

        # Reset statistics
        self.total_loads = 0
        self.total_unloads = 0

        # Optionally preload around initial position
        if initial_position is not None:
            self.update(initial_position)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"DynamicSceneManager("
            f"loaded={stats['loaded_sections']}/{stats['total_sections']}, "
            f"loads={stats['total_loads']}, "
            f"unloads={stats['total_unloads']})"
        )
