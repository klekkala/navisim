import networkx as nx
import os
import logging
import pickle
import numpy as np
import open3d as o3d
from skimage.morphology import binary_dilation, disk
from skimage.measure import find_contours
from shapely.geometry import Polygon
from pathlib import Path

from src.beogym.pointcloud.pointcloud import smooth_out_point_cloud, point_cloud_to_height_map
from src.paths import SEQUENCE_GRAPH_FOLDER



class Node:
    def __init__(self, node_id, height_limit, grid_resolution):
        self.node_id = node_id
        self.sector_name = f'sec{node_id}'

        self.elevation_map = self._get_elevation_map(height_limit=height_limit, grid_resolution = grid_resolution)

        matrix_path = Path(SEQUENCE_GRAPH_FOLDER) / self.sector_name / 'transformation.txt'
        self.transformation_matrix = self._get_transformation_matrix(matrix_path)

    def _get_elevation_map(self, height_limit, grid_resolution):
        '''
        Compute 3d point cloud onto 2d elevation map

        :param point_cloud: 3d point cloud
        :param height_limit: height threshold to clip any height above the limit
        :param grid_resolution: resolution used for visualizing elevation map

        :return: elevation map

            x is the row informations.
            z is the column informations.

            Min offset of row, Min offset of col
            min_bound[index_x], min_bound[index_z]
        '''
        pcd_path = Path(SEQUENCE_GRAPH_FOLDER) / self.sector_name / 'clean_pcd.pcd'
        point_cloud_data = o3d.io.read_point_cloud(pcd_path)

        index_x = 0
        index_y = 1  # index of the height coordinate
        index_z = 2

        point_cloud_np = np.asarray(point_cloud_data.points)
        min_bound = np.rint(point_cloud_data.get_min_bound()).astype(int)
        max_bound = np.rint(point_cloud_data.get_max_bound()).astype(int)
        point_cloud_np = smooth_out_point_cloud(point_cloud_np, axis_height=index_y, min_height=-10,
                                                max_height=height_limit)

        grid_width = (np.abs(max_bound[index_x]) + np.abs(min_bound[index_x])) * grid_resolution
        grid_height = (np.abs(max_bound[index_z]) + np.abs(min_bound[index_z])) * grid_resolution

        # Extract the highest points
        max_height = np.max(point_cloud_np[:, index_y])
        min_height = np.min(point_cloud_np[:, index_y])
        # min_height = 0

        elevation_map = point_cloud_to_height_map(point_cloud_np, grid_lower_bound=min_bound, grid_width=grid_width,
                                                  grid_height=grid_height, min_height=min_height,
                                                  grid_resolution=grid_resolution)
        return elevation_map


    def _get_boundary_polygon(self, occupancy_map, level=0.5, disk_size=5):
        """
        Generate a boundary polygon for all coordinates with value 1 in the occupancy map.

        :param occupancy_map: 2D numpy array, Occupancy map with binary values (0 and 1).
        :return: Polygon, Shapely Polygon object representing the boundary.
        """
        dilated_map = binary_dilation(occupancy_map, disk(disk_size))
        contours = find_contours(dilated_map, level=level)
        max_contour = max(contours, key=len)
        polygon = Polygon(max_contour)

        return polygon

    def _get_transformation_matrix(self, matrix_path):
        """
        Reads the transformation file and generate a transformation matrix.

        :param matrix_path(str):Path to the transformation.txt file.
        :return: np.ndarray, Transformation matrix.
        """
        try:
            with open(matrix_path, 'r') as file:
                lines = file.readlines()
                cleaned_lines = [line.replace('[', '').replace(']', '') for line in lines]
                matrix = [[float(val) for val in line.split()] for line in cleaned_lines]
            return np.array(matrix)
        except Exception as e:
            print("An error occurred while reading the transformation matrix:")
            print(e)
            return None


class BeogymSequenceGraphTest(nx.Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self, graph_name='beogym_sequence_graph.pkl'):
        os.makedirs(SEQUENCE_GRAPH_FOLDER, exist_ok=True)
        save_path = os.path.join(SEQUENCE_GRAPH_FOLDER, graph_name)

        logging.info(f'SequenceGraph saved at {save_path}')

        with open(save_path, "wb") as f:
            pickle.dump(self, f)


def load(graph_name='beogym_sequence_graph.pkl'):
    graph_path = os.path.join(SEQUENCE_GRAPH_FOLDER, graph_name)
    with open(graph_path, "rb") as f:
        return pickle.load(f)


graph = BeogymSequenceGraphTest()
graph.save()
