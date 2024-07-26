from skimage.morphology import binary_dilation, disk
from skimage.measure import find_contours
from shapely.geometry import Polygon
from src.beogym.visualization import point_cloud_to_occupany_map
from src.beogym.pointcloud.pointcloud import *

from src.paths import *

import open3d as o3d
import os
import networkx as nx
import numpy as np
import pickle

class Node:
    '''
    Class represents a node in the sequence graph that contains
    the path to the splat_file, and the metadata of each sector

    Attributes:
        node_id(str): Unique Id of each node
        splat_file(str): Path to the Guassian splat file of the sector
        pose_transformation_path(np.ndarray): transformation matrix derived from the transformation file.
        sector_boundary(): polygon boundary of each sector
    '''

    def __init__(self, id, sector_name, grid_resolution = 10):
        '''
        :param id: id of the sector of the point cloud
        :param sector_name: name of sector of this node
        '''
        self.node_id = id
        sector_path = os.path.join(GAUSSIAN_SPLAT_FOLDER, sector_name)
        self.splat_file_path = os.path.join(sector_path, 'point_cloud.ply')
        self.guassian_splat = o3d.io.read_point_cloud(self.splat_file_path)

        #TODO(jiwon) update point cloud to use
        #TODO(Hao Peng) change the file name
        point_cloud_path = os.path.join(GAUSSIAN_SPLAT_FOLDER, 'surfaceMap_clean.pcd')
        self.global_point_cloud = get_point_cloud(point_cloud_path)
        self.global_elevation_map, self.offset_x, self.offset_y, self.min_height = get_elevation_map(point_cloud=self.global_point_cloud, grid_resolution=grid_resolution)
        self.grid_resolution = grid_resolution

        # self.elevation_map, self.elevation_shift_x, self.elevation_shift_y, self.min_elevation = get_elevation_map(self.guassian_splat)
        # self.occupancy_map = point_cloud_to_occupany_map(self.elevation_map, threshold=0.175)
        self.transformation_matrix = self._get_transformation_matrix(os.path.join(sector_path, 'transformation.txt'))
        # self.sector_boundary = self._get_boundary_polygon(occupancy_map=self.occupancy_map)

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

    # def is_agent_within_boundary(self, agent_coordinates):
    #     '''
    #     Returns whether or not the agent is within the given boundary of the point cloud
    #     :return true or false
    #     '''
    #     ox, oy = self.origin_coordinate
    #     xmin, xmax = ox + self.border_polygon.start_x, ox +  self.border_polygon.end_x
    #     ymin, ymax = oy +  self.border_polygon.end_y, oy +  self.border_polygon.start_y

    #     agent_x, agent_z = agent_coordinates
    #     return xmin <= agent_x < xmax and ymin <= agent_z < ymax

    # def get_global_pose_from_local(self, x, y, z):
    #     '''
    #     Converts local agent x,y,z coordinate to global
    #     :param x, y, z: local coordinate of the agent
    #     '''
    #     return np.dot(self.trans[:-1], np.array([x, y, z]))+self.trans[-1]

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


class BeogymSequenceGraph(nx.Graph):
    """
    Specialized graph for representing the sequence graph to be used in
    navigation between sectors in the simulation.

    This class extends networkx.Graph and includes additional functionality
    for managing nodes and edges specific to the Beogym simulation requirements.

    Attributes:
        self.elevation_map = path to the global elevation map (TODO: only used temporarily until we get elevation map for each sector)
        self.current_node = node in sequence graph that the agent is currently residing
    """

    def __init__(self, global_point_cloud, initial_nodes=None, grid_resolution = 10, *args, **kwargs):
        """
        :param global_point_cloud(str) : Name of the global point cloud data file
        :param initial_nodes(list of nodes): A list containining nodes for the initial sequence graph
        :param initial_edges(list of tuple(node, node)): A list containing edges for the initial sequence graph
        """
        super().__init__(*args, **kwargs)

        if initial_nodes:
            for node in initial_nodes:
                self.add_node(node)
                self.add_edge_to_existing_nodes(node)

        # self.save()
        self.current_node = None

    # TODO(jiwon-hae) : Implement get_node to return the node that agent is currently residing
    def get_node(self, coo_x, coo_z):
        """
        Get node in the sequence graph given agent's global x and z coordinate
        :return Node
        """
        elevation_map, occupancy_map = None, None

        # TODO(jiwon): retrieve current node from the nodes available
        if not self.current_node:
            print(self.nodes)
            current_node = list(self.nodes)[0]
        else:
            # TODO(jiwon) : reduce search scope by using the adjcant nodes to the current node as agent cannot jump
            print('')
        return current_node

    def add_edge_to_existing_nodes(self, node: Node):
        def invert_4x3_matrix(T):
            """Invert a 4x3 transformation matrix."""
            R = T[:-1, :]
            t = T[-1, :]

            R_inv = np.linalg.inv(R)
            t_inv = -np.dot(R_inv, t)

            T_inv = np.vstack((R_inv, t_inv))
            return T_inv

        def compute_relative_transform(T_global_i, T_global_j):
            """Compute the relative transformation matrix from sector i to sector j."""
            # Extract rotation and translation parts
            R_i = T_global_i[:-1, :]
            t_i = T_global_i[-1, :]

            R_j = T_global_j[:-1, :]
            t_j = T_global_j[-1, :]

            # Invert the rotation and translation of sector i
            R_i_inv = np.linalg.inv(R_i)
            t_i_inv = -np.dot(R_i_inv, t_i)

            # Compute relative rotation and translation
            R_relative = np.dot(R_i_inv, R_j)
            t_relative = np.dot(R_i_inv, t_j) + t_i_inv

            # Combine into a new 4x3 transformation matrix
            T_relative = np.vstack((R_relative, t_relative))
            return T_relative

        existing_nodes = list(self.nodes)
        if not existing_nodes:
            return

        new_transformation_matrix = node.transformation_matrix

        for existing_node in existing_nodes:
            T_relative = compute_relative_transform(new_transformation_matrix, existing_node.transformation_matrix)
            self.add_edge(node, existing_node, transform=T_relative)

    def add_node(self, node: Node):
        """
        :param node: Node class containing the splat file path and metadata file path of the sector
        """
        super().add_node(node)

    def save(self, save_path=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, '..', 'cache',
                                 'beogym_sequence_graph.pkl') if not save_path else save_path
        with open(save_path, "wb") as f:
            pickle.dump(self, f)


def load_saved_sequence_graph(graph_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(current_dir, '..', 'cache', graph_name)
    with open(graph_path, "rb") as f:
        return pickle.load(f)

