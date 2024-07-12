from pointcloud.pointcloud import PointCloud
import os
import networkx as nx

import pickle

class Boundary:
    '''
    Class that represents the boundary of the point cloud
    represented by the left-top coordinate and right-bottom coordinates
    '''
    def __init__(self, start_x, start_y, end_x, end_y):
        '''
        :param start_x, start_y : x,y coordinate of the top left corner of the boundary
        :param end_x, end_y : x,y coordinate of the bottom right corner of the boundary
        '''
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

    def __iter__(self):
        yield self.start_x
        yield self.start_y
        yield self.end_x
        yield self.end_y

class Node:
    '''
    Class represents a node in the sequence graph
    '''
    def __init__(self, id, pointcloud_file_path, transformation_file_path, boundary: Boundary):
        '''
        :param pointcloud_file_path: path of the point cloud file
        :param transformation_file_path: path of the transformation.txt file
        :param boundary: boundary of the point cloud 
        '''
        self.trajectory = pointcloud_file_path
        self.trajectory_id = id
        self.endpoint = None
        self.trans = transformation_file_path
        self.boundary = boundary

    def get_point_cloud(self):
        return PointCloud(file_path = self.trajectory)

def get_sequence_graph(graph_name = 'graph.pkl'):
    '''
    Computes and saves the sequence graph
    
    :param graph_name: name of the graph file to be saved
    '''
    graph_folder = 'src/assets/splat'
    if not os.path.isdir(graph_folder):
        os.mkdir(graph_folder)

    sequenceSec1= Node(id='sec1', pointcloud_file_path = 'src/assets/splat/sec1/point_cloud.ply', transformation_file_path = 'src/assets/sec1/transformation.txt', boundary=Boundary(-5, -5, 5, 5)) #boundary=Boundary(-20, -20, 20, 20))
    sequenceSec2 = Node(id='sec2', pointcloud_file_path = 'src/assets/splat/sec2/point_cloud.ply', transformation_file_path = 'src/assets/sec2/transformation.txt', boundary=Boundary(-5, -5, 5, 5)) #boundary=Boundary(-20, -20, 20, 20))

    graph = nx.Graph()
    graph.add_node(sequenceSec1)
    graph.add_node(sequenceSec2)
    graph.add_edge(sequenceSec1, sequenceSec2)

    save_sequence_graph(graph, f'{graph_folder}/{graph_name}')
    return load_sequence_graph(f'{graph_folder}/{graph_name}')

def save_sequence_graph(graph, save_path):
    graph = graph
    with open(save_path, "wb") as f:
        pickle.dump(graph, f)

def load_sequence_graph(graph_path):
    with open(graph_path, "rb") as f:
        return pickle.load(f)
    