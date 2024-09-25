import unittest
import os
import datetime

from unittest.mock import MagicMock

import sys
sys.path.append('/home/student/motion_model/jiwon/navisim')

from src.beogym.sequence_graph.sequence_graph import *

class TestBeoGym(unittest.TestCase):
    def setUp(self):
        self.sequence_graph = BeogymSequenceGraph(global_point_cloud=None)
        self.graph_name = 'test_graph.pkl'
        self.dummy_node1 = Node(id = 1)
        self.dummy_node2 = Node(id = 2)

    def test_sequence_graph_save(self):
        self.sequence_graph.save(self.graph_name)
        
        save_path = os.path.join(SEQUENCE_GRAPH_FOLDER, self.graph_name)
        result = os.path.isfile(save_path)
        self.assertTrue(result)

    def test_sequence_graph_load(self):
        loaded_sequence_graph = load_saved_sequence_graph(self.graph_name)
        self.assertIsInstance(loaded_sequence_graph, BeogymSequenceGraph)


    def test_sequence_graph_add_node(self):
        self.sequence_graph.add_node(self.dummy_node1)
        self.assertTrue(len(self.sequence_graph.nodes) == 1)

        self.sequence_graph.add_node(self.dummy_node2)
        self.assertTrue(len(self.sequence_graph.nodes) == 2)


    def test_sequence_graph_save_graph_with_nodes(self):
        test_sequence_graph = BeogymSequenceGraph(global_point_cloud=None)
        graph_name = 'test_graph_with_nodes.pkl'
        num_nodes = 10
        for i in range(num_nodes):
            node = Node(id = 1 if i % 2 == 0 else 2)
            print(f'node_saved : {i}')
            test_sequence_graph.add_node(node)

        # Test whether the graph is saved with the nodes appended
        start_time = datetime.datetime.now()
        test_sequence_graph.save(graph_name)
        end_time = datetime.datetime.now()
        time_elapsed = end_time - start_time

        print('time elapsed to save:', f'{time_elapsed.microseconds} microseconds')
        save_path = os.path.join(SEQUENCE_GRAPH_FOLDER, self.graph_name)
        result = os.path.isfile(save_path)
        self.assertTrue(result)

        # test whether the graph with node is loaded well
        start_time = datetime.datetime.now()
        loaded_sequence_graph = load_saved_sequence_graph(graph_name)
        end_time = datetime.datetime.now()
        time_elapsed = end_time - start_time

        print('time elapsed to load:', f'{time_elapsed.microseconds} microseconds')

        self.assertIsInstance(loaded_sequence_graph, BeogymSequenceGraph)
        num_nodes = len(loaded_sequence_graph.nodes)
        self.assertTrue(num_nodes == num_nodes)
        
        node = list(loaded_sequence_graph.nodes)[0]
        self.assertTrue(type(node.guassian_splat), o3d.geometry.PointCloud)


if __name__ == '__main__':
    unittest.main()
