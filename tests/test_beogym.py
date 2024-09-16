import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from unittest.mock import MagicMock
from beogym.sequence_graph.sequence_graph import *

class TestBeoGym(unittest.TestCase):
    def setUp(self):
        self.mock_sequence_graph = MagicMock()
        self.mock_sequence_graph.save = BeogymSequenceGraph.save
        return True
    
    def test_sequence_graph_save(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, '..', 'cache',
                                 'beogym_sequence_graph.pkl') if not save_path else save_path
        
        self.mock_sequence_graph.save(save_path = save_path)
        self.assertTrue(os.path.isfile(save_path))

    def test_sequence_graph_load(self):
        graph_name = 'beogym_sequence_graph.pkl'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, '..', 'cache',
                                 graph_name) if not save_path else save_path
        
        self.mock_sequence_graph.save(save_path = save_path)
        graph_loaded = load_saved_sequence_graph(graph_name)
        self.assertIsNotNone(graph_loaded)
