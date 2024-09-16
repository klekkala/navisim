import unittest
import os
from unittest.mock import MagicMock
from src.paths import SEQUENCE_GRAPH_FOLDER

from src.beogym.sequence_graph.sequence_graph_test import *


class TestBeoGym(unittest.TestCase):
    def setUp(self):
        self.mock_sequence_graph = MagicMock()
        self.mock_sequence_graph.save = BeogymSequenceGraphTest.save

        self.mock_sequence_graph_node = MagicMock()
        return True

    def test_sequence_graph_load(self):
        loaded_sequence_graph = load()
        self.assertIsInstance(loaded_sequence_graph, BeogymSequenceGraphTest)

    def test_sequence_graph_node(self):
        self.mock_sequence_graph_node.__init__(self.mock_sequence_graph_node, height_limit=0.5, grid_resolution=10)
        self.assertTrue(True)
