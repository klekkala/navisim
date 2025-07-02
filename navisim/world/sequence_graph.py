import pickle
from collections import defaultdict

try:
    from ..world.sector import Sector
except ImportError: 
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from navisim.world.sector import Sector

class SequenceGraph:
    def __init__(self, path):
        self.graph = self._load_sequence_graph(path)
        self.sequences = self._load_get_sequences(self.graph)
    
    def __getattr__(self, name):
        # Called only if the attribute is not found in self
        return getattr(self.graph, name)

    def get_sequence(self, seq_id):
        return self.sequences[seq_id]
    
    def get_sequence_ids(self):
        return list(self.sequences.keys())

    def _load_get_sequences(self, sequence_graph):
        sequences = defaultdict(list)
        for node, data in sequence_graph.nodes(data=True):
            sequences[node] = self._load_sectors(node, data['sectors'])
        return sequences
    
    def _load_sectors(self, seq_id, sector_ids):
        sectors = [Sector(seq_id, sec_id) for sec_id in sector_ids]
        for i, sector in enumerate(sectors):
            if i > 0:
                sector.prev = sectors[i - 1]
            if i < len(sectors) - 1:
                sector.next = sectors[i + 1]
        return sectors

    def _load_sequence_graph(self, graph_path):
        try:
            with open(graph_path, 'rb') as f:
                sequence_graph = pickle.load(f)
            return sequence_graph
        except FileNotFoundError:
            print(f'{graph_path} is not available')

    