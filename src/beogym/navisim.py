import networkx as nx
from experimental.data_loader import *

class Navisim:
    @abstractmethod
    def load_maps(self):
        pass

    @abstractmethod
    def move(self):
        pass


class NavisimImple(Navisim):
    def __init__(self, sequence_graph):
        return

    def load_maps(self, trajectory_id):
        get_elevation_map(seq_id, str(sector_id))
    
    
    
