import pickle
import networkx as nx

from util.graph_visualizer import *
from database.rocksdb import RocksDB

def load_sequence_graph(graph_path):
    try:
        with open(graph_path, 'rb') as f:
            sequence_graph = pickle.load(f)
        return sequence_graph
    except FileNotFoundError:
        print(f'{graph_path} is not available')

def check_graph(graph):
    for node, data in sequence_graph.nodes(data=True):
        print(f"Node {node}", data.keys())
        print(len(data['polygon']))
    

sequence_graph = load_sequence_graph('/lab/kiran/navisim-1/assets/sequence_graph.gpickle')
plot_seq_graph(sequence_graph, out_path='/lab/kiran/navisim-1/assets/test')
check_graph(sequence_graph)