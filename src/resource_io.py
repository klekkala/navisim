import pickle
import networkx as nx
import json
import io
import numpy as np
from shapely import wkb

from collections import defaultdict
from util.graph_visualizer import *
from database.rocksdb import RocksDB

def load_sequence_graph(graph_path):
    try:
        with open(graph_path, 'rb') as f:
            sequence_graph = pickle.load(f)
        return sequence_graph
    except FileNotFoundError:
        print(f'{graph_path} is not available')

def get_sequence(sequence_graph):
    sectors = defaultdict(list)
    for node, data in sequence_graph.nodes(data=True):
        sectors[node] = data['sectors']
    return sectors


def plot_elevation_map(data, ax, title, label, polygon = None):
    im = ax.imshow(
        data,
        cmap='viridis',
        vmax=10,
        origin='lower',
        aspect='equal'
    )

    if polygon:
        x, y = polygon.exterior.xy
        plt.plot(x, y, color='red', linewidth=2)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label=label)


def plot_occupancy_map(data, ax, title, label, polygon = None):
    """
    Plots a binary occupancy map and saves it to a file.
    """
    im = ax.imshow(
        data,
        cmap='binary',
        vmin=0,
        vmax=1,
        origin='lower',
        aspect='equal'
    )

    if polygon:
        x, y = polygon.exterior.xy
        plt.plot(x, y, color='red', linewidth=2)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label=label)

def plot_maps(elevation_map, occupancy_map, polygon):
    """
    Plots an elevation map on a given Axes object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_elevation_map(elevation_map, axes[0], title = 'Elevation Map', label = 'Elevation', polygon = polygon)
    plot_occupancy_map(occupancy_map, axes[1], title = 'Occupancy Map', label = 'Occupancy', polygon = polygon)
    plt.tight_layout()
    plt.show()


def get_map(date, session, sector, name):
    key = {
        "date": date,
        'session': session,
        'sector': sector,
        'file_name': name
    }   
    raw_data = RocksDB().get(json.dumps(key))
    buffer = io.BytesIO(raw_data)
    return np.load(buffer, allow_pickle=True)

def get_saved_polygon(date, session, sector):
    key = {
        "date": date,
        'session': session,
        'sector': sector,
        'file_name': 'boundary'
    }   
    data = RocksDB().get(json.dumps(key))
    return wkb.loads(data)

def get_elevation_map(node_id, sector_id):
    date, session = node_id.split('/')
    elevation_map = get_map(date, session, sector_id, 'elevation')
    return elevation_map

def get_occupancy_map(node_id, sector_id):
    date, session = node_id.split('/')
    occupancy_map = get_map(date, session, sector_id, 'occupancy')
    return occupancy_map

def get_polygon(node_id, sector_id):
    date, session = node_id.split('/')
    return get_saved_polygon(date, session, sector_id)


# sequence_graph = load_sequence_graph('/lab/kiran/navisim-1/assets/sequence_graph.gpickle')
# plot_seq_graph(sequence_graph, out_path='/lab/kiran/navisim-1/assets/test')
# sectors = get_sectors(sequence_graph)

# sector_id = list(sectors.keys())[0]
# print(sector_id, sectors[sector_id][0])
# # get_elevation_map(sector_id, sectors[sector_id][0])

# plot_elevation_map(elev1, axes[0], title="Elevation")
# plot_elevation_map(elev2, axes[1], title="Occupancy")

