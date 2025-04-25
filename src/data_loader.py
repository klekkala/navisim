import pickle
import networkx as nx
import json
import io
import numpy as np
import open3d as o3d

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


def plot_elevation_map(data, ax, title, label, polygon = None, position = None):
    im = ax.imshow(
        data,
        cmap='viridis',
        vmax=10,
        origin='lower',
        aspect='equal'
    )

    if polygon:
        x, y = polygon.exterior.xy
        ax.plot(x, y, color='red', linewidth=2)
    
    if position:
        x, y = position
        ax.plot(x, y, marker='o', color='red', markersize=6)
        ax.text(x + 1, y + 1, f'Agent({x},{y})', color='black', fontsize=8)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label=label)


def plot_occupancy_map(data, ax, title, label, polygon = None, position = None):
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
        ax.plot(x, y, color='red', linewidth=2)
    
    if position:
        x, y = position
        ax.plot(x, y, marker='o', color='red', markersize=2)
        ax.text(x + 1, y + 1, f'Agent({x},{y})', color='black', fontsize=8)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label=label)

def plot_pcd(pcd, ax, title):
    points = np.asarray(pcd.points)
    if points.size == 0:
        ax.text2D(0.3, 0.5, "Empty Point Cloud", transform=ax.transAxes)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue', alpha=0.6)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)


def plot_maps(elevation_map, occupancy_map, polygon, pos, pcd = None):
    """
    Plots an elevation map on a given Axes object.
    """
    fig = plt.figure(figsize=(21, 6))

    # 2D axes
    ax1 = fig.add_subplot(1, 3 if pcd else 2, 1)
    ax2 = fig.add_subplot(1, 3 if pcd else 2, 2)

    # 3D axis
    if pcd:
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    plot_elevation_map(elevation_map, ax1, title = 'Elevation Map', label = 'Elevation', polygon = polygon, position = pos)
    plot_occupancy_map(occupancy_map, ax2, title = 'Occupancy Map', label = 'Occupancy', polygon = polygon, position = pos)
    if pcd :
        plot_pcd(pcd, ax3, title= 'PointCloud')
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

def get_gaussian_splat(node_id, sector_id):
    date, session = node_id.split('/')
    key = {
        "date": date,
        'session': session,
        'sector': sector_id,
        'file_name': 'gaussian'
    }
    data  = RocksDB().get(json.dumps(key))
    file_name = f'{date}_{session}_{sector_id}.ply'
    with open(f'{date}_{session}_{sector_id}.ply', "wb") as f:
        f.write(data)
    
    try:
        pcd = o3d.io.read_point_cloud(file_name)
        if len(pcd.points) == 0:
                print("❌ Loaded PLY file, but contains no points.")
        else:
            print(f"✅ Loaded PLY file with {len(pcd.points)} points.")
        return pcd
    except Exception as e:
        print("❌ Failed to load Gaussian splat:", e)
        
    
    

# sequence_graph = load_sequence_graph('/lab/kiran/navisim-1/assets/sequence_graph.gpickle')
# plot_seq_graph(sequence_graph, out_path='/lab/kiran/navisim-1/assets/test')
# sectors = get_sectors(sequence_graph)

# sector_id = list(sectors.keys())[0]
# print(sector_id, sectors[sector_id][0])
# # get_elevation_map(sector_id, sectors[sector_id][0])

# plot_elevation_map(elev1, axes[0], title="Elevation")
# plot_elevation_map(elev2, axes[1], title="Occupancy")

