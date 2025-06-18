import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def visualize_occupancy_map(map, save_path=None, boundary = None, title = "Occupancy Map"):
    """
    Plots a binary occupancy map and saves it to a file.
    """
    
    plt.figure(figsize=(10, 8))
    plt.imshow(
        map,
        cmap='binary',  # Binary colormap (0 = white, 1 = black)
        vmin=0,
        vmax=1,
        origin='lower',
        aspect='equal'
    )
    
    if boundary:
        x, y = boundary.exterior.xy
        plt.plot(x, y, color='red', linewidth=2)

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    plt.close()

def visualize_elevation_map(map, boundary = None, save_path=None, title = "Elevation Map"):
    """
    Plots a map with an overlay of contour lines.
    
    Parameters:
    - data (2D array-like): The data for which the map and contours are to be plotted.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(map, cmap='viridis',  origin='lower')
    
    if boundary:
        x, y = boundary.exterior.xy
        plt.plot(x, y, color='red', linewidth=2)
    
    plt.title(title)
    plt.tight_layout()
    plt.axis('off')
    plt.colorbar(label='Elevation')
    plt.xlabel('X')
    plt.ylabel('Y')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    plt.close()

def visualize_seq_graph(graph, out_path, title = 'sequence_graph'):
    os.makedirs(out_path, exist_ok=True)
    plt.figure(figsize=(8, 6))  # Adjust the figure size
    
    # Get positions for a spring layout (better for dense graphs)
    pos = nx.spring_layout(graph)  
    
    # Draw the graph
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=100,
        node_color="skyblue",
        font_size=5,
        font_color="black",
        edge_color="gray",
    )
    
    # Add a title
    plt.title(title, fontsize=14)
    plt.savefig(f'{out_path}/{title.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_birdeye_view_map(map, out_path, title ='sequence_birdeye_view'):
    os.makedirs(out_path, exist_ok=True)

    plt.figure()
    cmap = cm.get_cmap('tab10', len(map))  # Use a colormap with a fixed number of colors
    
    for i, (key, positions) in enumerate(map.items()):
        # Extract x, y, z from positions
        x = positions[:, 0]
        y = positions[:, 2]
        # color = cmap(i)
        
        # Create scatter plot
        plt.scatter(x, y, s=1, linewidth=0.01)
        plt.text(x[-1], y[-1], key, fontsize = 4, ha = 'right', va='bottom')

    # Add labels and title
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    plt.savefig(f'{out_path}/{title.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_3d_scatter_map(map, out_path, title="sequence_3D_trajectories"):
    """
    Plots a 3D scatter graph of x, y, z coordinates to show the traveled paths.

    :param positions: Array of shape (N, 3), where each row is [x, y, z].
    :param title: Title of the 3D plot.
    """
    os.makedirs(out_path, exist_ok=True)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    cmap = cm.get_cmap('tab10', len(map))  # Use a colormap with a fixed number of colors
    
    for i, (key, positions) in enumerate(map.items()):
        # Extract x, y, z from positions
        x = positions[:, 0]
        y = positions[:, 2]
        z = positions[:, 1]
        # color = cmap(i)
        
        # Create scatter plot
        scatter = ax.scatter(x, y, z, label = key, s=1)

    # Add a color bar to represent the path sequence
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label('Path Progression')

    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    plt.savefig(f'{out_path}/{title.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()