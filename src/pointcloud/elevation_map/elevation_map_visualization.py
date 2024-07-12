import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyArrowPatch

def plot_elevation_map(elevation_map, grid_resolution, shift_x, shift_y, save_path, cmap='terrain', withCV2 = False, agent_location=[]):
    '''
    Plot and save elevation map 

    :param elevation_map: elevation map of the point cloud
    :param grid_resolution : resolution used to plot the elevation map
    :param roi : region of interest used to plot the elevation map
    :param shift_x: degree of x-coordinate shifted to plot 3d point cloud onto 2d elevation map
    :param shift_y: degree of y-coordinate shifted to plot 3d point cloud onto 2d elevation map
    :param agent_location: location of the agent on the elevation map
    :param save_path: path to save the figure of the elevation map
    :param cmap: color map used for figure visualization
    '''

    elevation_map
    x_indices = np.arange(elevation_map.shape[0])
    y_indices = np.arange(elevation_map.shape[1])

    # Update extent for the plot
    x_indices = x_indices / grid_resolution + shift_x
    y_indices = y_indices / grid_resolution + shift_y
    extent = [min(y_indices), max(y_indices), min(x_indices), max(x_indices)]
    
    # Plot 2D elevation map
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(elevation_map, cmap=cmap, origin='lower', extent=extent)
    plt.colorbar(im, orientation='vertical', label="Elevation")
    ax.set_xlabel("Y coordinate")
    ax.set_ylabel("X coordinate")
    ax.set_title('2D Elevation Map')
    ax.invert_yaxis()
    
    # Check if agent_location is provided and plot it
    # if agent_location:
    #     for location in agent_location:
    #         agent_x, agent_y, agent_rotation = location
    #         plt.quiver(agent_y, agent_x, np.cos(agent_rotation), np.sin(agent_rotation), scale=17)

    if agent_location:
        for location in agent_location:
            agent_x, agent_y, _ = location
            plt.plot(agent_y, agent_x, 'ro')  # 'ro' stands for red dots
    
    plt.savefig(save_path)

    if withCV2:
        plt.draw()
        plt.pause(0.01)  # Pause briefly to allow GUI events to process
    else:
        plt.show()

def plot_occupnacy_map(occupancy_map, grid_resolution, shift_x, shift_y, save_path, cmap='terrain', withCV2 = False, agent_location=[]):
    '''
     Plot and save occupancy map 

    :param occupancy_map: occupancy of the point cloud
    :param grid_resolution : resolution used to plot the elevation map
    :param roi : region of interest used to plot the elevation map
    :param shift_x: degree of x-coordinate shifted to plot 3d point cloud onto 2d elevation map
    :param shift_y: degree of y-coordinate shifted to plot 3d point cloud onto 2d elevation map
    :param save_path: path to save the figure of the elevation map
    '''

    occupancy_map = point_cloud_to_occupany_map(occupancy_map, 4)

    x_indices = np.arange(occupancy_map.shape[0])
    y_indices = np.arange(occupancy_map.shape[1])

    # Update extent for the plot
    x_indices = x_indices / grid_resolution + shift_x
    y_indices = y_indices / grid_resolution + shift_y
    extent = [min(y_indices), max(y_indices), min(x_indices), max(x_indices)]

    # Define a binary colormap
    cmap = ListedColormap(['white', 'black'])

    # Plot 2D occupancy map
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(occupancy_map, cmap=cmap, origin='lower', extent=extent, vmin=0, vmax=1)
    plt.colorbar(im, orientation='vertical', label="Occupancy", ticks=[0, 1])
    ax.set_xlabel("Y coordinate")
    ax.set_ylabel("X coordinate")
    ax.set_title('2D Occupancy Map')
    ax.invert_yaxis()

    # Check if agent_location is provided and plot it
    # if agent_location:
    #     for location in agent_location:
    #         agent_x, agent_y, agent_rotation = location
    #         plt.quiver(agent_y, agent_x, np.cos(-agent_rotation - np.pi), np.sin(-agent_rotation - np.pi), scale=17)

    plt.savefig(save_path)

    if withCV2:
        plt.draw()
        plt.pause(0.01)  # Pause briefly to allow GUI events to process
    else:
        plt.show()

def point_cloud_to_occupany_map(elevation_map, threshold):
    '''
    Compute the occupancy map given the elevation map

    :param elevation_map: point cloud on 2d plane
    :param threshold: threshold to determine the existence of an obstacle

    :return: occupancy
    '''
    temp = elevation_map
    rows = len(elevation_map)
    columns = len(elevation_map[0])
    for i in range(rows):
        for j in range(columns):
            point = elevation_map[i][j]
            if (point >= threshold):
                temp[i][j] = 1
            else:
                temp[i][j] = 0
    return temp
