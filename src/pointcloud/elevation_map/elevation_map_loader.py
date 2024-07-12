import numpy as np
import open3d as o3d
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import scipy.interpolate
import matplotlib.pyplot as plt

def multiple_pass_interpolation(data, max_passes=5):
    interpolated = np.copy(data)
    rows, cols = data.shape

    def interpolate_once(input_data):
        output_data = np.copy(input_data)
        for i in range(rows):
            for j in range(cols):
                if input_data[i, j] <= 0:
                    neighbors = []
                    for x in [i-1, i, i+1]:
                        for y in [j-1, j, j+1]:
                            if 0 <= x < rows and 0 <= y < cols:
                                if input_data[x, y] > 0:
                                    neighbors.append(input_data[x, y])
                    if neighbors:
                        output_data[i, j] = sum(neighbors) / len(neighbors)
        return output_data
    
    for _ in range(max_passes):
        new_data = interpolate_once(interpolated)
        if np.array_equal(new_data, interpolated):
            break
        interpolated = new_data

    return interpolated

def iterative_interpolation(data, max_iterations=4):
    for _ in range(max_iterations):
        data = multiple_pass_interpolation(data)
    return data

def interpolation(data):
    mask = np.isnan(data)
    x, y = np.indices(data.shape)
    known_points = np.array((x[~mask], y[~mask])).T
    known_values = data[~mask]
    nan_points = np.array((x[mask], y[mask])).T

    # Initial interpolation using nearest-neighbor to handle edges
    nearest_interpolated_values = scipy.interpolate.griddata(known_points, known_values, nan_points, method='nearest')
    data[mask] = nearest_interpolated_values

    # Refine interpolation using linear method
    mask = np.isnan(data)  # Update mask after nearest interpolation
    if np.any(mask):
        nan_points = np.array((x[mask], y[mask])).T
        linear_interpolated_values = scipy.interpolate.griddata(known_points, known_values, nan_points, method='linear')
        data[mask] = linear_interpolated_values

    # Apply Gaussian smoothing to improve smoothness
    smoothed_data = scipy.ndimage.gaussian_filter(data, sigma=1)

    return smoothed_data

def interpolate_and_smooth_elevation(elevation_data, max_distance = 2, method = 'linear'):
    """
    Interpolate missing values in an elevation map and smooth the result.
    
    Parameters:
        elevation_data (np.ndarray): 2D array of elevation data with np.nan representing missing values.
        max_distance (float): Maximum distance to consider for interpolation. Points further than this distance from known points will not be interpolated.
        method (str): Interpolation method to use ('linear', 'nearest', 'cubic').
    
    Returns:
        np.ndarray: The elevation data with interpolated values and smoothed.
        np.ndarray: The smoothed elevation data.
    """

    x, y = np.indices(elevation_data.shape)

    # Mask of known points (non-NaN)
    known_mask = ~np.isnan(elevation_data)

    # Known coordinates and values
    known_coords = np.array([x[known_mask], y[known_mask]]).T
    known_values = elevation_data[known_mask]

    # Coordinates of missing points
    missing_coords = np.array([x[~known_mask], y[~known_mask]]).T

    # Build a KDTree for the known coordinates
    tree = cKDTree(known_coords)

    # Query the tree for the nearest neighbors within the max distance
    distances, _ = tree.query(missing_coords, distance_upper_bound=max_distance)

    # Only interpolate points within the max distance
    valid_mask = distances != np.inf
    valid_missing_coords = missing_coords[valid_mask]

    # Interpolate missing values only for valid coordinates
    interpolated_values = griddata(known_coords, known_values, valid_missing_coords, method=method)

    # Create a copy of the elevation array to fill with interpolated values
    filled_elevation = elevation_data.copy()
    filled_elevation[tuple(valid_missing_coords.T)] = interpolated_values

    # Clamp interpolated values to the range of the known data
    min_val = np.nanmin(elevation_data)
    max_val = np.nanmax(elevation_data)
    filled_elevation = np.clip(filled_elevation, min_val, max_val)

    # Ensure there are no NaNs before smoothing
    if np.isnan(filled_elevation).any():
        filled_elevation = np.nan_to_num(filled_elevation, nan=min_val)

    # Apply Gaussian filter for smoothing
    smoothed_elevation = gaussian_filter(filled_elevation, sigma=1)
    # smoothed_elevation = np.round(smoothed_elevation).astype(int)

    return smoothed_elevation

def smooth_out_point_cloud(points, axis_height, min_height, max_height):
    '''
    Smoothes point cloud by removing abnormalities

    :param points: point cloud coordinates
    :param axis_height: axis of the point cloud that represents the height
    :param max_val: maximum height for clipping
    
    :return: smoothened coorindates
    '''
    points = points[(points[:, axis_height] >= min_height) & (points[:, axis_height] <= max_height)]
    return points

def point_cloud_to_height_map(points, grid_lower_bound, grid_width, grid_height, min_height, grid_resolution):
    '''
    Convert 3d point cloud onto 2d elevation map by plotting height on 2d plane
    Inverse Distance Weighting (IDW), with some adaptations to include nearest-neighbor checks using KDTree and post-processing for smoothing and rounding
    
    :param points: point cloud coordinates
    :param grid_width: width of the 2d plane
    :param grid_height: height of the 2d plane
    :param min_height: minimum height of the original point cloud
    :param grid_resolution: resolution used to display point cloud on 2d plane

    :return: elevation map of the point cloud
    '''
    index_x = 0
    index_y = 1 # index of the height coordinate
    index_z = 2 

    # Create 2D top-view grid
    _2d_map = np.full((grid_width, grid_height), np.nan)
    # _2d_map = np.full((grid_width, grid_height), 0)

    # Assign elevation values to the grid
    for point in points:
        # Below two lines will plot the x-y coordinates captured by the sensor to the 2D matrix
        x_idx = int((point[index_x] - grid_lower_bound[index_x]) * grid_resolution)
        y_idx = int((point[index_z] - grid_lower_bound[index_z]) * grid_resolution)

        if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
            # _2d_map[x_idx, y_idx] = point[index_y] + np.abs(min_height)
            if ((point[index_y] + np.abs(min_height)) < 7):
                _2d_map[x_idx, y_idx] = point[index_y] + np.abs(min_height)

    return _2d_map

def get_point_cloud(file_path):
    '''
    Reads point cloud file
    :param file_path: relative path of the point cloud file
    :return: point cloud file
    '''
    file_path = "src/assets/surfaceMap.pcd"
    # file_path = "src/assets/surfaceMap_clean.pcd"
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def crop_elevation_map(elevation_map, x_start, x_end, y_start, y_end):
    cropped_map = elevation_map[x_start:x_end, y_start:y_end]
    return cropped_map

def get_elevation_map(point_cloud, height_limit=10, grid_resolution=10):
    '''
    Compute 3d point cloud onto 2d elevation map

    :param point_cloud: 3d point cloud
    :param height_limit: height threshold to clip any height above the limit
    :param grid_resolution: resolution used for visualizing elevation map

    :return: elevation map

        x is the row informations.
        z is the column informations.

        Min offset of row, Min offset of col
        min_bound[index_x], min_bound[index_z]
    '''
    index_x = 0
    index_y = 1 # index of the height coordinate
    index_z = 2

    point_cloud_np = np.asarray(point_cloud.points)

    min_bound = np.rint(point_cloud.get_min_bound()).astype(int)
    max_bound = np.rint(point_cloud.get_max_bound()).astype(int)

    point_cloud_np = smooth_out_point_cloud(point_cloud_np, axis_height=index_y, min_height=-10, max_height=height_limit)

    grid_width  = (np.abs(max_bound[index_x]) + np.abs(min_bound[index_x])) * grid_resolution
    grid_height = (np.abs(max_bound[index_z]) + np.abs(min_bound[index_z])) * grid_resolution

    # Extract the highest points
    max_height = np.max(point_cloud_np[:, index_y])
    min_height = np.min(point_cloud_np[:, index_y])
    # min_height = 0

    elevation_map = point_cloud_to_height_map(point_cloud_np, grid_lower_bound=min_bound, grid_width=grid_width,
                                              grid_height=grid_height, min_height=min_height,
                                              grid_resolution=grid_resolution)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Default values
    # recommend to be in square shape
    # x_start, x_end, y_start, y_end = -100, -60, 138, 178
    # x_start, x_end, y_start, y_end = -130, -20, -20, 80

    # sec1
    # x_start, x_end = -120, -40
    # y_start, y_end = 120, 198

    # sec2
    x_start, x_end = -120, -40
    y_start, y_end = 160, 220

    elevation_map = crop_elevation_map(elevation_map, (x_start + np.abs(min_bound[index_x])) * grid_resolution, (x_end + np.abs(min_bound[index_x])) * grid_resolution,
                                                        (y_start + np.abs(min_bound[index_z])) * grid_resolution, (y_end + np.abs(min_bound[index_z])) * grid_resolution)
 
    shift_x = x_start
    shift_y = y_start

    # shift_x = min_bound[index_x]
    # shift_y = min_bound[index_z]

    elevation_map = interpolation(elevation_map)
    # elevation_map = iterative_interpolation(elevation_map)

    min_elevation = np.min(elevation_map[:, index_y])
    return elevation_map, shift_x, shift_y, min_elevation

def get_coverage_roi(points, invalid_height, coverage):
    '''
    Compute the region of interest given the coverage 

    :param points: point cloud coordinates on 2d plane
    :param coverage: percent of the valid coodinates needs to be covered by the roi
    :param invalid_height: height that needs to be filtered before computing the roi

    :return: numpy array that represents the roi of the targeted region
    '''
    non_zero = np.transpose(np.where(points != invalid_height))
    center = np.array(points.shape) / 2
    distances = np.linalg.norm(non_zero - center, axis=1)

    sorted_indices = non_zero[np.argsort(distances)]
    num_total_points = len(sorted_indices)
    num_points_to_select = int(num_total_points * coverage)

    selected_indices = sorted_indices[:num_points_to_select]
    min_x, min_y = np.min(selected_indices, axis=0)
    max_x, max_y = np.max(selected_indices, axis=0)

    roi = np.array([min_x, min_y, max_x - min_x + 1, max_y - min_y + 1], dtype='int')
    return roi

def point_cloud_to_occupany_map(elevation_map, threshold):
    '''
    Compute the occupancy map given the elevation map

    :param elevation_map: point cloud on 2d plane
    :param threshold: threshold to determine the existence of an obstacle

    :return: occupancy
    '''
    elevation_map[elevation_map >= threshold] = 1
    elevation_map[elevation_map < threshold] = 0
    return elevation_map
