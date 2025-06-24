# TODO: Deprecated — legacy support for Vislization. This class will be removed in a future update.
# import numpy as np
# import matplotlib.pyplot as plt
# import open3d as o3d
# from matplotlib.colors import LinearSegmentedColormap, ListedColormap
# # from scipy.spatial import ConvexHull
# # from scipy.spatial import Delaunay


# def plot_agent(ax, grid_resolution, agent_path):
#     for path in agent_path:
#         agent_x, agent_y, agent_rotation = path
#         # Convert agent coordinates according to grid resolution and offset
#         agent_x_plot = (agent_x / grid_resolution)
#         agent_y_plot = (agent_y / grid_resolution)

#         agent_rotation_degrees = agent_rotation * 360
#         ax.scatter(agent_y_plot, agent_x_plot, color='red', s=10, marker=(3, 0, agent_rotation_degrees - 90))  # using a triangle marker


# def plot_elevation_map(elevation_map, grid_resolution, roi, shift_x, shift_y, agent_path, save_path, cmap='terrain'):
#     if roi is None:
#         m = elevation_map
#         x_indices = np.arange(m.shape[0])
#         y_indices = np.arange(m.shape[1])
#     else:
#         roi_x, roi_y, roi_width, roi_height = roi
#         m = elevation_map[roi_x:roi_x + roi_width, roi_y:roi_y + roi_height]
#         x_indices = (np.arange(m.shape[0]) + roi_x)
#         y_indices = (np.arange(m.shape[1]) + roi_y)

#     # Update extent for the plot
#     x_indices = x_indices / grid_resolution + shift_x
#     y_indices = y_indices / grid_resolution + shift_y
#     extent = [min(y_indices), max(y_indices), min(x_indices), max(x_indices)]

#     # Plot 2D elevation map
#     fig, ax = plt.subplots(figsize=(6, 6))
#     im = ax.imshow(m, cmap=cmap, origin='lower', extent=extent)
#     plt.colorbar(im, orientation='vertical', label="Elevation")
#     ax.set_xlabel("X coordinate")
#     ax.set_ylabel("Y coordinate")
#     ax.set_title('2D Elevation Map')
#     ax.invert_yaxis()
#     ax.invert_xaxis()

#     # Check if agent_location is provided and plot it
#     if agent_path:
#         plot_agent(ax, grid_resolution, agent_path)

#     plt.savefig(save_path)
#     plt.show()


# def plot_occupnacy_map(occupancy_map, roi, grid_resolution, shift_x, shift_y, save_path):
#     if roi is None:
#         m = occupancy_map
#         x_indices = np.arange(m.shape[0])
#         y_indices = np.arange(m.shape[1])
#     else:
#         roi_x, roi_y, roi_width, roi_height = roi
#         m = occupancy_map[roi_x:roi_x + roi_width, roi_y:roi_y + roi_height]
#         x_indices = np.arange(m.shape[0]) + roi_x 
#         y_indices = np.arange(m.shape[1]) + roi_y

#     colors = [(0, 'green'), (1, 'red')]  # Define color points
#     cmap = LinearSegmentedColormap.from_list('custom', colors)


#     x_indices = x_indices / grid_resolution + shift_x
#     y_indices = y_indices / grid_resolution + shift_y
#     extent = [min(y_indices), max(y_indices), min(x_indices), max(x_indices)]

#     fig = plt.figure(figsize=(6, 6))
#     plt.imshow(m, cmap=cmap, origin='lower',
#                extent=extent)
#     plt.colorbar(orientation='vertical', label="Elevation")
#     plt.gca().invert_yaxis()
#     plt.gca().invert_xaxis()
#     plt.xlabel("X coordinate")
#     plt.ylabel("Y coordinate")
#     plt.title('Occupancy Map')
#     plt.savefig(save_path)
#     plt.show()

# # def _3d_visualize_elevation_map(max_bound, min_bound, z_max, z_min, highest_points, elevation_map, voxel_size=2):
# #     # Add a grid at the lowest point of z
# #     grid_scale = 0.2
# #     x_size = max_bound[0] - min_bound[0]
# #     y_size = max_bound[1] - min_bound[1]
# #     x_spaces = np.linspace(min_bound[0], max_bound[0], num=int(x_size * grid_scale))
# #     y_spaces = np.linspace(min_bound[1], max_bound[1], num=int(y_size * grid_scale))

# #     lines = []
# #     for i, x in enumerate(x_spaces):
# #         for j, y in enumerate(y_spaces):
# #             if i < len(x_spaces) - 1:
# #                 lines.append([[x, y, z_min], [x_spaces[i + 1], y, z_min]])
# #             if j < len(y_spaces) - 1:
# #                 lines.append([[x, y, z_min], [x, y_spaces[j + 1], z_min]])

# #     line_set = o3d.geometry.LineSet(
# #         points=o3d.utility.Vector3dVector(np.array(lines).reshape(-1, 3)),
# #         lines=o3d.utility.Vector2iVector(np.arange(0, len(lines) * 2).reshape(-1, 2))
# #     )

# #     # Create a cube from 0 to the z-value for points with z-values greater than 0
# #     colormap = plt.get_cmap('jet')
# #     cubes = []
# #     for point in highest_points:
# #         if point[2] > 0:
# #             height = point[2] - z_min
# #             color = colormap((height) / (z_max - z_min))[:3]

# #             cube = o3d.geometry.TriangleMesh.create_box(width=voxel_size,
# #                                                         height=voxel_size,
# #                                                         depth=height)
# #             cube.translate([point[0] - voxel_size / 2, point[1] - voxel_size / 2, z_min])
# #             cube.paint_uniform_color(color)
# #             cubes.append(cube)
# #     mesh = cubes[0]
# #     for c in cubes[1:]:
# #         mesh += c

# #     # Calculate the vertices of convex hulls on the xy plane
# #     xy_points = elevation_map[:, :2]
# #     hull = ConvexHull(xy_points)
# #     hull_indices = hull.vertices

# #     hull_vertices = elevation_map[hull_indices, :]
# #     hull_points = xy_points[hull.vertices]
# #     hull_points_3d = np.hstack((hull_points, np.full((hull_points.shape[0], 1), z_min)))

# #     # Delaunay triangulation of convex hull vertices
# #     delaunay = Delaunay(hull_points)

# #     # Create Open3D triangular meshes
# #     hull_mesh = o3d.geometry.TriangleMesh()

# #     # Add vertex
# #     hull_mesh.vertices = o3d.utility.Vector3dVector(hull_points_3d)

# #     # Add triangular faces
# #     hull_mesh.triangles = o3d.utility.Vector3iVector(delaunay.simplices)

# #     # Calculate the normal vector so that the mesh can render correctly
# #     hull_mesh.compute_vertex_normals()

# #     # Coloring with deep green filling
# #     hull_mesh.paint_uniform_color([0.0, 0.5, 0.2])

# #     o3d.visualization.draw_geometries([line_set, mesh, hull_mesh])

# def plot_elevation_map(elevation_map, grid_resolution, shift_x, shift_y, save_path, cmap='terrain', withCV2 = False, agent_location=[]):
#     '''
#     Plot and save elevation map 

#     :param elevation_map: elevation map of the point cloud
#     :param grid_resolution : resolution used to plot the elevation map
#     :param roi : region of interest used to plot the elevation map
#     :param shift_x: degree of x-coordinate shifted to plot 3d point cloud onto 2d elevation map
#     :param shift_y: degree of y-coordinate shifted to plot 3d point cloud onto 2d elevation map
#     :param agent_location: location of the agent on the elevation map
#     :param save_path: path to save the figure of the elevation map
#     :param cmap: color map used for figure visualization
#     '''

#     elevation_map
#     x_indices = np.arange(elevation_map.shape[0])
#     y_indices = np.arange(elevation_map.shape[1])

#     # Update extent for the plot
#     x_indices = x_indices / grid_resolution + shift_x
#     y_indices = y_indices / grid_resolution + shift_y
#     extent = [min(y_indices), max(y_indices), min(x_indices), max(x_indices)]
    
#     # Plot 2D elevation map
#     fig, ax = plt.subplots(figsize=(6, 6))
#     im = ax.imshow(elevation_map, cmap=cmap, origin='lower', extent=extent)
#     plt.colorbar(im, orientation='vertical', label="Elevation")
#     ax.set_xlabel("Y coordinate")
#     ax.set_ylabel("X coordinate")
#     ax.set_title('2D Elevation Map')
#     ax.invert_yaxis()
    
#     # Check if agent_location is provided and plot it
#     # if agent_location:
#     #     for location in agent_location:
#     #         agent_x, agent_y, agent_rotation = location
#     #         plt.quiver(agent_y, agent_x, np.cos(agent_rotation), np.sin(agent_rotation), scale=17)

#     if agent_location:
#         for location in agent_location:
#             agent_x, agent_y, _ = location
#             plt.plot(agent_y, agent_x, 'ro')  # 'ro' stands for red dots
    
#     plt.savefig(save_path)

#     if withCV2:
#         plt.draw()
#         plt.pause(0.01)  # Pause briefly to allow GUI events to process
#     else:
#         plt.show()

# def plot_occupnacy_map(occupancy_map, grid_resolution, shift_x, shift_y, save_path, cmap='terrain', withCV2 = False, agent_location=[]):
#     '''
#      Plot and save occupancy map 

#     :param occupancy_map: occupancy of the point cloud
#     :param grid_resolution : resolution used to plot the elevation map
#     :param roi : region of interest used to plot the elevation map
#     :param shift_x: degree of x-coordinate shifted to plot 3d point cloud onto 2d elevation map
#     :param shift_y: degree of y-coordinate shifted to plot 3d point cloud onto 2d elevation map
#     :param save_path: path to save the figure of the elevation map
#     '''

#     occupancy_map = point_cloud_to_occupany_map(occupancy_map, 4)

#     x_indices = np.arange(occupancy_map.shape[0])
#     y_indices = np.arange(occupancy_map.shape[1])

#     # Update extent for the plot
#     x_indices = x_indices / grid_resolution + shift_x
#     y_indices = y_indices / grid_resolution + shift_y
#     extent = [min(y_indices), max(y_indices), min(x_indices), max(x_indices)]

#     # Define a binary colormap
#     cmap = ListedColormap(['white', 'black'])

#     # Plot 2D occupancy map
#     fig, ax = plt.subplots(figsize=(6, 6))
#     im = ax.imshow(occupancy_map, cmap=cmap, origin='lower', extent=extent, vmin=0, vmax=1)
#     plt.colorbar(im, orientation='vertical', label="Occupancy", ticks=[0, 1])
#     ax.set_xlabel("Y coordinate")
#     ax.set_ylabel("X coordinate")
#     ax.set_title('2D Occupancy Map')
#     ax.invert_yaxis()

#     # Check if agent_location is provided and plot it
#     # if agent_location:
#     #     for location in agent_location:
#     #         agent_x, agent_y, agent_rotation = location
#     #         plt.quiver(agent_y, agent_x, np.cos(-agent_rotation - np.pi), np.sin(-agent_rotation - np.pi), scale=17)

#     plt.savefig(save_path)

#     if withCV2:
#         plt.draw()
#         plt.pause(0.01)  # Pause briefly to allow GUI events to process
#     else:
#         plt.show()

# def point_cloud_to_occupany_map(elevation_map, threshold):
#     '''
#     Compute the occupancy map given the elevation map

#     :param elevation_map: point cloud on 2d plane
#     :param threshold: threshold to determine the existence of an obstacle

#     :return: occupancy
#     '''
#     temp = elevation_map
#     rows = len(elevation_map)
#     columns = len(elevation_map[0])
#     for i in range(rows):
#         for j in range(columns):
#             point = elevation_map[i][j]
#             if (point >= threshold):
#                 temp[i][j] = 1
#             else:
#                 temp[i][j] = 0
#     return temp

# def plot_elevation_map_io(elevation_map, grid_resolution, shift_x, shift_y, save_path, cmap='terrain', saveOnly = False, agent_location=[]):
#     '''
#     Plot and save elevation map 

#     :param elevation_map: elevation map of the point cloud
#     :param grid_resolution : resolution used to plot the elevation map
#     :param roi : region of interest used to plot the elevation map
#     :param shift_x: degree of x-coordinate shifted to plot 3d point cloud onto 2d elevation map
#     :param shift_y: degree of y-coordinate shifted to plot 3d point cloud onto 2d elevation map
#     :param agent_location: location of the agent on the elevation map
#     :param save_path: path to save the figure of the elevation map
#     :param cmap: color map used for figure visualization
#     '''

#     x_indices = np.arange(elevation_map.shape[0])
#     y_indices = np.arange(elevation_map.shape[1])

#     # Update extent for the plot
#     x_indices = x_indices / grid_resolution + shift_x
#     y_indices = y_indices / grid_resolution + shift_y
#     extent = [min(y_indices), max(y_indices), min(x_indices), max(x_indices)]
    
#     # Plot 2D elevation map
#     fig, ax = plt.subplots(figsize=(6, 6))
#     im = ax.imshow(elevation_map, cmap=cmap, origin='lower', extent=extent)
#     plt.colorbar(im, orientation='vertical', label="Elevation")
#     ax.set_xlabel("Y coordinate")
#     ax.set_ylabel("X coordinate")
#     ax.set_title('2D Elevation Map')
#     # ax.invert_yaxis()
#     # ax.invert_xaxis()
    
#     # Check if agent_location is provided and plot it
#     if agent_location:
#         for location in agent_location:
#             agent_x, agent_z, agent_y, agent_rotation = location
#             plt.quiver(agent_z, agent_x, np.cos(agent_rotation), np.sin(agent_rotation), scale=17)

#     buffer = BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     plt.close(fig)

#     return buffer