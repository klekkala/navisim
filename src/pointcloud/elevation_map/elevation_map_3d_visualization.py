# import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt

# from scipy.spatial import ConvexHull
# from scipy.spatial import Delaunay


# def visualize_3d(point_cloud, height_axis=1, voxel_size=1):
#     '''
#     3d visualization point cloud

#     :param voxel_size : size of the cube used for visualizing height
#     :param height_axis : axis of point cloud that represents the height
#     '''
#     point_cloud_np = np.asarray(point_cloud.points)

#     min_bound = point_cloud.get_min_bound()
#     max_bound = point_cloud.get_max_bound()
    
#     z_max = np.max(point_cloud_np[:, height_axis])
#     z_min = np.min(point_cloud_np[:, height_axis])

#     point_cloud_np[:, [1, 2]] = point_cloud_np[:, [2, 1]]
#     line_set = get_line_set(z_min, z_max, min_bound, max_bound)
#     cubes = get_cubes(point_cloud_np, z_min, z_max, voxel_size=voxel_size)
#     hull_mesh = get_hulls(point_cloud_np, z_min)

#     mesh = cubes[0]
#     for c in cubes[1:]:
#         mesh += c

#     o3d.visualization.draw_geometries([line_set, mesh, hull_mesh])


# def get_line_set(min_height, max_height, min_bound, max_bound, grid_scale=0.1):
#     '''
#     Compute line set to be used as grid of the 3d visualization of the point cloud

#     :param min_height: min height in the point cloud after processing
#     :param min_bound: min x,y,z values of the point cloud before processing
#     :param max_bound: max x,y,z values of the point cloud before processing
#     :param grid_scale : scale used to scale up or scale down the visualization
    
#     :return: list of line set
#     '''
#     x_size = max_bound[0] - min_bound[0]
#     y_size = max_bound[1] - min_bound[1]
#     x_spaces = np.linspace(min_bound[0], max_bound[0], num=int(x_size * grid_scale))
#     y_spaces = np.linspace(min_bound[1], max_bound[1], num=int(y_size * grid_scale))

#     lines = []
#     for i, x in enumerate(x_spaces):
#         for j, y in enumerate(y_spaces):
#             if i < len(x_spaces) - 1:
#                 lines.append([[x, y, min_height], [x_spaces[i + 1], y, min_height]])
#             if j < len(y_spaces) - 1:
#                 lines.append([[x, y, min_height], [x, y_spaces[j + 1], min_height]])

#     line_set = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(np.array(lines).reshape(-1, 3)),
#         lines=o3d.utility.Vector2iVector(np.arange(0, len(lines) * 2).reshape(-1, 2))
#     )

#     return line_set


# def get_cubes(point_cloud_np, min_height, max_height, voxel_size):
#     '''
#     Compute relative height of each point cloud coordinate and its positioning on 3d plane

#     :param point_cloud_np: numpy array of the point cloud
#     :param min_height: min height of the point cloud after processing
#     :param max_height: max height of the point cloud after processing
#     :param voxel_size: size of the cube used for visualizing

#     :return: list of colored cubes with coordinate assigned for positioning
#     '''
#     colormap = plt.get_cmap('jet')
#     cubes = []
#     for point in point_cloud_np:
#         if point[2] > 0:
#             height = point[2] - min_height
#             color = colormap((height) / (max_height - min_height))[:3]

#             cube = o3d.geometry.TriangleMesh.create_box(width=voxel_size,
#                                                         height=voxel_size,
#                                                         depth=height)
#             cube.translate([point[0] - voxel_size / 2, point[1] - voxel_size / 2, min_height])
#             cube.paint_uniform_color(color)
#             cubes.append(cube)

#     return cubes


# def get_hulls(point_cloud_np, min_height):
#     '''
#     Compute convex hull of the triangle mesh

#     :param point_cloud_np: numpy array of the point cloud
#     :param min_height: min height of the point cloud after processing

#     :return: convex hull of the triangle mesh
#     '''

#     xy_points = point_cloud_np[:, :2]
#     hull = ConvexHull(xy_points)
#     hull_indices = hull.vertices

#     hull_vertices = point_cloud_np[hull_indices, :]
#     hull_points = xy_points[hull.vertices]
#     hull_points_3d = np.hstack((hull_points, np.full((hull_points.shape[0], 1), min_height)))

#     # Delaunay triangulation of convex hull vertices
#     delaunay = Delaunay(hull_points)

#     # Create Open3D triangular meshes
#     hull_mesh = o3d.geometry.TriangleMesh()

#     # Add vertex
#     hull_mesh.vertices = o3d.utility.Vector3dVector(hull_points_3d)

#     # Add triangular faces
#     hull_mesh.triangles = o3d.utility.Vector3iVector(delaunay.simplices)

#     # Calculate the normal vector so that the mesh can render correctly
#     hull_mesh.compute_vertex_normals()

#     # Coloring with deep green filling
#     hull_mesh.paint_uniform_color([0.0, 0.5, 0.2])

#     return hull_mesh