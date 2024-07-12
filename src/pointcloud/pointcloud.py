from pointcloud.elevation_map.elevation_map_loader import *

class PointCloud:
    '''
    Class that contains the information about the point cloud
    '''
    def __init__(self, file_path, height_limit=3.5, grid_resolution=8, boundary = None):
        '''
        :param file_path: relative path of the splat file
        :param height_limit: height threshold to clip heights beyond the limit
        :param grid_resolution : resolution used to compute the elevation map
        '''
        self.grid_resolution = grid_resolution
        self.point_cloud = get_point_cloud(file_path)
        self.point_cloud_name = file_path.split("/")[-1].replace(".ply", "")
        self.elevation_map, self.offset_x, self.offset_y, self.min_height = get_elevation_map(self.point_cloud, height_limit=height_limit, grid_resolution=grid_resolution)

    def get_occupancy_map(self, occupancy_threshold = 1.75 * 0.1):
        return point_cloud_to_occupany_map(self.elevation_map, threshold = occupancy_threshold)
    
    def get_elevation_map(self):
        return self.elevation_map
    
    def get_elevation_map_info(self):
        return self.offset_x, self.offset_y, self.min_height
    
