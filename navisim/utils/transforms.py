"""Coordinate transformation utilities."""

import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_matrix(pose_6dof: list) -> np.ndarray:
    """
    Convert 6DOF pose to 4x4 transformation matrix.
    
    Args:
        pose_6dof: [x, y, z, roll, pitch, yaw] pose
        
    Returns:
        4x4 transformation matrix
    """
    x, y, z, roll, pitch, yaw = pose_6dof
    
    # Create rotation matrix from Euler angles
    rotation = Rotation.from_euler('xyz', [roll, pitch, yaw])
    
    # Create 4x4 transformation matrix
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = [x, y, z]
    
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> list:
    """
    Convert 4x4 transformation matrix to 6DOF pose.
    
    Args:
        matrix: 4x4 transformation matrix
        
    Returns:
        [x, y, z, roll, pitch, yaw] pose
    """
    # Extract position
    position = matrix[:3, 3]
    
    # Extract rotation and convert to Euler angles
    rotation = Rotation.from_matrix(matrix[:3, :3])
    euler = rotation.as_euler('xyz')
    
    return [position[0], position[1], position[2], euler[0], euler[1], euler[2]]


def compose_poses(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
    """
    Compose two poses (pose1 * pose2).
    
    Args:
        pose1: First 4x4 transformation matrix
        pose2: Second 4x4 transformation matrix
        
    Returns:
        Composed 4x4 transformation matrix
    """
    return pose1 @ pose2


def inverse_pose(pose: np.ndarray) -> np.ndarray:
    """
    Compute inverse of a pose.
    
    Args:
        pose: 4x4 transformation matrix
        
    Returns:
        Inverse 4x4 transformation matrix
    """
    inverse = np.eye(4, dtype=np.float64)
    
    # Inverse rotation (transpose)
    R_inv = pose[:3, :3].T
    inverse[:3, :3] = R_inv
    
    # Inverse translation
    inverse[:3, 3] = -R_inv @ pose[:3, 3]
    
    return inverse