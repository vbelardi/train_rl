import numpy as np
import pandas as pd
import open3d as o3d

def load_obstacle_map(filename, voxel_size, space_size):
    voxel_space_size = np.ceil(space_size / voxel_size).astype(int)
    states = np.zeros(voxel_space_size)
    df = pd.read_csv(filename, header=None)
    for _, row in df.iterrows():
        x, y, z = np.ceil((np.array(row)-[0,0,-6]) / voxel_size).astype(int)
        if 0 <= x < voxel_space_size[0] and 0 <= y < voxel_space_size[1] and 0 <= z < voxel_space_size[2]:
            states[x, y, z] = -1
    return states, voxel_space_size

def render_open3d(states, voxel_size, voxel_space_size):
    points = []
    colors = []
    for x in range(voxel_space_size[0]):
        for y in range(voxel_space_size[1]):
            for z in range(voxel_space_size[2]):
                if states[x, y, z] == -1:
                    points.append([x * voxel_size, y * voxel_size, z * voxel_size])
                    colors.append([1.0, 0.0, 0.0])
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxel_size)
    o3d.visualization.draw_geometries([voxel_grid])

# Parameters
voxel_size = 0.3
space_size = np.array([100, 30, 15])  # Define space size

# Load map and render
states, voxel_space_size = load_obstacle_map("map.csv", voxel_size, space_size)
render_open3d(states, voxel_size, voxel_space_size)