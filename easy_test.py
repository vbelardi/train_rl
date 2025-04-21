import voxelgrid
import open3d as o3d

global_vg = voxelgrid.create_voxelgrid_from_config("./src/multi_agent_pkgs/env_builder/config/env_default_config.yaml")

def render_open3d_init(global_vg):
    points = []
    colors = []
    for x in range(134):
        for y in range(134):
            for z in range(67):
                val = global_vg.get_voxel_int([x, y, z])
                if val == 100:
                    points.append((x, y, z))
                    colors.append([1.0, 0.0, 0.0])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=0.3)


    # Render both the voxel grid and the drone spheres
    o3d.visualization.draw_geometries([voxel_grid])

render_open3d_init(global_vg)

