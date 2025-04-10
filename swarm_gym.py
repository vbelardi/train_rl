import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from gymnasium import spaces

import time 


import open3d as o3d

import voxelgrid


'''
possible problems :
reward depending on completeness of the grid or not
following path just for a few steps or until goal
grid and drone positions are flatten in a single vector and has same weight while drone positions should be more important


change environments and obstacles for each simulation
add drones
'''

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DroneExplorationEnv(gym.Env):
    def __init__(self):
        super(DroneExplorationEnv, self).__init__()
        
        self.voxel_size = 0.3
        self.num_drones = 1
        
        # Chargement de la grille voxel avec la configuration
        self.global_vg = voxelgrid.create_voxelgrid_from_config("./src/multi_agent_pkgs/env_builder/config/env_small_config.yaml")
    
        self.voxel_space_size = self.global_vg.get_dim()
        self.origin = self.global_vg.get_origin()

        self.max_steps = 1000
        self.step_count = 0
        self.total_reward = 0
        

        self.observation = voxelgrid.VoxelGrid(
            self.origin, self.global_vg.get_dim(), 
            self.voxel_size, False)
        
        

        # Initialisation des positions des drones
        self.initial_position1 = np.array([0.5, 0.5, 0.5])
        #self.initial_position2 = np.array([7.0, 15.0, 4.0])
        #self.initial_position3 = np.array([9.0, 15.0, 4.0])
        self.drone_positions = np.array(self.initial_position1)
        #self.drone_positions = np.array([self.initial_position1, self.initial_position2, self.initial_position3])

        #self.render_open3d_init()
        
        # Définition des espaces d'observation et d'action
        
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0, high=2, shape=self.voxel_space_size, dtype=np.uint8),
            "drone_positions": spaces.Box(low=0, high=1, shape=(3*self.num_drones,), dtype="float32")
        })
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3*self.num_drones,), dtype="float32")



    def reset(self, seed=None, options=None):
        # Optionally, handle the seed if needed:
        if seed is not None:
            # You could use the seed to reset randomness in your environment.
            pass

        # Recreate your voxel grid with your desired configuration.
        self.global_vg = voxelgrid.create_voxelgrid_from_config(
            "./src/multi_agent_pkgs/env_builder/config/env_small_config.yaml"
        )
        self.total_reward = 0
        self.step_count = 0
        #self.drone_positions = np.array(self.initial_position1)
        grid_origin = np.array(self.origin)
        grid_dims = np.array(self.global_vg.get_real_dim())
        # You might also want to leave a small margin (e.g., 0.1) from the boundaries.
        margin = 0.2

        self.drone_positions = np.random.uniform(low=grid_origin + margin, 
                                                high=grid_origin + grid_dims - margin, 
                                                size=(3,))
        self.observation = voxelgrid.VoxelGrid(
            self.origin, self.global_vg.get_dim(), 
            self.voxel_size, False)
        #self.drone_positions = np.array([self.initial_position1, self.initial_position2, self.initial_position3])

        # Gymnasium expects reset() to return (observation, info)
        return self.get_observation(), {}


    def get_observation(self):
        """Retourne l'état actuel sous forme de dictionnaire."""
        obs = voxelgrid.get_data_np(self.observation)
        obs = np.where(obs == -1, 0, np.where(obs == 0, 1, 2))
        return {
            "observation": obs.astype(np.uint8),
            "drone_positions": (self.drone_positions/self.global_vg.get_real_dim()).astype(np.float32)
        }

    def direction_to_goal_point(self, drone_position, direction_vector, voxel_grid_origin, voxel_grid_dims):
        """
        Converts a normalized direction vector into the real intersection point at the boundary of the voxel grid.
        """
        norm = np.linalg.norm(direction_vector)
        if norm == 0:
            return drone_position  # No movement if the direction is zero.
        direction_unit = direction_vector / norm

        grid_min = np.array(voxel_grid_origin)
        grid_max = np.array(voxel_grid_origin) + np.array(voxel_grid_dims)

        travel_distances = []
        for i in range(3):
            if direction_unit[i] > 0:
                travel_distance = (grid_max[i] - drone_position[i]) / direction_unit[i]
            elif direction_unit[i] < 0:
                travel_distance = (grid_min[i] - drone_position[i]) / direction_unit[i]
            else:
                travel_distance = np.inf  # No movement along this axis.
            travel_distances.append(travel_distance)

        min_travel = min(travel_distances)
        goal_point = drone_position + direction_unit * min_travel
        # Clip to remain within a defined range.
        goal_point = np.clip(goal_point, [0.1, 0.1, 0.1], [4.9, 4.9, 4.9])
        return goal_point


    
    def step(self, actions):
        """Executes a simulation step using goal points computed from the continuous action vectors."""
        self.step_count += 1
        actions = np.array(actions, dtype=np.float32).reshape(self.num_drones, 3)
        drone_positions = np.array(self.drone_positions, dtype=np.float32).reshape(self.num_drones, 3)
        # Compute goal points for each drone based on its current position and the direction vector.
        goal_points = []
        for i, action in enumerate(actions):
            current_position = drone_positions[i]
            goal_point = self.direction_to_goal_point(current_position, action, self.origin, self.global_vg.get_real_dim())
            goal_points.append(goal_point)
        goal_points = np.array(goal_points)
        # Pass goal_points as a parameter to step_cpp instead of raw actions.
        unknown_bef = np.sum(voxelgrid.get_data_np(self.observation) == -1)
        observation, _, done, info = voxelgrid.step_cpp(drone_positions, goal_points, self.observation, self.global_vg, 5)
        self.drone_positions = np.array(observation["drone_positions"], dtype=np.float32).reshape(self.num_drones* 3,)
        self.observation = observation["observation"]
        unknown_after = np.sum(voxelgrid.get_data_np(self.observation) == -1)
        reward = (unknown_bef - unknown_after)*1.0
        if reward < 1.0:
            reward = -1.0
        elif reward < 5.0:
            reward = 0.0
        elif reward > 20.0:
            reward = 5.0
        else:
            reward = 1.0
        self.total_reward += reward
    
        
        obs = voxelgrid.get_data_np(observation["observation"])
        obs = np.where(obs == -1, 0, np.where(obs == 0, 1, 2))
        observation["observation"] = obs.astype(np.uint8)
        observation["drone_positions"] = (self.drone_positions/self.global_vg.get_real_dim()).astype(np.float32)


        if self.step_count > self.max_steps:
            reward = -10
            return observation, reward, done, True, info
        return observation, reward, done, False, info



    # --------------------- Rendering Functions ---------------------

    def render_open3d(self):
        points = []
        colors = []
        for x in range(self.voxel_space_size[0]):
            for y in range(self.voxel_space_size[1]):
                for z in range(self.voxel_space_size[2]):
                    val = self.observation.get_voxel_int([x, y, z])
                    if val == 0:
                        points.append((x, y, z))
                        colors.append([0.0, 0.0, 1.0])
                    if val == 100:
                        points.append((x, y, z))
                        colors.append([1.0, 0.0, 0.0])
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=self.voxel_size)

        # Create a blue sphere for each drone position
        drone_spheres = []
        for pos in self.drone_positions:            
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.voxel_size)
            # Translate the sphere to the drone's world position
            sphere.translate((pos-self.global_vg.get_origin())/0.3)
            sphere.paint_uniform_color([0.0, 1.0, 0.0])  # green color
            drone_spheres.append(sphere)
        
        

        # Render both the voxel grid and the drone spheres
        o3d.visualization.draw_geometries([voxel_grid]+drone_spheres)

    def render_open3d_actions(self, actions):
        points = []
        colors = []
        for x in range(self.voxel_space_size[0]):
            for y in range(self.voxel_space_size[1]):
                for z in range(self.voxel_space_size[2]):
                    val = self.observation.get_voxel_int([x, y, z])
                    if val == 0:
                        points.append((x, y, z))
                        colors.append([0.0, 0.0, 1.0])
                    if val == 100:
                        points.append((x, y, z))
                        colors.append([1.0, 0.0, 0.0])
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=self.voxel_size)

        # Create a blue sphere for each drone position
        drone_spheres = []
        for pos in self.drone_positions:            
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.voxel_size)
            # Translate the sphere to the drone's world position
            sphere.translate((pos-self.global_vg.get_origin())/0.3)
            sphere.paint_uniform_color([0.0, 1.0, 0.0])  # green color
            drone_spheres.append(sphere)
        
        goal_spheres = []
        for pos in actions:            
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.voxel_size)
            # Translate the sphere to the drone's world position
            sphere.translate((pos-self.global_vg.get_origin())/0.3)
            sphere.paint_uniform_color([1.0, 0.0, 1.0])  # Blue color
            goal_spheres.append(sphere)
        

        # Render both the voxel grid and the drone spheres
        o3d.visualization.draw_geometries([voxel_grid]+drone_spheres+goal_spheres)

    def render_open3d_init(self):
        points = []
        colors = []
        for x in range(self.voxel_space_size[0]):
            for y in range(self.voxel_space_size[1]):
                for z in range(self.voxel_space_size[2]):
                    val = self.global_vg.get_voxel_int([x, y, z])
                    if val == 100:
                        points.append((x, y, z))
                        colors.append([1.0, 0.0, 0.0])
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=self.voxel_size)

        # Create a blue sphere for each drone position
        drone_spheres = []
        for pos in self.drone_positions:            
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5*self.voxel_size)
            # Translate the sphere to the drone's world position
            sphere.translate(pos)
            sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Blue color
            drone_spheres.append(sphere)

        # Render both the voxel grid and the drone spheres
        o3d.visualization.draw_geometries([voxel_grid] + drone_spheres)


