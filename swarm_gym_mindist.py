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



# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DroneExplorationEnv(gym.Env):
    def __init__(self):
        super(DroneExplorationEnv, self).__init__()
        
        self.voxel_size = 0.3
        self.num_drones = 3
        
        # Chargement de la grille voxel avec la configuration
        self.global_vg = voxelgrid.create_voxelgrid_from_config("./src/multi_agent_pkgs/env_builder/config/env_RL20_config.yaml")
    
        self.voxel_space_size = self.global_vg.get_dim()
        self.origin = self.global_vg.get_origin()

        self.max_steps = 180
        self.step_count = 0
        self.total_reward = 0
        self.old_completeness = -1.0
        

        self.observation = voxelgrid.VoxelGrid(
            self.origin, self.global_vg.get_dim(), 
            self.voxel_size, False)
        
        self.count_map = voxelgrid.VoxelGrid(
            self.origin, self.global_vg.get_dim(), 
            self.voxel_size, True
        )


        self.grid_origin = np.array(self.origin)
        self.grid_dims = np.array(self.global_vg.get_real_dim())
        self.margin = 0.2

        self.episode_counter = 0

        self.drone_positions = self.generate_free_positions()

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0, high=2, shape=self.voxel_space_size, dtype=np.uint8),
            "drone_positions": spaces.Box(low=0, high=1, shape=(3*self.num_drones,), dtype="float32")
        })
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3*self.num_drones,), dtype="float32")



    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.global_vg = voxelgrid.create_voxelgrid_from_config(
            "./src/multi_agent_pkgs/env_builder/config/env_RL20_config.yaml"
        )
        self.total_reward = 0
        self.step_count = 0
        self.episode_counter += 1

        self.drone_positions = self.generate_free_positions(use_reset_sampling=True)

        # Réinitialiser la carte de comptage (visite des voxels)
        self.count_map = voxelgrid.VoxelGrid(
            self.origin, self.global_vg.get_dim(), 
            self.voxel_size, True
        )

        self.observation = voxelgrid.VoxelGrid(
            self.origin, self.global_vg.get_dim(),
            self.voxel_size, False
        )


        return self.get_observation(), {}

    
    def generate_free_positions(self, use_reset_sampling=False):
        positions = []
        # Fallback : uniforme aléatoire
        for _ in range(self.num_drones):
            while True:
                pos = np.random.uniform(
                    low=self.grid_origin + self.margin,
                    high=self.grid_origin + self.grid_dims - self.margin,
                    size=(3,)
                )
                voxel_index = np.floor((pos - self.global_vg.get_origin()) / self.voxel_size).astype(int)
                if self.global_vg.get_voxel_int(voxel_index.tolist()) == 0:
                    positions.append(pos)
                    break

        return np.concatenate(positions)



    def get_observation(self):
        """Retourne l'état actuel sous forme de dictionnaire."""
        obs = voxelgrid.get_data_np(self.observation)
        obs = np.where(obs == -1, 0, np.where(obs == 0, 1, 2))
        norm_drone_pos = self.drone_positions.copy()
        for i in range(self.num_drones):
            norm_drone_pos[i*3:(i+1)*3] = (self.drone_positions[i*3:(i+1)*3] - self.origin) / self.global_vg.get_real_dim()
        return {
            "observation": obs.astype(np.uint8),
            "drone_positions": (norm_drone_pos).astype(np.float32)
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
        goal_point = np.clip(goal_point, self.grid_origin + self.margin, self.grid_origin + self.grid_dims - self.margin)
        return goal_point


    
    def step(self, actions):
        """Executes a simulation step using goal points computed from the continuous action vectors."""
        #time_start = time.time()
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
        #unknown_bef = np.sum(voxelgrid.get_data_np(self.observation) == -1)
        old_counts = voxelgrid.get_data_np(self.count_map)
        #start_time = time.time()
        observation, self.count_map, done, info = voxelgrid.step_cpp(drone_positions, goal_points, self.observation, self.count_map, self.global_vg, 5)
        #end_time = time.time()
        #print("Time taken for step_cpp: ", end_time - start_time)
        self.drone_positions = np.array(observation["drone_positions"], dtype=np.float32).reshape(self.num_drones* 3,)
        self.observation = observation["observation"]
        unknown_after = np.sum(voxelgrid.get_data_np(self.observation) == -1)

        completeness = 1 - (unknown_after / (self.voxel_space_size[0] * self.voxel_space_size[1] * self.voxel_space_size[2]))
    
        # Progressive reward scaling - higher rewards for later exploration
        
        if completeness > 0.8:
            exploration_scale = 2.0  
        elif completeness > 0.85:
            exploration_scale = 3.0
        elif completeness > 0.9:
            exploration_scale = 5.0 
        elif completeness > 0.93:
            exploration_scale = 10.0
        elif completeness > 0.95:
            exploration_scale = 20.0
        else:
            exploration_scale = 1.0        
        reward = 0.0
        counts = voxelgrid.get_data_np(self.count_map)
        difference = counts - old_counts
        counts = counts/(self.max_steps*5)
        diff_mask = difference > 0
        
        # Base step penalty
        
        if self.num_drones > 1:
            positions = self.drone_positions.reshape(self.num_drones, 3)
            min_distance = float('inf')
            for i in range(self.num_drones):
                for j in range(i+1, self.num_drones):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    min_distance = min(min_distance, dist)
            
            # Encourage minimum separation of 6 voxels
            optimal_distance = 6 * self.voxel_size
            min_distance_bonus = 0.1 * min(1.0, min_distance / optimal_distance)
        
        # Scale rewards for newly discovered voxels
        
        voxel_rewards = (1.0 - counts)[diff_mask] * exploration_scale
        reward = voxel_rewards.sum() / (4**3 / 0.3**3)
        reward -= 0.02
        reward += min_distance_bonus
        if completeness > 0.85 and completeness - self.old_completeness < 0.001:
            reward -= 0.1
        if completeness > 0.9 and self.old_completeness <= 0.9:
            reward += 5.0  # Bonus for reaching 90%
        if completeness > 0.93 and self.old_completeness <= 0.93:
            reward += 10.0  # Bonus for reaching 93%
        self.total_reward += reward
        self.old_completeness = completeness

        
    
        
        obs = voxelgrid.get_data_np(observation["observation"])
        obs = np.where(obs == -1, 0, np.where(obs == 0, 1, 2))
        observation["observation"] = obs.astype(np.uint8)
        # Normalize drone positions to the voxel grid dimensions
        norm_drone_pos = self.drone_positions.copy()
        for i in range(self.num_drones):
            norm_drone_pos[i*3:(i+1)*3] = (self.drone_positions[i*3:(i+1)*3] - self.origin) / self.global_vg.get_real_dim()
        observation["drone_positions"] = (norm_drone_pos).astype(np.float32)

        #end_time = time.time()
        #print("Time taken for step: ", end_time - time_start)



        if completeness > 0.95:
            reward += 100.0
            reward += 500.0 * (1 - self.step_count / self.max_steps)
            print("Episode completed with completeness: ", completeness, " at step: ", self.step_count)
            done = True

        if self.step_count >= self.max_steps:
            print("Completessness: ", completeness)
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


