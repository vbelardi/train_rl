import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Device configuration
device = "cpu"
print(f"Using device: {device}")

voxel_size = 0.3
model = PPO.load("ppo_drone_exploration_model", device=device)

def direction_to_goal_point(drone_position, direction_vector, voxel_grid_origin, voxel_grid_dims):
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

# Initialize the grid (gvg) as in your original example.
gvg = np.zeros((17, 17, 17), np.int32)


# Initial drone position.
current_position = np.array([4.5, 4.5, 4.5])



obs = {
    "observation": gvg.flatten(), 
    "drone_positions": (current_position / np.array([5.0, 5.0, 5.0])).astype(np.float32).reshape(3,)
}

# Predict the action.
actions, _ = model.predict(obs, deterministic=True)
actions = np.array(actions, dtype=np.float32).reshape(1, 3)
action = actions[0]
print("  Current position:", current_position)
print("  Action (direction):", action)
    
    # Compute the goal point.
goal_point = direction_to_goal_point(current_position, action, [0.0, 0.0, 0.0], [5.0, 5.0, 5.0])
print("  Goal point:", goal_point)
