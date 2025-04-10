import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import voxelgrid

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
global_vg = voxelgrid.create_voxelgrid_from_config("./src/multi_agent_pkgs/env_builder/config/env_small_config.yaml")
glob = voxelgrid.get_data_np(global_vg)


# Initial drone position.
current_position = np.array([0.5, 0.5, 0.5])

# Number of simulation steps.
num_steps = 500

for step in range(num_steps):
    # Prepare the observation.
    # Note: The drone_positions are normalized by dividing by the grid dimensions (assumed here as [5.0,5.0,5.0]).
    obs = {
        "observation": gvg.astype(np.uint8), 
        "drone_positions": (current_position / np.array([5.0, 5.0, 5.0])).astype(np.float32).reshape(3,)
    }
    
    # Predict the action.
    actions, _ = model.predict(obs, deterministic=True)
    actions = np.array(actions, dtype=np.float32).reshape(1, 3)
    action = actions[0]
    print(f"Step {step+1}")
    print("  Current position:", current_position)
    print("  Action (direction):", action)
    
    # Compute the goal point.
    goal_point = direction_to_goal_point(current_position, action, [0.0, 0.0, 0.0], [5.0, 5.0, 5.0])
    print("  Goal point:", goal_point)
    
    # Convert current and goal points into voxel indices.
    g_p = (goal_point / voxel_size).astype(np.int32)
    curr_voxel = (current_position / voxel_size).astype(np.int32)
    
    # Mark the current and goal positions on the grid.

    
    # Compute intermediate points along the straight-line path.
    num_points = 50
    line_points = np.linspace(current_position, goal_point, num_points)
    radius = 3
    index = 0
    for pt in line_points:
        index += 1
        if index >= 10:
            break
        
        current_position = pt
        voxel_coord = (pt / voxel_size).astype(int)
        for i in range(voxel_coord[0] - radius, voxel_coord[0] + radius + 1):
            for j in range(voxel_coord[1] - radius, voxel_coord[1] + radius + 1):
                for k in range(voxel_coord[2] - radius, voxel_coord[2] + radius + 1):
                    if 0 <= i < gvg.shape[0] and 0 <= j < gvg.shape[1] and 0 <= k < gvg.shape[2]:
                        if np.linalg.norm(np.array([i, j, k]) - voxel_coord) <= radius:
                            if glob[i, j, k] == 100:
                                gvg[i, j, k] = 1.0
                            else:
                                gvg[i, j, k] = 0.5
    
    # Update the current position for the next step.


goal_point = direction_to_goal_point(current_position, action, [0.0, 0.0, 0.0], [5.0, 5.0, 5.0])
g_p = (goal_point / voxel_size).astype(np.int32)
gvg[g_p[0], g_p[1], g_p[2]] = 8.0  # Blue: goal position.

# --------------------- Visualization ---------------------
color_map = {
    0.5: "gray",   # Discovered space.
    1.0: "green",  # Obstacle.
    5.0: "red",    # Current (or past) position.
    8.0: "blue",   # Goal position.
    9.0: "green"   # Discovered path.
}

x, y, z, colors = [], [], [], []
for i in range(gvg.shape[0]):
    for j in range(gvg.shape[1]):
        for k in range(gvg.shape[2]):
            value = gvg[i, j, k]
            if value in color_map:
                x.append(i)
                y.append(j)
                z.append(k)
                colors.append(color_map[value])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=colors, s=50, edgecolors="k")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Visualization of Voxel Grid with Discovered Path")
ax.set_xlim(0, gvg.shape[0])
ax.set_ylim(0, gvg.shape[1])
ax.set_zlim(0, gvg.shape[2])
plt.show()
