import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize, VecFrameStack
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from sb3_contrib import RecurrentPPO
from gymnasium import spaces
import numpy as np
import voxelgrid

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AdvancedDroneExplorationEnv(gym.Env):
    def __init__(self):
        super(AdvancedDroneExplorationEnv, self).__init__()
        
        self.voxel_size = 0.3
        self.num_drones = 3
        
        # Load voxel grid with configuration
        self.global_vg = voxelgrid.create_voxelgrid_from_config("./src/multi_agent_pkgs/env_builder/config/env_RL20_config.yaml")
    
        self.voxel_space_size = self.global_vg.get_dim()
        self.origin = self.global_vg.get_origin()

        self.max_steps = 180
        self.step_count = 0
        self.total_reward = 0
        
        # Main observation grid
        self.observation = voxelgrid.VoxelGrid(
            self.origin, self.global_vg.get_dim(), 
            self.voxel_size, False)
        
        # Count map to track visits
        self.count_map = voxelgrid.VoxelGrid(
            self.origin, self.global_vg.get_dim(), 
            self.voxel_size, True
        )
        
        # Frontier grid to highlight exploration boundaries
        self.frontier_grid = voxelgrid.VoxelGrid(
            self.origin, self.global_vg.get_dim(), 
            self.voxel_size, True
        )

        # Initialize drone positions
        self.grid_origin = np.array(self.origin)
        self.grid_dims = np.array(self.global_vg.get_real_dim())
        self.margin = 0.2

        self.episode_counter = 0
        self.last_completeness = 0.0
        self.stagnation_counter = 0

        # Initialize drone positions and previous positions for tracking
        self.drone_positions = self.generate_free_positions()
        self.prev_drone_positions = np.copy(self.drone_positions)
        
        # Define observation and action spaces
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0, high=2, shape=self.voxel_space_size, dtype=np.uint8),
            "drone_positions": spaces.Box(low=0, high=1, shape=(3*self.num_drones,), dtype="float32"),
            "frontier_map": spaces.Box(low=0, high=1, shape=self.voxel_space_size, dtype=np.uint8)
        })
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3*self.num_drones,), dtype="float32")
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.step_count = 0
        self.total_reward = 0
        self.episode_counter += 1
        self.last_completeness = 0.0
        self.stagnation_counter = 0
        
        # Reset global voxel grid (reload from config)
        self.global_vg = voxelgrid.create_voxelgrid_from_config(
            "./src/multi_agent_pkgs/env_builder/config/env_RL20_config.yaml"
        )
        
        # Reset observation grid and count map
        self.observation = voxelgrid.VoxelGrid(
            self.origin, self.global_vg.get_dim(), 
            self.voxel_size, False)
        
        self.count_map = voxelgrid.VoxelGrid(
            self.origin, self.global_vg.get_dim(), 
            self.voxel_size, True)
            
        self.frontier_grid = voxelgrid.VoxelGrid(
            self.origin, self.global_vg.get_dim(), 
            self.voxel_size, True)
        
        # Generate new drone positions
        self.drone_positions = self.generate_free_positions()
        self.prev_drone_positions = np.copy(self.drone_positions)
        
        # Initial raycasting is now handled by step_cpp
        # Update frontier map
        self.update_frontier_map()
        
        # Get observation
        observation = self.get_observation()
        info = {}
        
        return observation, info
    
    def generate_free_positions(self, use_reset_sampling=False):
        """Generate free positions for drones with improved distribution"""
        positions = []
        
        # Define minimum distance between drones to ensure good separation
        min_distance = 3.0
        
        # Try to place drones with a minimum distance between them
        for _ in range(self.num_drones):
            valid = False
            attempts = 0
            max_attempts = 100
            
            while not valid and attempts < max_attempts:
                # Generate a random position within the grid boundaries with margin
                pos = np.random.uniform(
                    low=self.grid_origin + self.margin,
                    high=self.grid_origin + self.grid_dims - self.margin,
                    size=(3,)
                )
                
                # Check if position is free
                voxel_index = np.floor((pos - self.global_vg.get_origin()) / self.voxel_size).astype(int)
                if self.global_vg.get_voxel_int(voxel_index.tolist()) == 0:
                    # Check distance to other drones
                    valid = True
                    for existing_pos in positions:
                        distance = np.linalg.norm(pos - existing_pos)
                        if distance < min_distance:
                            valid = False
                            break
                
                attempts += 1
            
            if valid:
                positions.append(pos)
            else:
                # If we can't find a valid position with minimum distance after max attempts,
                # just place it randomly in a free position
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
    
    def update_frontier_map(self):
        """Update the frontier map to highlight borders between known and unknown space"""
        obs_data = voxelgrid.get_data_np(self.observation)
        self.frontier_array = np.zeros(self.voxel_space_size)
        
        # Mark cells that are free and adjacent to unknown space as frontier cells
        D, H, W = self.voxel_space_size
        for x in range(1, D-1):
            for y in range(1, H-1):
                for z in range(1, W-1):
                    # If this voxel is free (value 0 in obs_data)
                    if obs_data[x, y, z] == 0:
                        # Check 6-connected neighbors
                        neighbors = [
                            (x+1, y, z), (x-1, y, z),
                            (x, y+1, z), (x, y-1, z),
                            (x, y, z+1), (x, y, z-1)
                        ]
                        
                        for nx, ny, nz in neighbors:
                            # If neighbor is unknown (value -1 in obs_data), mark this as frontier
                            if 0 <= nx < D and 0 <= ny < H and 0 <= nz < W:
                                if obs_data[nx, ny, nz] == -1:  # Unknown space
                                    self.frontier_array[x, y, z] = 1
                                    break
    
    def get_observation(self):
        """Create the observation dictionary with all components"""
        # Get voxel data and convert -1,0,100 to 0,1,2 as in original code
        obs_data = voxelgrid.get_data_np(self.observation)
        obs_data = np.where(obs_data == -1, 0, np.where(obs_data == 0, 1, 2))
        
        # Normalize drone positions to [0,1] range
        normalized_positions = np.zeros(3 * self.num_drones, dtype=np.float32)
        for i in range(self.num_drones):
            normalized_positions[i*3:(i+1)*3] = (
                self.drone_positions[i*3:(i+1)*3] - self.origin) / self.global_vg.get_real_dim()
        
        return {
            "observation": obs_data.astype(np.uint8),
            "drone_positions": normalized_positions,
            "frontier_map": self.frontier_array.astype(np.uint8)
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
        goal_point = np.clip(goal_point, grid_min + self.margin, grid_max - self.margin)
        return goal_point
    
    def step(self, actions):
        """Executes a simulation step using the step_cpp function"""
        self.step_count += 1
        self.prev_drone_positions = np.copy(self.drone_positions)
        
        # Reshape actions and drone positions
        actions = np.array(actions, dtype=np.float32).reshape(self.num_drones, 3)
        drone_positions = np.array(self.drone_positions, dtype=np.float32).reshape(self.num_drones, 3)
        
        # Compute goal points for each drone based on its current position and the direction vector
        goal_points = []
        for i, action in enumerate(actions):
            current_position = drone_positions[i]
            goal_point = self.direction_to_goal_point(
                current_position, action, self.origin, self.global_vg.get_real_dim()
            )
            goal_points.append(goal_point)
        
        goal_points = np.array(goal_points)
        
        # Store current count data for reward calculation
        old_counts = voxelgrid.get_data_np(self.count_map)
        
        # Use step_cpp to perform raycasting and get updated observation
        observation, self.count_map, done, info = voxelgrid.step_cpp(
            drone_positions, goal_points, self.observation, self.count_map, 
            self.global_vg, 5  # 5 rays per degree
        )
        
        # Update drone positions
        self.drone_positions = np.array(observation["drone_positions"], dtype=np.float32).reshape(self.num_drones * 3,)
        
        # Update observation
        self.observation = observation["observation"]
        
        # Count unknown voxels for completeness calculation
        obs_data = voxelgrid.get_data_np(self.observation)
        unknown_voxels = np.sum(obs_data == -1)
        total_voxels = self.voxel_space_size[0] * self.voxel_space_size[1] * self.voxel_space_size[2]
        completeness = 1 - (unknown_voxels / total_voxels)
        
        # Update frontier map
        self.update_frontier_map()
        
        # Calculate rewards
        # Progressive reward scaling - higher rewards for later exploration
        if completeness > 0.95:
            exploration_scale = 32.0
        elif completeness > 0.93:
            exploration_scale = 16.0
        elif completeness > 0.90:
            exploration_scale = 8.0
        elif completeness > 0.85:
            exploration_scale = 4.0
        elif completeness > 0.80:
            exploration_scale = 2.0
        else:
            exploration_scale = 1.0
            
        # Basic reward calculation
        reward = 0.0
        
        # Reward for new discoveries
        counts = voxelgrid.get_data_np(self.count_map)
        difference = counts - old_counts
        counts = counts / (self.max_steps * 5)  # Normalize by max possible counts
        diff_mask = difference > 0  # Mask for newly discovered voxels
        
        # Set base step penalty
        reward -= 0.02
        
        # Drone distance bonus
        min_distance_bonus = 0.0
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
            reward += min_distance_bonus
        
        # Frontier seeking reward
        frontier_reward = 0.0
        frontier_data = self.frontier_array
        for i in range(self.num_drones):
            pos = self.drone_positions[i*3:(i+1)*3]
            voxel_pos = ((pos - self.grid_origin) / self.voxel_size).astype(int)
            
            # Check if position is within bounds
            D, H, W = self.voxel_space_size
            if (0 <= voxel_pos[0] < D and 0 <= voxel_pos[1] < H and 0 <= voxel_pos[2] < W):
                # Search in a small radius around the drone for frontier cells
                radius = 3
                x_min, x_max = max(0, voxel_pos[0]-radius), min(D-1, voxel_pos[0]+radius)
                y_min, y_max = max(0, voxel_pos[1]-radius), min(H-1, voxel_pos[1]+radius)
                z_min, z_max = max(0, voxel_pos[2]-radius), min(W-1, voxel_pos[2]+radius)
                
                # Count frontier cells in vicinity
                frontier_count = np.sum(frontier_data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1])
                frontier_reward += min(frontier_count * 0.01, 0.5)

        reward += 0.1*frontier_reward
        
        # Reward for newly discovered voxels with progressive scaling
        if np.any(diff_mask):
            voxel_rewards = (1.0 - counts)[diff_mask] * exploration_scale
            discovery_reward = voxel_rewards.sum() / (4**3 / 0.3**3)
            reward += discovery_reward
        
        # Check for stagnation
        if np.sum(difference) == 0:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            
        # Add stagnation penalty
        if self.stagnation_counter > 5:
            stagnation_penalty = -0.1 * (self.stagnation_counter - 5) / 10
            reward += max(stagnation_penalty, -2.0)
        
        # Add completion bonuses
        if completeness > 0.95 and self.last_completeness <= 0.95:
            reward += 20.0
            reward += 100.0 * (1 - self.step_count / self.max_steps)  # Time bonus
            
        # Update last completeness
        self.last_completeness = completeness
        
        # Accumulate total reward
        self.total_reward += reward
        
        # Format observation
        obs = voxelgrid.get_data_np(self.observation)
        obs = np.where(obs == -1, 0, np.where(obs == 0, 1, 2))
        
        # Create observation dictionary
        observation_dict = {
            "observation": obs.astype(np.uint8),
            "drone_positions": np.array([(self.drone_positions[i*3:(i+1)*3] - self.origin) / self.global_vg.get_real_dim() 
                                       for i in range(self.num_drones)], dtype=np.float32).flatten(),
            "frontier_map": self.frontier_array.astype(np.uint8)
        }
        
        # Update info dictionary
        info["completeness"] = completeness
        info["total_reward"] = self.total_reward
        info["step_count"] = self.step_count
        
        # Check termination conditions
        truncated = False
        if self.step_count >= self.max_steps:
            print(f"Completeness: {completeness}")
            truncated = True
        
        return observation_dict, reward, done, truncated, info

# Custom feature extractor for the advanced environment
class AdvancedFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        D, H, W = observation_space.spaces["observation"].shape
        drone_shape = observation_space.spaces["drone_positions"].shape
        
        # Residual block for better spatial understanding
        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super(ResidualBlock, self).__init__()
                self.conv1 = nn.Conv3d(channels, channels, 3, 1, 1)
                self.bn1 = nn.BatchNorm3d(channels)
                self.conv2 = nn.Conv3d(channels, channels, 3, 1, 1)
                self.bn2 = nn.BatchNorm3d(channels)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                residual = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += residual  # Skip connection
                return self.relu(out)
        
        # 3D CNN for voxel processing with residual connections
        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 32, (5,5,3), (2,2,1), (1,1,1)), nn.ReLU(),
            nn.Conv3d(32, 32, (5,5,3), (2,2,1), (1,1,1)), nn.ReLU(),
            ResidualBlock(32),
            nn.Conv3d(32, 64, 3, 2, 1), nn.ReLU(),
            ResidualBlock(64),
            nn.Conv3d(64, 128, 3, 1, 1), nn.ReLU(),
            ResidualBlock(128),
            nn.Flatten()
        )
        
        # Additional CNN for processing frontier map
        self.frontier_cnn = nn.Sequential(
            nn.Conv3d(1, 16, (5,5,3), (2,2,1), (1,1,1)), nn.ReLU(),
            nn.Conv3d(16, 32, (5,5,3), (2,2,1), (1,1,1)), nn.ReLU(),
            nn.Conv3d(32, 32, 3, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        
        # Position MLP
        self.pos_mlp = nn.Sequential(
            nn.Linear(drone_shape[0], 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        )
        
        # Calculate flattened dimensions for CNN outputs
        with torch.no_grad():
            d = D//4; h = H//4; w = W//4
            dummy_voxel = torch.zeros(1, 3, d, h, w)
            dummy_frontier = torch.zeros(1, 1, d, h, w)
            voxel_flat = self.cnn3d(dummy_voxel).shape[1]
            frontier_flat = self.frontier_cnn(dummy_frontier).shape[1]
        
        # Fusion network
        self.fuse = nn.Sequential(
            nn.Linear(voxel_flat + frontier_flat + 128, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, features_dim), nn.ReLU()
        )
        
        self._features_dim = features_dim

    def forward(self, obs):
        # Process voxel grid
        v = obs["observation"].long()
        u = (v==0).unsqueeze(1).float()  # Free space
        f = (v==1).unsqueeze(1).float()  # Occupied space
        o = (v==2).unsqueeze(1).float()  # Unknown space
        x = torch.cat([u, f, o], dim=1)
        
        # Process frontier map
        frontier = obs["frontier_map"].unsqueeze(1).float()
        
        # Downsample to match CNN input size
        _, D, H, W = obs["observation"].shape
        x = F.adaptive_avg_pool3d(x, output_size=(D//4, H//4, W//4))
        frontier = F.adaptive_avg_pool3d(frontier, output_size=(D//4, H//4, W//4))
        
        # Forward passes
        c = self.cnn3d(x)
        fr = self.frontier_cnn(frontier)
        p = self.pos_mlp(obs["drone_positions"])
        
        # Fusion
        return self.fuse(torch.cat([c, fr, p], 1))

class LearningRateScheduler(BaseCallback):
    def __init__(self, initial_lr=3e-4, min_lr=5e-6, decay_factor=0.5, 
                 decay_steps=500_000, verbose=0):
        super(LearningRateScheduler, self).__init__(verbose)
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        
    def _on_step(self):
        # Calculate new learning rate
        progress = self.num_timesteps / self.decay_steps
        new_lr = max(self.min_lr, self.initial_lr * (self.decay_factor ** progress))
        
        # Update learning rate
        self.model.learning_rate = new_lr
        
        # Log if verbose
        if self.verbose > 0 and self.n_calls % 10000 == 0:
            print(f"Timestep: {self.num_timesteps}, Learning rate: {new_lr:.6f}")
            
        return True

if __name__ == "__main__":
    venv = make_vec_env(AdvancedDroneExplorationEnv, n_envs=20, vec_env_cls=SubprocVecEnv)

    policy_kwargs = dict(
        features_extractor_class=AdvancedFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model = RecurrentPPO(
        "MultiInputLstmPolicy", venv,
        device=device,
        policy_kwargs=policy_kwargs,
        n_steps=180, n_epochs=10,
        learning_rate=3e-4, gamma=0.98,
        gae_lambda=0.95, ent_coef=5e-3,
        clip_range=0.2, verbose=1
    )
    cb = CheckpointCallback(save_freq=50_000, save_path="./advanced/", name_prefix="advanced_check")
    lr_scheduler = LearningRateScheduler(initial_lr=3e-4, min_lr=5e-6, decay_factor=0.75, decay_steps=25_000)

    callbacks = [cb, lr_scheduler]
    model.learn(total_timesteps=10_000_000, callback=callbacks)
    model.save("advanced_drone_exploration_model")