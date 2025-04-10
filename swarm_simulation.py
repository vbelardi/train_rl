import gymnasium as gym
import torch
import numpy as np
import time
from swarm_gym import DroneExplorationEnv
import voxelgrid 
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gymnasium as gym
import torch
import numpy as np
import time
from swarm_gym import DroneExplorationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        """
        Custom extractor for an observation dictionary with:
          - "observation": a 3D voxel grid of shape (D, H, W).
          - "drone_positions": a low-dimensional vector.
        
        The pipeline:
          1. Convert voxel grid into 3-channel one-hot encoding.
          2. Apply a 3D CNN on the voxel grid.
          3. Concatenate with drone position vector.
          4. Fuse with an MLP.
        """
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)

        voxel_shape = observation_space.spaces["observation"].shape[:3]  # (D, H, W)

        # 3-channel CNN
        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1),  # Input channels = 3
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Flatten()
        )

        # Compute CNN output dimension
        dummy_voxel = th.zeros(1, 3, *voxel_shape)  # 3 channels now
        cnn_output_dim = self.cnn3d(dummy_voxel).shape[1]

        fusion_input_dim = cnn_output_dim + 3  # +3 for drone position vector
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        voxel = observations["observation"]  # (batch, D, H, W)

        # One-hot encoding to 3 channels
        channel_unknown = (voxel == 0).unsqueeze(1).float()  # (batch, 1, D, H, W)
        channel_free    = (voxel == 1).unsqueeze(1).float()
        channel_obstacle= (voxel == 2).unsqueeze(1).float()
        voxel_input = th.cat([channel_unknown, channel_free, channel_obstacle], dim=1)

        cnn_features = self.cnn3d(voxel_input)

        # Concatenate with drone position vector
        fused = th.cat([cnn_features, observations["drone_positions"]], dim=1)
        features = self.fusion_mlp(fused)
        return features


policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=256)
)

if __name__ == "__main__":
    env = make_vec_env(DroneExplorationEnv, n_envs=10, vec_env_cls=SubprocVecEnv)
    check_env(DroneExplorationEnv(), warn=True)

    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    checkpoint_callback = CheckpointCallback(save_freq=1_000_000, save_path="./models/", name_prefix="ppo_checkpoint")
    # ,  callback=checkpoint_callback
    model.learn(total_timesteps=1_000_000)
    model.save("ppo_drone_exploration_model")
