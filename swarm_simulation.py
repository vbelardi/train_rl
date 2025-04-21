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
import torch.nn.functional as F
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def downsample_voxel_grid_onehot(voxel_input, output_size):
    """
    Downsamples a one-hot encoded voxel grid (B, 3, D, H, W) using nearest neighbor,
    preserving one-hot exclusivity.
    """
    # Convert one-hot to label
    labels = th.argmax(voxel_input, dim=1)  # (B, D, H, W)

    # Downsample using nearest neighbor
    labels = labels.unsqueeze(1).float()
    labels_downsampled = th.nn.functional.interpolate(labels, size=output_size, mode='nearest')
    labels_downsampled = labels_downsampled.squeeze(1).long()

    # Convert back to one-hot
    voxel_downsampled = th.nn.functional.one_hot(labels_downsampled, num_classes=3)
    voxel_downsampled = voxel_downsampled.permute(0, 4, 1, 2, 3).float()

    return voxel_downsampled
    # -----------------------------------------------------------------------------


def downsample_voxel_grid_priority(voxel_input, output_size):
    """
    Downsamples a one-hot encoded voxel grid (B, 3, D, H, W) with priority: obstacle > unknown > free.
    Uses adaptive max-pooling per channel to preserve priority blocks.
    
    Args:
      voxel_input: Tensor of shape (B,3,D,H,W), one-hot encoding of {unknown,free,obstacle}.
      output_size: tuple (d2, h2, w2) target grid size.
    Returns:
      voxel_down: Tensor of shape (B,3,d2,h2,w2), one-hot with priority applied.
    """
    # 1) Extract per-class masks
    unk_mask  = voxel_input[:, 0:1]  # (B,1,D,H,W)
    free_mask = voxel_input[:, 1:2]
    obs_mask  = voxel_input[:, 2:3]
    
    # 2) Adaptive max-pool each mask to output_size
    unk_ds  = F.adaptive_max_pool3d(unk_mask,  output_size).squeeze(1)  # (B, D2,H2,W2)
    free_ds = F.adaptive_max_pool3d(free_mask, output_size).squeeze(1)
    obs_ds  = F.adaptive_max_pool3d(obs_mask,  output_size).squeeze(1)
    
    # 3) Build label grid with priority
    # default = free (1)
    labels = torch.ones_like(obs_ds, dtype=torch.long)
    # override unknown (0) where unk_ds>0
    labels[unk_ds > 0] = 0
    # override obstacle (2) where obs_ds>0  (highest priority)
    labels[obs_ds > 0] = 2
    
    # 4) Convert back to one-hot
    voxel_down = F.one_hot(labels, num_classes=3)        # (B, D2,H2,W2, 3)
    voxel_down = voxel_down.permute(0, 4, 1, 2, 3).float()  # (B,3,D2,H2,W2)
    return voxel_down




class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        """
        Custom extractor for observation dict:
          - "observation": a 3D voxel grid of shape (D, H, W)
          - "drone_positions": a low-dimensional vector
        """
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)

        self.original_voxel_shape = observation_space.spaces["observation"].shape[:3]
        self.downsampled_shape = (40, 40, 12)

        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Flatten()
        )

        dummy_voxel = th.zeros(1, 3, *self.downsampled_shape)
        cnn_output_dim = self.cnn3d(dummy_voxel).shape[1]

        fusion_input_dim = cnn_output_dim + 3  # 3 for drone_positions
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        voxel = observations["observation"]  # (B, D, H, W)

        # One-hot encoding
        channel_unknown  = (voxel == 0).unsqueeze(1).float()
        channel_free     = (voxel == 1).unsqueeze(1).float()
        channel_obstacle = (voxel == 2).unsqueeze(1).float()
        voxel_input = th.cat([channel_unknown, channel_free, channel_obstacle], dim=1)  # (B, 3, D, H, W)

        # Downsample
        voxel_input_downsampled = downsample_voxel_grid_priority(voxel_input, self.downsampled_shape)

        # CNN
        cnn_features = self.cnn3d(voxel_input_downsampled)

        # Concatenate with drone position
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

    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path="./models/", name_prefix="ppo_checkpoint")
    model.learn(total_timesteps=10_000_000, callback=checkpoint_callback)
    model.save("ppo_drone_exploration_model")


'''
if __name__ == "__main__":
    # 1. Create your vectorized environment
    env = make_vec_env(DroneExplorationEnv, n_envs=10, vec_env_cls=SubprocVecEnv)
    check_env(DroneExplorationEnv(), warn=True)

    # 2. Load the existing model (and re‑attach it to our env)
    model = PPO.load("ppo_drone_exploration_model", env=env)
    # Note: custom_objects is only needed if you want to override saved hyperparams.

    # 3. (Optional) Set up a checkpoint callback so you get periodic backups
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./models/",
        name_prefix="ppo_checkpoint"
    )

    # 4. Continue training for additional timesteps
    additional_timesteps = 500_000  # e.g. train 500k more steps
    model.learn(
        total_timesteps=additional_timesteps,
        reset_num_timesteps=False
    )

    # 5. Save (overwrite) the improved model
    model.save("ppo_drone_exploration_model")
    print(f"Model re‑trained for {additional_timesteps} steps and saved.")
'''
