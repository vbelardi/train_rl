
import gymnasium as gym
import torch
import numpy as np
from swarm_gym import DroneExplorationEnv
import voxelgrid
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import torch.nn.functional as F
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize


def downsample_voxel_grid_priority(voxel_input, output_size):
    """
    Downsamples (B,3,D,H,W) en appliquant priorité obstacle>unknown>free via adaptive pool.
    """
    unk  = voxel_input[:, 0:1]
    free = voxel_input[:, 1:2]
    obs  = voxel_input[:, 2:3]
    unk_ds  = F.adaptive_max_pool3d(unk,  output_size).squeeze(1)
    free_ds = F.adaptive_max_pool3d(free, output_size).squeeze(1)
    obs_ds  = F.adaptive_max_pool3d(obs,  output_size).squeeze(1)
    labels = torch.ones_like(obs_ds, dtype=torch.long)    # par défaut free=1
    labels[unk_ds  > 0] = 0   # unknown
    labels[obs_ds  > 0] = 2   # obstacle
    voxel_down = F.one_hot(labels, num_classes=3)
    return voxel_down.permute(0,4,1,2,3).float()

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        # full voxel shape:
        D, H, W = observation_space.spaces["observation"].shape

        # 3D CNN with larger kernels and dilation for wider receptive field
        self.cnn3d = nn.Sequential(
            # Larger first kernel to capture broader context
            nn.Conv3d(3, 16, kernel_size=5, padding=2, padding_mode = 'replicate'), nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1, padding_mode = 'replicate'), nn.ReLU(), nn.AvgPool3d(2), 
            nn.Conv3d(32, 64, kernel_size=(5,5,3), padding=(2,2,1), padding_mode = 'replicate'), nn.ReLU(), nn.AvgPool3d(2),
            nn.Flatten()
        )
        # determine flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, D//4, H//4, W//4)
            cnn_out = self.cnn3d(dummy).shape[1]

        # project CNN features into a compact vector
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out, 512), nn.ReLU(), nn.LayerNorm(512)
        )

        # MLP for drone position encoding
        pos_dim = observation_space.spaces["drone_positions"].shape[0]
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.LayerNorm(128)
        )

        # fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, features_dim), nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, obs):
        # one-hot encode occupancy channels
        vox = obs["observation"].long()
        u = (vox == 0).unsqueeze(1).float()
        f = (vox == 1).unsqueeze(1).float()
        o = (vox == 2).unsqueeze(1).float()
        x = torch.cat([u, f, o], dim=1)  # (B,2,D,H,W)
        _, D, H, W = obs["observation"].shape
        x_small = F.adaptive_avg_pool3d(x, output_size=(D//4, H//4, W//4))

        # CNN feature extraction
        c = self.cnn3d(x_small)        # (B, cnn_out)
        c = self.cnn_proj(c)     # (B,512)

        # position embedding
        p = self.pos_mlp(obs["drone_positions"])  # (B,128)

        # fuse and return
        return self.fusion(torch.cat([c, p], dim=1))  # (B, features_dim)


if __name__ == "__main__":
    # création de l'env
    env = make_vec_env(DroneExplorationEnv, n_envs=20, vec_env_cls=SubprocVecEnv)
    #env = VecNormalize(env, norm_obs=True, norm_reward=True)
    check_env(DroneExplorationEnv(), warn=True)

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256)
    )

    model = PPO(
        "MultiInputPolicy", env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=512,
        batch_size=128,
        ent_coef=0.05,
        learning_rate=1e-4,
        clip_range=0.1,

    )

    cb = CheckpointCallback(save_freq=50_000, save_path="./models/", name_prefix="ppo_ckpt")
    model.learn(total_timesteps=10_000_000, callback=cb)
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
        save_freq=50_000,
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
