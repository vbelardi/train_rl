import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize, VecFrameStack
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO
import voxelgrid
from swarm_gym import DroneExplorationEnv

class Custom3DGridExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        D, H, W = observation_space.spaces["observation"].shape
        # 3D CNN
        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 32, (5,5,3), (2,2,1), (1,1,1)), nn.ReLU(),
            nn.Conv3d(32,32,3,2,1), nn.ReLU(),
            nn.Conv3d(32,64,3,2,1), nn.ReLU(),
            nn.Conv3d(64,64,3,1,1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            d = D//4; h = H//4; w = W//4
            dummy = torch.zeros(1,3,d,h,w)
            flat = self.cnn3d(dummy).shape[1]
        # position MLP
        self.pos_mlp = nn.Sequential(
            nn.Linear(3,32), nn.ReLU(),
            nn.Linear(32,32), nn.ReLU()
        )
        # fusion
        self.fuse = nn.Sequential(
            nn.Linear(flat+32,1024), nn.ReLU(),
            nn.Linear(1024,512), nn.ReLU(),
            nn.Linear(512,features_dim), nn.ReLU()
        )
        self._features_dim = features_dim

    def forward(self, obs):
        v = obs["observation"].long()
        u = (v==0).unsqueeze(1).float()
        f = (v==1).unsqueeze(1).float()
        o = (v==2).unsqueeze(1).float()
        x = torch.cat([u,f,o],dim=1)
        _, D, H, W = obs["observation"].shape
        x = F.adaptive_avg_pool3d(x, output_size=(D//4, H//4, W//4))
        c = self.cnn3d(x)
        p = self.pos_mlp(obs["drone_positions"])
        return self.fuse(torch.cat([c,p],1))

def make_env():
    env = DroneExplorationEnv()
    return env

if __name__ == "__main__":
    venv = make_vec_env(DroneExplorationEnv, n_envs=20, vec_env_cls=SubprocVecEnv)
    #venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
    #venv = VecFrameStack(venv, n_stack=4)

    policy_kwargs = dict(
        features_extractor_class=Custom3DGridExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model = RecurrentPPO(
        "MultiInputLstmPolicy", venv,
        policy_kwargs=policy_kwargs,
        n_steps=512, n_epochs=10,
        learning_rate=1e-4, gamma=0.98,
        gae_lambda=0.95, ent_coef=1e-2,
        clip_range=0.2, verbose=1
    )
    cb = CheckpointCallback(save_freq=25_000, save_path="./models/", name_prefix="rppo_check")
    model.learn(total_timesteps=10_000_000, callback=cb)
    model.save("rppo_3dcuriosity")