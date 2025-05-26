import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize, VecFrameStack
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO
import voxelgrid
from swarm_gym_mindist import DroneExplorationEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class Custom3DGridExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        D, H, W = observation_space.spaces["observation"].shape
        drone_shape = observation_space.spaces["drone_positions"].shape
        # 3D CNN
        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 32, (5,5,3), (2,2,1), (1,1,1)), nn.ReLU(),
            nn.Conv3d(32, 32, (5,5,3), (2,2,1), (1,1,1)), nn.ReLU(),
            nn.Conv3d(32,64,3,2,1), nn.ReLU(),
            nn.Conv3d(64,128,3,1,1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            d = D//4; h = H//4; w = W//4
            dummy = torch.zeros(1,3,d,h,w)
            flat = self.cnn3d(dummy).shape[1]
        # position MLP
        self.pos_mlp = nn.Sequential(
            nn.Linear(drone_shape[0],32), nn.ReLU(),
            nn.Linear(32,64), nn.ReLU(),
            nn.Linear(64,64), nn.ReLU(),
        )
        # fusion
        self.fuse = nn.Sequential(
            nn.Linear(flat+64,1024), nn.ReLU(),
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
        
        # Update both model's learning rate and optimizer's learning rate
        self.model.learning_rate = new_lr
        
        # THIS IS THE KEY LINE - Update optimizer directly
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Log if verbose
        if self.verbose > 0 and self.n_calls % 10000 == 0:
            print(f"Timestep: {self.num_timesteps}, Learning rate: {new_lr:.6f}")
            actual_lr = self.model.policy.optimizer.param_groups[0]['lr']
            print(f"Optimizer's actual learning rate: {actual_lr:.6f}")
            
        return True


if __name__ == "__main__":
    venv = make_vec_env(DroneExplorationEnv, n_envs=20, vec_env_cls=SubprocVecEnv)
    #venv = VecNormalize(venv, norm_obs=True, norm_reward=True)


    policy_kwargs = dict(
        features_extractor_class=Custom3DGridExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model = RecurrentPPO(
        "MultiInputLstmPolicy", venv,
        device=device,
        policy_kwargs=policy_kwargs,
        n_steps=180, n_epochs=3,
        learning_rate=3e-4, gamma=0.9,
        gae_lambda=0.9, ent_coef=1e-6,
        clip_range=0.2, target_kl=0.1, verbose=1
    )
    cb = CheckpointCallback(save_freq=50_000, save_path="./finalsim/", name_prefix="finalsim_check")
    #lr_scheduler = LearningRateScheduler(initial_lr=3e-4, min_lr=5e-6, decay_factor=0.75, decay_steps=25_000)

    #callbacks = [cb, lr_scheduler]
    model.learn(total_timesteps=10_000_000, callback=cb)
    model.save("sim1_final")

'''
if __name__ == "__main__":
    # 1. Create your vectorized environment
    env = make_vec_env(DroneExplorationEnv, n_envs=20, vec_env_cls=SubprocVecEnv)

    # 2. Load the existing model (and re‑attach it to our env)
    model = RecurrentPPO.load("./finalsim/finalsim_check_19000000_steps", env=env, learning_rate=5e-5, gamma=0.90,ent_coef=1e-6, gae_lambda=0.9, clip_range=0.2, n_epochs=3, n_steps=360, target_kl=0.1)
    # Note: custom_objects is only needed if you want to override saved hyperparams.

    # 3. (Optional) Set up a checkpoint callback so you get periodic backups
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path="./finalsim/",
        name_prefix="finalsim_check"
    )
    #lr_scheduler = LearningRateScheduler(initial_lr=1e-4, min_lr=5e-6, decay_factor=0.75, decay_steps=500_000)
    # Add the learning rate scheduler to the callback list
    #callbacks = [checkpoint_callback, lr_scheduler]

    # 4. Continue training for additional timesteps
    additional_timesteps = 20_000_000  # e.g. train 500k more steps
    model.learn(
        total_timesteps=additional_timesteps,
        reset_num_timesteps=False,
        callback=checkpoint_callback
    )

    # 5. Save (overwrite) the improved model
    model.save("finalsim_final")
    print(f"Model re‑trained for {additional_timesteps} steps and saved.")
'''