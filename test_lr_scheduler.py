import gymnasium as gym
import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Simple callback for testing learning rate scheduling
class TestLearningRateScheduler(BaseCallback):
    def __init__(self, initial_lr=3e-4, min_lr=5e-6, decay_factor=0.5, 
                 decay_steps=500_000, verbose=1):
        super(TestLearningRateScheduler, self).__init__(verbose)
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
        
        # Log if verbose - force print every 100 steps for testing
        if self.n_calls % 100 == 0:
            print(f"Timestep: {self.num_timesteps}, Learning rate: {new_lr:.6f}")
            # Print the model's actual learning rate to confirm changes
            print(f"Model's learning rate: {self.model.learning_rate:.6f}")
            
        return True

# Simple environment
class DummyEnv(gym.Env):
    def __init__(self):
        super(DummyEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, **kwargs):
        observation = np.random.random(4)
        return observation, {}

    def step(self, action):
        observation = np.random.random(4)
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

if __name__ == "__main__":
    # Create a vectorized environment
    venv = make_vec_env(lambda: DummyEnv(), n_envs=1)
    
    # Initialize the model
    model = PPO("MlpPolicy", venv, verbose=1, learning_rate=3e-4)
    
    # Initialize the learning rate scheduler
    lr_scheduler = TestLearningRateScheduler(
        initial_lr=3e-4,
        min_lr=5e-6,
        decay_factor=0.75,
        decay_steps=100,  # Smaller number for quick testing
        verbose=1
    )
    cb = CheckpointCallback(save_freq=50_000, save_path="./test/", name_prefix="test")
    callbacks = [cb, lr_scheduler]
    # Train the model with the scheduler
    model.learn(total_timesteps=2000, callback=callbacks)
    print("Training completed")
