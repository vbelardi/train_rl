import torch
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
from sb3_contrib import RecurrentPPO
from sim_1 import Custom3DGridExtractor

# Test model loading with the same parameters
observation_space = Dict({
    "observation": gym.spaces.Box(low=0, high=2, shape=(67, 67, 20), dtype=np.uint8),
    "drone_positions": gym.spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)
})
action_space = gym.spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)

policy_kwargs = dict(
    features_extractor_class=Custom3DGridExtractor,
    features_extractor_kwargs=dict(features_dim=256),
)

model = RecurrentPPO.load(
    "sim1_check_4000000_steps",
    custom_objects={
        "features_extractor_class": Custom3DGridExtractor,
        "policy_kwargs": policy_kwargs,
        "observation_space": observation_space,
        "action_space": action_space,
    }
)
print("Model loaded successfully!")