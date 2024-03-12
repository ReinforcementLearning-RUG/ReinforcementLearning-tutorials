import gymnasium as gym
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env

vec_env = make_atari_env("ALE/DonkeyKong-v5", n_envs=4, seed=0)     # n_envs=4 train on 4 instances of the environment in parallel.
vec_env = VecFrameStack(vec_env, n_stack=4)

if not os.path.exists("./dqn_dk.zip"):
    model = DQN(
        "CnnPolicy",               # What model to use to approximate Q-function.
        vec_env,
        verbose=1,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,    # epsilon-greedy schedule
        learning_rate=1e-4,
        batch_size=32,
        learning_starts=100000,
        target_update_interval=1000,
        buffer_size=100000,             # Replay buffer size
        optimize_memory_usage=False
)

if not os.path.exists("./dqn_dk.zip"):
    model.learn(total_timesteps=1000000)
    model.save("dqn_dk")
else:
    model = DQN.load("./dqn_dk.zip")
