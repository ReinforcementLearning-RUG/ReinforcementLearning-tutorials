"""
This script contains code for training the DQN agent. Long training times can result in a kernel crash with
jupyter notebooks.
"""

import gymnasium as gym
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env
import threading
import numpy as np
from notebooks.util.metricstracker import MetricsTracker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    The class does two things: Save the best agent periodically and record return values
    so that they can be displayed after training.
    This is technically not proper class design since this callback class it has two purposes,
    but for simplicity and brevity I have it in one class.

    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1, agent_id: str = "agent"):
        super().__init__(verbose)
        self.tracker = MetricsTracker()
        self.agent_id = agent_id

        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Retrieve training reward
        x, y = ts2xy(load_results(self.log_dir), "timesteps")
        # print(y[-1:])
        if len(y) > 0:
            self.tracker.record_reward(self.agent_id, y[-1:][0])

        if self.n_calls % self.check_freq == 0:

            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
                    self.tracker.plot_rewards()

        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.tracker.plot_rewards()
    
if __name__ == "__main__":
    # Create log dir
    log_directory = "./util/logs"
    os.makedirs(log_directory, exist_ok=True)

    # The monitor argument allows us to retrieve relevant information such as the latest reward later on.
    vec_env = make_atari_env("ALE/DonkeyKong-v5", n_envs=4, seed=0, monitor_dir=log_directory)     # n_envs=4 train on 4 instances of the environment in parallel.
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model = DQN(
        "CnnPolicy",               # What model to use to approximate Q-function.
        vec_env,
        verbose=1,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,    # epsilon-greedy schedule
        learning_rate=1e-4,
        batch_size=32,
        learning_starts=100000,
        target_update_interval=1000,
        buffer_size=100000,             # Replay buffer size
        optimize_memory_usage=False
    )
    model.learn(total_timesteps=1e6, callback=SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_directory))    # If you are training for a long time I would do it in a source file and not a jupyter notebook.
    model.save("dqn_trained_dk")
