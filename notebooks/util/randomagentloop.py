import gymnasium as gym

from notebooks.util.metricstracker import MetricsTracker


def run_random_gymnasium_loop(env_name, num_timesteps):
    # Create the gym environment
    env = gym.make(env_name, render_mode=None)

    # Reset the environment to get the initial state
    observation = env.reset()
    tracker = MetricsTracker()

    episode_reward = 0
    # Run the gymnasium loop
    for _ in range(num_timesteps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            tracker.record_reward(agent_id="random policy", reward=episode_reward)
            obs, info = env.reset()

    # Close the environment
    env.close()
    MetricsTracker().plot_rewards("RANDOM")

if __name__ == "__main__":
    run_random_gymnasium_loop("ALE/DonkeyKong-v5", int(1e6))
