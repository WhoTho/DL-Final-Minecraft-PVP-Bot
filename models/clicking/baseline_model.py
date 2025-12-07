"""
Baseline clicking model using Stable Baselines3 RL algorithms
"""

import numpy as np
import gymnasium as gym
from environments.clicking.environment import ClickingEnv

try:
    from stable_baselines3 import PPO, SAC, TD3, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        StopTrainingOnRewardThreshold,
    )
    from stable_baselines3.common.monitor import Monitor
    import torch

    torch.set_default_tensor_type(torch.FloatTensor)

    SB3_AVAILABLE = True
except ImportError:
    print(
        "Stable Baselines3 not available. Install with: pip install stable-baselines3[extra]"
    )
    SB3_AVAILABLE = False


class ClickingModel:
    def __init__(self, env_kwargs=None, model_kwargs=None):
        self.env_kwargs = env_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.n_envs = 4  # Number of parallel environments

        # Create vectorized training environment (4 parallel envs)
        self.env = make_vec_env(
            lambda: ClickingEnv(**self.env_kwargs),
            n_envs=self.n_envs,
        )
        # Single eval environment
        self.eval_env = ClickingEnv(**self.env_kwargs)

        default_kwargs = {
            "learning_rate": 3e-4,
            "n_steps": 1024,  # 1024 steps per env * 4 envs = 4096 total steps per update
            "batch_size": 512,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
            "device": "cpu",
        }

        kwargs = default_kwargs.copy()
        kwargs.update(self.model_kwargs)

        # Create model
        self.model = PPO("MlpPolicy", self.env, verbose=1, **kwargs)

        print(f"Created PPO model with {self.n_envs} parallel environments")
        print(f"Model kwargs: {kwargs}")

    def train(
        self, total_timesteps: int, eval_freq: int, eval_episodes: int, save_path: str
    ):
        """
        Train the model with evaluation callbacks
        """

        # Setup evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=save_path,
            log_path=f"{save_path}_logs",
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        )

        # Train the model
        print(f"Training clicking model for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True
        )

        # Save final model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_path = f"{save_path}_{timestamp}_final"
        self.model.save(timestamped_path)
        print(f"Model saved to {timestamped_path}")

        # Also save as _latest
        latest_path = f"{save_path}_latest"
        self.model.save(latest_path)
        print(f"Model saved to {latest_path}")

        return timestamped_path

    def evaluate(self, n_episodes: int, render: bool = False):
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")

        env = ClickingEnv(render_mode="human" if render else None)

        mean_reward, std_reward = evaluate_policy(
            self.model,
            env,
            n_eval_episodes=n_episodes,
            deterministic=True,
            render=render,
        )

        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward

    def predict(self, observation, deterministic=True):
        """Make a prediction using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def load(self, path):
        """Load a trained model"""
        # Recreate vectorized environment for loading
        env = make_vec_env(
            lambda: ClickingEnv(**self.env_kwargs),
            n_envs=self.n_envs,
        )
        self.model = PPO.load(path, env=env)
        self.env = env
        print(
            f"Loaded clicking model from {path} with {self.n_envs} parallel environments"
        )

    def save(self, path):
        """Save the current model"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.save(path)
        print(f"Saved model to {path}")


def train_model(timesteps: int):
    """
    Train the best performing model with more timesteps
    """
    print(f"Training PPO model for {timesteps} timesteps...")
    model = ClickingModel()
    save_path = model.train(
        total_timesteps=timesteps,
        eval_freq=timesteps // 20,
        eval_episodes=10,
        save_path=f"best_baseline_ppo_clicking",
    )

    # Final evaluation
    print("\nFinal evaluation:")
    model.evaluate(n_episodes=20, render=False)

    return model, save_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        timesteps = int(sys.argv[1])
    else:
        timesteps = 100_000
    train_model(timesteps=timesteps)
