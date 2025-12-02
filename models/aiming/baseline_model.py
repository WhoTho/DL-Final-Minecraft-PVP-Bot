"""
Baseline aiming model using Stable Baselines3 RL algorithms
"""

import numpy as np
import gymnasium as gym
from enviroments.aiming.enviroment import AimingEnv

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


class BaselineAimingModel:
    """
    Baseline model using various RL algorithms from Stable Baselines3
    """

    def __init__(self, algorithm="PPO", env_kwargs=None, model_kwargs=None):
        if not SB3_AVAILABLE:
            raise ImportError(
                "Stable Baselines3 is required. Install with: pip install stable-baselines3[extra]"
            )

        self.algorithm_name = algorithm
        self.env_kwargs = env_kwargs or {}
        self.model_kwargs = model_kwargs or {}

        # Create environment
        self.env = AimingEnv(**self.env_kwargs)
        self.eval_env = AimingEnv(**self.env_kwargs)

        # Algorithm mapping
        self.algorithms = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C}

        if algorithm not in self.algorithms:
            raise ValueError(
                f"Algorithm {algorithm} not supported. Choose from: {list(self.algorithms.keys())}"
            )

        self.model = None
        self._setup_model()

    def _setup_model(self):
        """Setup the RL model with default hyperparameters"""
        algorithm_class = self.algorithms[self.algorithm_name]

        # Default hyperparameters for each algorithm
        default_kwargs = {
            "PPO": {
                "learning_rate": 3e-4,
                "n_steps": 5000,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "policy_kwargs": dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
                "device": "cpu",
            },
            "SAC": {
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "policy_kwargs": dict(net_arch=[64, 64]),
            },
            "TD3": {
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": (1, "episode"),
                "gradient_steps": -1,
                "policy_kwargs": dict(net_arch=[64, 64]),
            },
            "A2C": {
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.01,
                "vf_coef": 0.25,
                "max_grad_norm": 0.5,
                "policy_kwargs": dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
            },
        }

        # Merge with user-provided kwargs
        kwargs = default_kwargs[self.algorithm_name].copy()
        kwargs.update(self.model_kwargs)

        # Create model
        self.model = algorithm_class("MlpPolicy", self.env, verbose=1, **kwargs)

        print(f"Created {self.algorithm_name} model with kwargs: {kwargs}")

    def train(
        self, total_timesteps=10000, eval_freq=100, eval_episodes=10, save_path=None
    ):
        """
        Train the model with evaluation callbacks
        """
        if save_path is None:
            save_path = f"baseline_{self.algorithm_name.lower()}_aiming_model"

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
        print(f"Training {self.algorithm_name} for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True
        )

        # Save final model
        final_path = f"{save_path}_final"
        self.model.save(final_path)
        print(f"Model saved to {final_path}")

        return final_path

    def evaluate(self, n_episodes=10, render=False):
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")

        env = AimingEnv(render_mode="human" if render else None)

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
        algorithm_class = self.algorithms[self.algorithm_name]
        self.model = algorithm_class.load(path, env=self.env)
        print(f"Loaded {self.algorithm_name} model from {path}")

    def save(self, path):
        """Save the current model"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.save(path)
        print(f"Saved model to {path}")


def compare_algorithms(timesteps=10000, algorithms=None, n_eval_episodes=10):
    """
    Compare different RL algorithms on the aiming task
    """
    if not SB3_AVAILABLE:
        print("Stable Baselines3 not available. Cannot run comparison.")
        return

    if algorithms is None:
        algorithms = ["PPO", "SAC", "A2C"]  # TD3 often needs more tuning

    results = {}

    for alg in algorithms:
        print(f"\n{'='*50}")
        print(f"Training {alg}")
        print(f"{'='*50}")

        try:
            # Create and train model
            model = BaselineAimingModel(algorithm=alg)
            save_path = model.train(
                total_timesteps=timesteps, eval_freq=timesteps // 10
            )

            # Evaluate
            mean_reward, std_reward = model.evaluate(n_episodes=n_eval_episodes)
            results[alg] = {
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "model_path": save_path,
            }

        except Exception as e:
            print(f"Error training {alg}: {e}")
            results[alg] = {"error": str(e)}

    # Print comparison results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")

    for alg, result in results.items():
        if "error" in result:
            print(f"{alg:8}: ERROR - {result['error']}")
        else:
            print(
                f"{alg:8}: {result['mean_reward']:8.2f} ± {result['std_reward']:6.2f}"
            )

    return results


def train_best_model(algorithm="PPO", timesteps=10000):
    """
    Train the best performing model with more timesteps
    """
    print(f"Training {algorithm} model for {timesteps} timesteps...")

    model = BaselineAimingModel(algorithm=algorithm)
    save_path = model.train(
        total_timesteps=timesteps,
        eval_freq=timesteps // 20,
        save_path=f"best_baseline_{algorithm.lower()}_aiming",
    )

    # Final evaluation
    print("\nFinal evaluation:")
    model.evaluate(n_episodes=20, render=False)

    return model, save_path


def demo_trained_model(model_path, algorithm="PPO"):
    """
    Demo a trained model in the visual environment
    """
    print(f"Loading and demonstrating {algorithm} model from {model_path}")

    model = BaselineAimingModel(algorithm=algorithm)
    model.load(model_path)

    # Visual demonstration
    env = AimingEnv(render_mode="human")

    for episode in range(3):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0

        print(f"\nEpisode {episode + 1}:")

        while True:
            action = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            env.render()

            if terminated or truncated:
                break

        print(f"Episode {episode + 1}: {total_reward:.2f} reward in {steps} steps")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "compare":
            # Compare different algorithms
            compare_algorithms(timesteps=300)

        elif command == "train":
            # Train best model
            algorithm = sys.argv[2] if len(sys.argv) > 2 else "PPO"
            train_best_model(algorithm=algorithm, timesteps=50000)

        elif command == "demo":
            # Demo a trained model
            model_path = (
                sys.argv[2] if len(sys.argv) > 2 else "best_baseline_ppo_aiming_final"
            )
            algorithm = sys.argv[3] if len(sys.argv) > 3 else "PPO"
            demo_trained_model(model_path, algorithm)

        else:
            print("Unknown command. Use: compare, train, or demo")

    else:
        # Default: quick comparison
        print("Running quick algorithm comparison...")
        compare_algorithms(timesteps=100, algorithms=["PPO", "A2C"])
