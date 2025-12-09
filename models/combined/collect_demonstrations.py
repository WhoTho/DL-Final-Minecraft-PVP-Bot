"""
Demonstration Collection for Skill Distillation

Collects expert demonstrations by running all three experts together
in the combined environment and recording observations and actions.
"""

from time import time
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from tqdm.rich import tqdm
import pickle

from stable_baselines3 import PPO
from environments.combined.environment import CombinedEnv


class ExpertWrapper:
    """Wrapper for a single-skill expert model."""

    def __init__(self, model_path: str, skill_name: str, device: str = "cpu"):
        """
        Args:
            model_path: Path to the trained expert model
            skill_name: Name of the skill (for logging)
            device: Device to run the model on
        """
        self.model_path = model_path
        self.skill_name = skill_name
        self.device = device
        self.model = PPO.load(model_path, device=device)
        print(f"✓ Loaded {skill_name} expert from {model_path}")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get expert action for the given observation."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action


def collect_demonstrations(
    n_episodes: int = 100,
    save_dir: str = "training_results",
    log_dir: str = "logs/distillation",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect demonstrations from all experts in the combined environment.

    Args:
        n_episodes: Number of demonstration episodes to collect
        save_dir: Directory to save demonstrations
        log_dir: Directory for logging

    Returns:
        Tuple of (observations, actions_dict)
    """
    print("\n" + "=" * 70)
    print("DEMONSTRATION COLLECTION")
    print("=" * 70)

    # Setup paths
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Load expert models
    print("\nLoading expert models...")
    movement_expert = ExpertWrapper("training_results/movement_latest", "Movement")
    aiming_expert = ExpertWrapper("training_results/aiming_latest", "Aiming")
    clicking_expert = ExpertWrapper("training_results/clicking_latest", "Clicking")

    # Create environment
    print("Creating environment...")
    demo_env = CombinedEnv()

    # Collect demonstrations
    print(f"\nCollecting {n_episodes} demonstration episodes...\n")

    observations = []
    actions = []

    for episode in tqdm(range(n_episodes), desc="Episodes", position=0):
        obs, _ = demo_env.reset()
        done = False
        episode_steps = 0

        while not done:
            # Get expert actions
            movement_action = movement_expert.predict(obs, deterministic=True)
            look_action = aiming_expert.predict(obs, deterministic=True)
            click_action = clicking_expert.predict(obs, deterministic=True)

            # Combine actions into flattened Box format
            # [w, a, s, d, space, sprint, click, dyaw, dpitch]
            combined_action = np.array(
                [
                    float(movement_action[0]),  # w
                    float(movement_action[1]),  # a
                    float(movement_action[2]),  # s
                    float(movement_action[3]),  # d
                    float(movement_action[4]),  # space
                    float(movement_action[5]),  # sprint
                    float(click_action),  # click
                    float(look_action[0]),  # dyaw
                    float(look_action[1]),  # dpitch
                ],
                dtype=np.float32,
            )

            # Store data
            observations.append(obs)
            actions.append(combined_action)

            # Step environment
            obs, reward, terminated, truncated, info = demo_env.step(combined_action)
            done = terminated or truncated
            episode_steps += 1

        tqdm.write(f"Episode {episode + 1}: {episode_steps} steps")

    # Convert to numpy arrays
    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)

    print(f"\n{'='*70}")
    print("COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total timesteps collected: {len(observations)}")
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"{'='*70}\n")

    # Save demonstrations
    demo_path = f"{save_dir}/demonstrations.pkl"
    Path(demo_path).parent.mkdir(parents=True, exist_ok=True)
    with open(demo_path, "wb") as f:
        pickle.dump({"observations": observations, "actions": actions}, f)
    print(f"✓ Demonstrations saved to {demo_path}\n")

    return observations, actions


if __name__ == "__main__":
    import sys

    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    collect_demonstrations(n_episodes=n_episodes)
