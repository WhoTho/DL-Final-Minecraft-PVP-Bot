"""
Skill distillation model using supervised learning.

This implements:
1. Supervised learning from skill datasets
2. Loading distilled models for PPO fine-tuning
3. Combined policy network for all skills
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
from tqdm import tqdm
import gymnasium as gym

from environments.combined.environment import CombinedEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    SB3_AVAILABLE = True
except ImportError:
    print(
        "Stable Baselines3 not available. Install with: pip install stable-baselines3[extra]"
    )
    SB3_AVAILABLE = False


class SkillDataset(Dataset):
    """Dataset for skill distillation"""

    def __init__(self, observations, actions):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class DistilledPolicyNetwork(nn.Module):
    """
    Distilled policy network that learns from all three skills.

    Architecture:
    - Shared encoder processes observations
    - Three skill-specific heads (movement, aiming, clicking)
    - Output combines all skills into single action
    """

    def __init__(self, obs_dim=20, action_dim=9, hidden_dim=256):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Movement head: [w, a, s, d, space, sprint] (6 actions)
        self.movement_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Sigmoid(),  # Binary movement actions
        )

        # Aiming head: [dyaw, dpitch] (2 actions)
        self.aiming_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh(),  # Continuous angle deltas
        )

        # Clicking head: [click] (1 action)
        self.clicking_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Binary click action
        )

    def forward(self, obs):
        # Encode observation
        features = self.encoder(obs)

        # Get skill-specific outputs
        movement = self.movement_head(features)
        aiming = self.aiming_head(features)
        clicking = self.clicking_head(features)

        # Combine into single action vector
        action = torch.cat([movement, clicking, aiming], dim=-1)

        return action

    def predict(self, obs, deterministic=True):
        """
        Predict action for a single observation (compatible with SB3 interface)
        """
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.FloatTensor(obs)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            action = self.forward(obs)
            return action.cpu().numpy()


class DistilledFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that wraps the distilled policy network's encoder.
    Used for PPO fine-tuning.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        obs_dim = observation_space.shape[0]
        hidden_dim = 256

        # Same architecture as distilled network's encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)


class DistillationModel:
    """
    Main class for skill distillation and fine-tuning.
    """

    def __init__(self, obs_dim=20, action_dim=9, hidden_dim=256, device="cpu"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Create distilled policy network
        self.policy = DistilledPolicyNetwork(obs_dim, action_dim, hidden_dim).to(device)

        self.optimizer = None
        self.ppo_model = None

    def train_supervised(
        self,
        dataset_path,
        epochs=50,
        batch_size=256,
        lr=1e-3,
        val_split=0.1,
        save_path="distilled_model.pth",
    ):
        """
        Train the distilled policy using supervised learning.
        """
        print(f"\nLoading dataset from {dataset_path}")
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)

        observations = data["observations"]
        actions = data["actions"]

        print(f"Dataset size: {len(observations)} samples")
        print(f"Observation shape: {observations.shape}")
        print(f"Action shape: {actions.shape}")

        # Split into train/val
        n_val = int(len(observations) * val_split)
        indices = np.random.permutation(len(observations))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_dataset = SkillDataset(
            observations[train_indices], actions[train_indices]
        )
        val_dataset = SkillDataset(observations[val_indices], actions[val_indices])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

        # Setup optimizer and loss
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")

        print(f"\nTraining for {epochs} epochs...")
        for epoch in range(epochs):
            # Training
            self.policy.train()
            train_loss = 0.0
            train_batches = 0

            for obs, target_actions in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs}"
            ):
                obs = obs.to(self.device)
                target_actions = target_actions.to(self.device)

                # Forward pass
                pred_actions = self.policy(obs)
                loss = criterion(pred_actions, target_actions)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            train_loss /= train_batches

            # Validation
            self.policy.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for obs, target_actions in val_loader:
                    obs = obs.to(self.device)
                    target_actions = target_actions.to(self.device)

                    pred_actions = self.policy(obs)
                    loss = criterion(pred_actions, target_actions)

                    val_loss += loss.item()
                    val_batches += 1

            val_loss /= val_batches

            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(save_path)
                print(f"  → Saved best model (val_loss: {val_loss:.6f})")

        print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
        print(f"Model saved to {save_path}")

        return best_val_loss

    def load(self, path):
        """Load distilled policy weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Loaded distilled model from {path}")

    def save(self, path):
        """Save distilled policy weights"""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
            },
            path,
        )

    def create_ppo_model(self, env_kwargs=None, model_kwargs=None):
        """
        Create a PPO model initialized with distilled policy weights for fine-tuning.
        """
        if not SB3_AVAILABLE:
            raise ImportError("Stable Baselines3 required for PPO fine-tuning")

        env_kwargs = env_kwargs or {}
        model_kwargs = model_kwargs or {}

        n_envs = 4

        # Create vectorized environment
        env = make_vec_env(
            lambda: CombinedEnv(**env_kwargs),
            n_envs=n_envs,
        )

        # Setup policy kwargs with feature extractor
        policy_kwargs = {
            "features_extractor_class": DistilledFeatureExtractor,
            "features_extractor_kwargs": dict(features_dim=128),
            "net_arch": [dict(pi=[128, 64], vf=[128, 64])],
        }

        default_kwargs = {
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 512,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": policy_kwargs,
            "device": self.device,
        }

        kwargs = default_kwargs.copy()
        kwargs.update(model_kwargs)

        # Create PPO model
        self.ppo_model = PPO("MlpPolicy", env, verbose=1, **kwargs)

        print(f"Created PPO model with {n_envs} parallel environments")
        print("Note: Transfer learning from distilled policy to PPO network")
        print("      Attempting to initialize PPO policy with distilled weights...")

        # Try to transfer weights from distilled policy to PPO policy
        try:
            self._transfer_to_ppo()
            print("✓ Successfully transferred distilled weights to PPO policy")
        except Exception as e:
            print(f"⚠ Could not transfer weights: {e}")
            print("  Starting PPO with random initialization")

        return self.ppo_model

    def _transfer_to_ppo(self):
        """
        Transfer distilled policy weights to PPO policy network.
        This is a best-effort initialization.
        """
        if self.ppo_model is None:
            return

        # Get PPO policy network
        ppo_policy = self.ppo_model.policy

        # Transfer feature extractor weights
        distilled_encoder = self.policy.encoder
        ppo_features_extractor = ppo_policy.features_extractor.encoder

        # Copy layer weights (type: ignore for children() calls)
        dist_layers = list(distilled_encoder.children())  # type: ignore
        ppo_layers = list(ppo_features_extractor.children())  # type: ignore

        for dist_layer, ppo_layer in zip(dist_layers, ppo_layers):
            if isinstance(dist_layer, nn.Linear) and isinstance(ppo_layer, nn.Linear):
                if dist_layer.weight.shape == ppo_layer.weight.shape:
                    with torch.no_grad():
                        ppo_layer.weight.copy_(dist_layer.weight)
                        ppo_layer.bias.copy_(dist_layer.bias)

    def fine_tune_ppo(self, total_timesteps, eval_freq, eval_episodes, save_path):
        """
        Fine-tune the distilled policy using PPO.
        """
        if self.ppo_model is None:
            raise ValueError("PPO model not created. Call create_ppo_model first.")

        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.callbacks import EvalCallback

        # Create eval environment
        eval_env = CombinedEnv()

        # Setup evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=f"{save_path}_logs",
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        )

        print(f"\nFine-tuning with PPO for {total_timesteps} timesteps...")
        self.ppo_model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )

        # Save final model
        final_path = f"{save_path}_final"
        self.ppo_model.save(final_path)
        print(f"Fine-tuned model saved to {final_path}")

        return final_path

    def predict(self, observation, deterministic=True):
        """Make a prediction using the distilled policy"""
        return self.policy.predict(observation, deterministic)

    def evaluate(self, n_episodes=10, render=False):
        """Evaluate the distilled policy in the environment"""
        env = CombinedEnv(render_mode="human" if render else None)

        total_rewards = []

        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            truncated = False

            while not done and not truncated:
                action = self.predict(obs, deterministic=True)[0]
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward

                if render:
                    env.render()

            total_rewards.append(episode_reward)
            print(f"Episode {episode+1}/{n_episodes}: Reward = {episode_reward:.2f}")

        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)

        print(f"\nMean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        env.close()

        return mean_reward, std_reward


def train_distilled_model(
    dataset_path="datasets/combined_dataset.pkl", epochs=50, batch_size=256, lr=1e-3
):
    """Train a distilled model from skill datasets"""

    print("=" * 70)
    print("SKILL DISTILLATION - SUPERVISED LEARNING")
    print("=" * 70)

    model = DistillationModel(obs_dim=20, action_dim=9, hidden_dim=256, device="cpu")

    val_loss = model.train_supervised(
        dataset_path=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        save_path="distilled_model_best.pth",
    )

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATING DISTILLED MODEL")
    print("=" * 70)
    model.evaluate(n_episodes=10, render=False)

    return model


def fine_tune_with_ppo(
    distilled_model_path="distilled_model_best.pth", timesteps=100_000
):
    """Fine-tune distilled model with PPO"""

    print("=" * 70)
    print("FINE-TUNING WITH PPO")
    print("=" * 70)

    # Load distilled model
    model = DistillationModel(obs_dim=20, action_dim=9, hidden_dim=256, device="cpu")
    model.load(distilled_model_path)

    # Create PPO model initialized with distilled weights
    model.create_ppo_model()

    # Fine-tune
    save_path = model.fine_tune_ppo(
        total_timesteps=timesteps,
        eval_freq=timesteps // 20,
        eval_episodes=10,
        save_path="distilled_ppo_finetuned",
    )

    return model, save_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "finetune":
        # Fine-tune mode
        timesteps = 100_000 if len(sys.argv) <= 2 else int(sys.argv[2])
        model, save_path = fine_tune_with_ppo(timesteps=timesteps)
    else:
        # Training mode
        epochs = 50 if len(sys.argv) <= 1 else int(sys.argv[1])
        model = train_distilled_model(epochs=epochs)
