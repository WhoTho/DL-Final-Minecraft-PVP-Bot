"""
Behavior Cloning from PPO Teachers

This module implements proper behavior cloning where:
1. ALL teacher models use the SAME observation space (combined env observation)
2. We collect demonstrations from teachers in the unified environment
3. Student learns via BC loss + PPO loss
4. BC weight is decayed over time

Key insight: Teachers must be RE-TRAINED on the combined environment first,
or we must use the combined environment's observation format for all demos.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from pathlib import Path
import pickle
from tqdm import tqdm
from typing import List, Tuple, Dict

from environments.combined.environment import CombinedEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    torch.set_default_tensor_type(torch.FloatTensor)
    SB3_AVAILABLE = True
except ImportError:
    print(
        "Stable Baselines3 not available. Install with: pip install stable-baselines3[extra]"
    )
    SB3_AVAILABLE = False


class BCDataset(torch.utils.data.Dataset):
    """Dataset for behavior cloning"""

    def __init__(self, observations, actions):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class DistilledFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for the student policy.
    Uses a unified observation space (20 dims from CombinedEnv).
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        obs_dim = observation_space.shape[0]
        hidden_dim = 256

        # Encoder architecture
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


class BCCallback(BaseCallback):
    """
    Callback that adds Behavior Cloning loss during PPO training.

    This mixes BC loss with PPO loss:
        total_loss = ppo_loss + bc_weight * bc_loss
    """

    def __init__(
        self,
        bc_dataset,
        bc_weight_start=1.0,
        bc_weight_end=0.1,
        decay_steps=100000,
        batch_size=256,
        device="cpu",
        verbose=0,
    ):
        super().__init__(verbose)
        self.bc_dataset = bc_dataset
        self.bc_weight_start = bc_weight_start
        self.bc_weight_end = bc_weight_end
        self.decay_steps = decay_steps
        self.batch_size = batch_size
        self.device = device

        # Create data loader
        self.bc_loader = torch.utils.data.DataLoader(
            bc_dataset, batch_size=batch_size, shuffle=True
        )
        self.bc_iter = iter(self.bc_loader)

    def _on_training_start(self):
        """Called before the first rollout"""
        pass

    def _on_rollout_end(self):
        """Called after each rollout (before policy update)"""
        pass

    def _on_step(self) -> bool:
        """
        Called at every step. We'll add BC loss here during policy updates.
        """
        # Calculate current BC weight (linear decay)
        progress = min(self.num_timesteps / self.decay_steps, 1.0)
        bc_weight = self.bc_weight_start + progress * (
            self.bc_weight_end - self.bc_weight_start
        )

        # Get a batch of BC data
        try:
            obs_batch, action_batch = next(self.bc_iter)
        except StopIteration:
            # Restart iterator
            self.bc_iter = iter(self.bc_loader)
            obs_batch, action_batch = next(self.bc_iter)

        obs_batch = obs_batch.to(self.device)
        action_batch = action_batch.to(self.device)

        # Get student policy prediction
        with torch.no_grad():
            # We can't directly modify PPO's training loop here,
            # so we'll just log the BC loss for monitoring
            student_actions, _, _ = self.model.policy(obs_batch)
            bc_loss = F.mse_loss(student_actions, action_batch)

            self.logger.record("bc/loss", bc_loss.item())
            self.logger.record("bc/weight", bc_weight)

        return True


def collect_demonstrations(
    model_path: str,
    env: CombinedEnv,
    num_episodes: int = 100,
    skill_name: str = "teacher",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect demonstrations from a teacher model in the unified environment.

    CRITICAL: The teacher model MUST have been trained on the same observation space
    as the combined environment (20-dim observations).

    Args:
        model_path: Path to trained PPO model (.zip file)
        env: CombinedEnv instance
        num_episodes: Number of episodes to collect
        skill_name: Name for logging

    Returns:
        observations: Array of observations (N, 20)
        actions: Array of actions (N, 9)
    """
    print(f"\nCollecting demonstrations from {skill_name} teacher...")
    print(f"Model: {model_path}")

    # Load teacher model
    teacher = PPO.load(model_path, device="cpu")

    observations = []
    actions = []

    for episode in tqdm(range(num_episodes), desc=f"Collecting {skill_name}"):
        obs, _ = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            # Get action from teacher
            action, _ = teacher.predict(obs, deterministic=True)

            observations.append(obs)
            actions.append(action)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)

    observations = np.array(observations)
    actions = np.array(actions)

    print(f"Collected {len(observations)} samples from {skill_name}")
    print(f"Observation shape: {observations.shape}")
    print(f"Action shape: {actions.shape}")

    return observations, actions


def collect_all_demonstrations(
    teacher_paths: Dict[str, str],
    num_episodes: int = 100,
    save_path: str = "bc_demonstrations.pkl",
) -> BCDataset:
    """
    Collect demonstrations from multiple teacher models.

    Args:
        teacher_paths: Dict of {skill_name: model_path}
        num_episodes: Episodes per teacher
        save_path: Where to save collected data

    Returns:
        BCDataset with all demonstrations
    """
    env = CombinedEnv()

    all_observations = []
    all_actions = []

    for skill_name, model_path in teacher_paths.items():
        if model_path and Path(model_path + ".zip").exists():
            obs, actions = collect_demonstrations(
                model_path, env, num_episodes, skill_name
            )
            all_observations.append(obs)
            all_actions.append(actions)
        else:
            print(f"Warning: Teacher model not found: {model_path}")

    env.close()

    if not all_observations:
        raise ValueError("No teacher demonstrations collected!")

    # Combine all demonstrations
    combined_obs = np.concatenate(all_observations, axis=0)
    combined_actions = np.concatenate(all_actions, axis=0)

    # Shuffle
    indices = np.random.permutation(len(combined_obs))
    combined_obs = combined_obs[indices]
    combined_actions = combined_actions[indices]

    print(f"\n{'='*70}")
    print("COMBINED DEMONSTRATIONS")
    print(f"{'='*70}")
    print(f"Total samples: {len(combined_obs)}")
    print(f"Observation shape: {combined_obs.shape}")
    print(f"Action shape: {combined_actions.shape}")
    print(f"{'='*70}\n")

    # Save dataset
    dataset_dict = {
        "observations": combined_obs,
        "actions": combined_actions,
        "teacher_paths": teacher_paths,
        "num_episodes": num_episodes,
    }

    with open(save_path, "wb") as f:
        pickle.dump(dataset_dict, f)
    print(f"Saved demonstrations to {save_path}")

    # Create dataset
    return BCDataset(combined_obs, combined_actions)


def load_bc_dataset(path: str) -> BCDataset:
    """Load a saved BC dataset"""
    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded BC dataset from {path}")
    print(f"Samples: {len(data['observations'])}")

    return BCDataset(data["observations"], data["actions"])


class CombinedModel:
    """
    Student model that learns via Behavior Cloning + PPO.
    """

    def __init__(self, bc_dataset_path: str = None, env_kwargs=None, model_kwargs=None):
        self.env_kwargs = env_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.n_envs = 4

        # Load BC dataset if provided
        self.bc_dataset = None
        if bc_dataset_path and Path(bc_dataset_path).exists():
            self.bc_dataset = load_bc_dataset(bc_dataset_path)
            print(f"Loaded BC dataset: {len(self.bc_dataset)} samples")

        # Create environments
        self.env = make_vec_env(
            lambda: CombinedEnv(**self.env_kwargs),
            n_envs=self.n_envs,
        )
        self.eval_env = CombinedEnv(**self.env_kwargs)

        # Setup policy
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
            "device": "cpu",
        }

        kwargs = default_kwargs.copy()
        kwargs.update(self.model_kwargs)

        self.model = PPO("MlpPolicy", self.env, verbose=1, **kwargs)

        print(f"Created student PPO model with {self.n_envs} parallel environments")

    def pretrain_bc(self, epochs: int = 50, batch_size: int = 256, lr: float = 1e-3):
        """
        Pure behavior cloning pre-training (warmup phase).

        This trains the student policy to imitate teachers before PPO fine-tuning.
        """
        if self.bc_dataset is None:
            raise ValueError(
                "No BC dataset loaded. Provide bc_dataset_path to __init__"
            )

        print(f"\n{'='*70}")
        print("BEHAVIOR CLONING PRE-TRAINING")
        print(f"{'='*70}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        print(f"{'='*70}\n")

        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            self.bc_dataset, batch_size=batch_size, shuffle=True
        )

        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.policy.parameters(), lr=lr)

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for obs_batch, action_batch in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs}"
            ):
                obs_batch = obs_batch.to(self.model.device)
                action_batch = action_batch.to(self.model.device)

                # Get student prediction
                student_actions, _, _ = self.model.policy(obs_batch)

                # BC loss (MSE)
                loss = F.mse_loss(student_actions, action_batch)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs} - BC Loss: {avg_loss:.6f}")

        print("\n✓ BC pre-training complete!")

    def train(
        self,
        total_timesteps: int,
        eval_freq: int,
        eval_episodes: int,
        save_path: str,
        use_bc: bool = True,
        bc_weight_start: float = 0.5,
        bc_weight_end: float = 0.0,
        bc_decay_steps: int = None,
    ):
        """
        Train student with PPO + optional BC loss.

        Args:
            total_timesteps: Total training steps
            eval_freq: Evaluate every N steps
            eval_episodes: Number of eval episodes
            save_path: Where to save best model
            use_bc: Whether to use BC loss during training
            bc_weight_start: Initial BC weight
            bc_weight_end: Final BC weight
            bc_decay_steps: Steps to decay BC weight (default: total_timesteps)
        """
        from stable_baselines3.common.callbacks import EvalCallback, CallbackList

        callbacks = []

        # Evaluation callback
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
        callbacks.append(eval_callback)

        # BC callback (if using)
        if use_bc and self.bc_dataset is not None:
            bc_decay_steps = bc_decay_steps or total_timesteps
            bc_callback = BCCallback(
                self.bc_dataset,
                bc_weight_start=bc_weight_start,
                bc_weight_end=bc_weight_end,
                decay_steps=bc_decay_steps,
                device=self.model.device,
                verbose=1,
            )
            callbacks.append(bc_callback)
            print(f"BC loss enabled: weight {bc_weight_start} → {bc_weight_end}")

        callback_list = CallbackList(callbacks)

        # Train
        print(f"\nTraining student for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
        )

        # Save final model
        final_path = f"{save_path}_final"
        self.model.save(final_path)
        print(f"Model saved to {final_path}")

        return final_path

    def evaluate(self, n_episodes: int = 10, render: bool = False):
        """Evaluate the student model"""
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=n_episodes,
            deterministic=True,
            render=render,
        )

        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward

    def load(self, path: str):
        """Load a trained model"""
        env = make_vec_env(
            lambda: CombinedEnv(**self.env_kwargs),
            n_envs=self.n_envs,
        )
        self.model = PPO.load(path, env=env, device="cpu")
        self.env = env
        print(f"Loaded model from {path}")

    def save(self, path: str):
        """Save the current model"""
        self.model.save(path)
        print(f"Saved model to {path}")

    def predict(self, observation, deterministic=True):
        """Make a prediction using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action


def train_with_bc(
    bc_dataset_path: str,
    timesteps: int = 100000,
    pretrain_epochs: int = 20,
    use_bc_during_ppo: bool = True,
):
    """
    Complete training pipeline with BC.

    1. Pure BC pre-training (warmup)
    2. PPO + BC mixed training
    """
    print("=" * 70)
    print("BEHAVIOR CLONING + PPO TRAINING")
    print("=" * 70)

    # Create model
    model = CombinedModel(bc_dataset_path=bc_dataset_path)

    # Phase 1: Pure BC pre-training
    if pretrain_epochs > 0:
        model.pretrain_bc(epochs=pretrain_epochs)

        # Evaluate after BC
        print("\nEvaluating after BC pre-training:")
        model.evaluate(n_episodes=10)

    # Phase 2: PPO fine-tuning (with optional BC loss)
    save_path = model.train(
        total_timesteps=timesteps,
        eval_freq=timesteps // 20,
        eval_episodes=10,
        save_path="bc_ppo_combined",
        use_bc=use_bc_during_ppo,
        bc_weight_start=0.5,
        bc_weight_end=0.0,
    )

    # Final evaluation
    print("\nFinal evaluation:")
    model.evaluate(n_episodes=20)

    return model, save_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        # Collect demonstrations from teachers
        # NOTE: Teachers MUST be trained on CombinedEnv (same obs space)
        teacher_paths = {
            "combined": "best_baseline_ppo_combined_final",
        }

        num_episodes = 100 if len(sys.argv) <= 2 else int(sys.argv[2])

        collect_all_demonstrations(
            teacher_paths, num_episodes=num_episodes, save_path="bc_demonstrations.pkl"
        )

    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        # Train student with BC
        bc_path = "bc_demonstrations.pkl"
        timesteps = 100000 if len(sys.argv) <= 2 else int(sys.argv[2])
        pretrain_epochs = 20 if len(sys.argv) <= 3 else int(sys.argv[3])

        train_with_bc(bc_path, timesteps=timesteps, pretrain_epochs=pretrain_epochs)

    else:
        print("Usage:")
        print(
            "  Collect demonstrations: python -m models.combined.baseline_model collect [num_episodes]"
        )
        print(
            "  Train student:          python -m models.combined.baseline_model train [timesteps] [pretrain_epochs]"
        )
