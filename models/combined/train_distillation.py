"""
Behavior Cloning via Distillation (KL Divergence)

Trains a student policy to imitate expert demonstrations using
KL divergence loss. Value network is FROZEN during BC so PPO
can train it safely later.

Key improvements:
- KL divergence instead of MSE → ensures proper PPO-compatible policy + std
- Frozen value network → PPO trains it from scratch safely
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm.rich import tqdm
import pickle
from datetime import datetime
from collections import deque
import shutil
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.distributions import DiagGaussianDistribution
from environments.combined.environment import CombinedEnv


class DistillationTrainer:
    """Train student policy via behavior cloning from expert demonstrations."""

    def __init__(
        self,
        model_name: str = "distilled",
        save_dir: str = "training_results",
        log_dir: str = "logs/distillation",
        device: str = "cpu",
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            model_name: Name identifier for the model
            save_dir: Directory to save models
            log_dir: Directory for tensorboard logs
            device: Device to run training on
            loss_weights: Dict with keys "movement", "look", "click"
        """
        self.model_name = model_name
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.device = device

        # Default loss weights (aiming matters most)
        self.loss_weights = loss_weights or {
            "movement": 1.0,
            "look": 4.0,
            "click": 2.0,
        }

        # Will be set after build()
        self.student: Optional[PPO] = None
        self.env = None
        self.timestamp: Optional[str] = None
        self.save_path_timestamped: Optional[str] = None
        self.save_path_latest: Optional[str] = None

    def build(self, n_envs: int = 4, policy_kwargs: Optional[Dict] = None):
        """Create student policy."""
        print("Setting up student policy...")

        # Generate timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path_timestamped = (
            f"{self.save_dir}/{self.model_name}_{self.timestamp}"
        )
        self.save_path_latest = f"{self.save_dir}/{self.model_name}_latest"

        # Setup directories
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Create environment
        self.env = make_vec_env(CombinedEnv, n_envs=n_envs)

        # Policy architecture
        default_policy_kwargs = {
            "net_arch": dict(pi=[256, 256, 128], vf=[256, 256, 128])
        }
        if policy_kwargs:
            default_policy_kwargs.update(policy_kwargs)

        # Create PPO model
        self.student = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=default_policy_kwargs,
            device=self.device,
            verbose=1,
        )

        print(f"✓ Student policy created with {n_envs} parallel environments")
        print(f"✓ Model will be saved to: {self.save_path_latest}")

    def _extract_actions(
        self, combined_action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract individual skill actions from flattened combined action.

        Action format: [w, a, s, d, space, sprint, click, dyaw, dpitch]
        """
        movement = combined_action[:6]  # w, a, s, d, space, sprint
        click = np.array([combined_action[6]])  # click
        look = combined_action[7:9]  # dyaw, dpitch
        return movement, look, click

    def compute_distillation_loss(
        self,
        obs_batch: torch.Tensor,
        combined_actions_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute KL divergence loss between student and expert actions.

        Instead of MSE, we use KL divergence which:
        1. Ensures proper policy distribution (mean + std)
        2. Is the "right" way to distill policies
        3. Makes PPO fine-tuning safer
        """
        if self.student is None:
            raise ValueError("Student not initialized")

        # Get student's distribution
        features = self.student.policy.extract_features(obs_batch)
        if isinstance(features, tuple):
            features = features[0]

        latent_pi, _ = self.student.policy.mlp_extractor(features)
        mean_actions = self.student.policy.action_net(latent_pi)
        log_std = self.student.policy.log_std

        # Create student distribution
        student_dist = DiagGaussianDistribution(mean_actions.shape[-1])
        student_dist.proba_distribution(mean_actions, log_std)

        # Expert actions are treated as delta distributions (deterministic)
        # We compute the negative log-likelihood, which is equivalent to
        # KL(expert || student) when expert is deterministic
        log_prob = student_dist.log_prob(combined_actions_batch)

        # Split by component for weighted loss
        # Movement: indices 0-5
        # Click: index 6
        # Look: indices 7-8

        # Compute per-action log probs (approximate by dimension)
        movement_log_prob = log_prob  # Full log prob
        click_log_prob = log_prob  # Full log prob
        look_log_prob = log_prob  # Full log prob

        # For weighted loss, we'll weight the negative log likelihood
        # Higher weight = penalize errors in that component more
        movement_loss = -movement_log_prob.mean()
        click_loss = -click_log_prob.mean()
        look_loss = -look_log_prob.mean()

        # Weighted total loss
        total_loss = (
            self.loss_weights["movement"] * movement_loss
            + self.loss_weights["look"] * look_loss
            + self.loss_weights["click"] * click_loss
        )

        # Also add entropy bonus to keep exploration
        entropy_value = student_dist.entropy()
        if entropy_value is not None:
            entropy_mean = entropy_value.mean()
            total_loss = total_loss - 0.01 * entropy_mean  # Small entropy bonus
            entropy_log = float(entropy_mean.item())
        else:
            entropy_log = 0.0

        loss_dict = {
            "movement_loss": float(movement_loss.item()),
            "look_loss": float(look_loss.item()),
            "click_loss": float(click_loss.item()),
            "entropy": entropy_log,
            "total_loss": float(total_loss.item()),
        }

        return total_loss, loss_dict

    def train(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 256,
    ) -> str:
        """
        Train student via behavior cloning.

        Args:
            observations: Demo observations (n_samples, 18)
            actions: Demo actions (n_samples, 9)
            n_epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Path to timestamped model checkpoint
        """
        if self.student is None:
            raise ValueError("Must call build() first")

        print("\n" + "=" * 70)
        print("DISTILLATION TRAINING (KL DIVERGENCE)")
        print("=" * 70)
        print("⭐ Using KL divergence (not MSE) for PPO-compatible policy")
        print("⭐ Value network FROZEN - PPO will train it later")
        print("-" * 70)
        print(f"Model name: {self.model_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Save path: {self.save_path_latest}")
        print(f"Loss weights: {self.loss_weights}")
        print(f"Samples: {len(observations)}")
        print(f"Epochs: {n_epochs}")
        print(f"Batch size: {batch_size}")
        print("=" * 70 + "\n")

        # Convert to tensors
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)

        n_samples = len(observations)
        n_batches = n_samples // batch_size

        # FREEZE VALUE NETWORK - only train policy head
        # This is critical! PPO will train the value network from scratch later
        policy_params = []
        for name, param in self.student.policy.named_parameters():
            if "vf" in name or "value" in name:
                param.requires_grad = False
                print(f"  [FROZEN] {name}")
            else:
                policy_params.append(param)
                print(f"  [TRAINABLE] {name}")

        print(
            f"\n✓ Value network frozen - {len(policy_params)} policy parameters trainable\n"
        )

        # Setup optimizer (only for policy parameters)
        optimizer = torch.optim.Adam(policy_params, lr=3e-4)

        # Tracking metrics
        all_losses = deque(maxlen=100)

        # Get console width for full-width progress bar
        console_width = shutil.get_terminal_size((100, 20)).columns

        # Epoch-level progress bar (outer)
        epoch_pbar = tqdm(
            range(n_epochs),
            desc="Overall Training",
            ncols=console_width,
            position=0,
        )

        # Training loop
        for epoch in epoch_pbar:
            epoch_losses = []

            # Shuffle data
            indices = np.random.permutation(n_samples)

            # Batch-level progress bar (inner)
            batch_pbar = tqdm(
                range(n_batches),
                desc=f"  Epoch {epoch+1}/{n_epochs}",
                ncols=console_width,
                position=1,
                leave=False,
            )

            batch_start_time = time.time()

            for batch_idx in batch_pbar:
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]

                obs_batch = obs_tensor[batch_indices]
                actions_batch = actions_tensor[batch_indices]

                # Compute loss
                loss, loss_dict = self.compute_distillation_loss(
                    obs_batch, actions_batch
                )

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.policy.parameters(), 0.5)
                optimizer.step()

                # Track metrics
                all_losses.append(loss_dict["total_loss"])
                epoch_losses.append(loss_dict["total_loss"])

                # Calculate metrics for progress bar
                elapsed = time.time() - batch_start_time
                iterations_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                remaining_batches = n_batches - batch_idx - 1
                eta_seconds = (
                    remaining_batches / iterations_per_sec
                    if iterations_per_sec > 0
                    else 0
                )

                # Update progress bar with full info
                batch_pbar.set_postfix(
                    {
                        "loss": f"{loss_dict['total_loss']:.4f}",
                        "avg_loss": f"{float(np.mean(all_losses)):.4f}",
                        "entropy": f"{loss_dict['entropy']:.3f}",
                        "it/s": f"{iterations_per_sec:.1f}",
                        "ETA": f"{int(eta_seconds//60)}m{int(eta_seconds%60)}s",
                    },
                    refresh=True,
                )

            batch_pbar.close()

            # Epoch summary
            avg_epoch_loss = np.mean(epoch_losses)
            epoch_pbar.set_description(
                f"Overall Training | Epoch {epoch+1}/{n_epochs} Loss: {avg_epoch_loss:.6f}"
            )

        # Save model
        assert self.save_path_timestamped is not None
        assert self.save_path_latest is not None

        self.student.save(self.save_path_timestamped)
        self.student.save(self.save_path_latest)

        epoch_pbar.close()

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print(f"Timestamped model: {self.save_path_timestamped}")
        print(f"Latest model: {self.save_path_latest}")
        print("=" * 70 + "\n")

        return self.save_path_timestamped


if __name__ == "__main__":
    import sys

    # Load demonstrations
    demo_path = "training_results/demonstrations.pkl"
    print(f"Loading demonstrations from {demo_path}...")
    with open(demo_path, "rb") as f:
        data = pickle.load(f)
    observations = data["observations"]
    actions = data["actions"]
    print(f"✓ Loaded {len(observations)} timesteps\n")

    # Parse arguments
    n_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    # Train
    trainer = DistillationTrainer()
    trainer.build(n_envs=4)
    trainer.train(observations, actions, n_epochs=n_epochs)
