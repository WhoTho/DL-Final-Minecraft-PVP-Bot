"""
PPO Fine-tuning of Distilled Model

Fine-tunes a distilled student policy with PPO in the combined environment.
This allows coordination between skills and reward optimization.
"""

from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from environments.combined.environment import CombinedEnv


class FineTuner:
    """Fine-tune a distilled model with PPO."""

    def __init__(
        self,
        model_path: str,
        model_name: str = "finetuned",
        save_dir: str = "training_results",
        log_dir: str = "logs",
        device: str = "cpu",
        n_envs: int = 4,
    ):
        """
        Args:
            model_path: Path to distilled model to fine-tune
            model_name: Name for fine-tuned model
            save_dir: Directory to save models
            log_dir: Directory for tensorboard logs
            device: Device to run training on
        """
        self.model_path = model_path
        self.model_name = model_name
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.device = device
        self.n_envs = n_envs

        # Generate timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path_timestamped = (
            f"{self.save_dir}/{self.model_name}_{self.timestamp}"
        )
        self.save_path_latest = f"{self.save_dir}/{self.model_name}_latest"
        self.log_path = f"{self.log_dir}/{self.model_name}_{self.timestamp}"

        # Setup directories
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        self.env = make_vec_env(CombinedEnv, n_envs=self.n_envs)

        # Load distilled model
        print(f"Loading distilled model from {model_path}...")
        self.model = PPO.load(
            model_path, device=device, ent_coef=0.0, clip_range=0.05, learning_rate=1e-5
        )

        print("Reconfiguring model for fine-tuning...")
        # Reset environment
        self.model.set_env(self.env)

        # Recreate optimizer to reset Adam moments
        old_opt = self.model.policy.optimizer
        opt_class = old_opt.__class__
        defaults = old_opt.defaults.copy()

        self.model.policy.optimizer = opt_class(
            self.model.policy.parameters(), **defaults
        )

        print(f"✓ Model loaded\n")

    def fine_tune(
        self,
        total_timesteps: int = 1_000_000,
        eval_freq: Optional[int] = None,
        eval_episodes: int = 10,
    ) -> str:
        """
        Fine-tune the model with PPO.

        Args:
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency (steps between evals)
            eval_episodes: Number of evaluation episodes
            n_envs: Number of parallel environments

        Returns:
            Path to timestamped model checkpoint
        """
        print("=" * 70)
        print("PPO FINE-TUNING")
        print("=" * 70)
        print(f"Model name: {self.model_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Loading from: {self.model_path}")
        print(f"Save path: {self.save_path_latest}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Parallel environments: {self.n_envs}")
        print("=" * 70 + "\n")

        # Calculate eval frequency if not provided
        if eval_freq is None:
            eval_freq = total_timesteps // 20

        # Create evaluation environment
        eval_env = CombinedEnv()

        # Create eval callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.save_path_timestamped,
            log_path=self.log_path,
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        )

        # Update tensorboard log
        self.model.tensorboard_log = self.log_path

        # Train with PPO
        print(f"Starting PPO training for {total_timesteps:,} timesteps...\n")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )

        # Save final models
        self.model.save(self.save_path_latest)

        print("\n" + "=" * 70)
        print("FINE-TUNING COMPLETE")
        print(f"Timestamped model: {self.save_path_timestamped}")
        print(f"Latest model: {self.save_path_latest}")
        print("=" * 70 + "\n")

        return self.save_path_timestamped

    def evaluate(self, n_episodes: int = 10) -> tuple:
        """
        Evaluate the fine-tuned model.

        Args:
            n_episodes: Number of evaluation episodes

        Returns:
            Tuple of (mean_reward, std_reward)
        """
        print(f"\nEvaluating model ({n_episodes} episodes)...\n")

        eval_env = make_vec_env(CombinedEnv, n_envs=4)
        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=n_episodes,
            deterministic=True,
            render=False,
        )

        print(f"\n✓ Mean reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
        return mean_reward, std_reward


if __name__ == "__main__":
    import sys

    # Parse arguments
    distilled_path = "training_results/distilled_latest"
    timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000

    # Fine-tune
    finetuner = FineTuner(
        model_path=distilled_path,
        model_name="finetuned",
    )
    print("\nEvaluating before fine-tuning:")
    finetuner.evaluate(n_episodes=10)

    finetuner.fine_tune(total_timesteps=timesteps)

    print("\nEvaluating fine-tuned model:")
    finetuner.evaluate(n_episodes=20)
