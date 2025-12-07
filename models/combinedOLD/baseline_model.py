"""
Combined model that integrates movement, aiming, and clicking skills
Uses pre-trained models as a starting point and trains them together

NOTE: This is the OLD approach. For better results, use skill distillation:
    1. Run: python -m models.combined.create_skill_datasets
    2. Run: python -m models.combined.distill_skills
    3. (Optional) Fine-tune: python -m models.combined.distill_skills finetune

Or use the workflow script: ./train_distilled.sh
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from environments.combined.environment import CombinedEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.policies import ActorCriticPolicy

    torch.set_default_tensor_type(torch.FloatTensor)

    SB3_AVAILABLE = True
except ImportError:
    print(
        "Stable Baselines3 not available. Install with: pip install stable-baselines3[extra]"
    )
    SB3_AVAILABLE = False


class CombinedSkillNetwork(BaseFeaturesExtractor):
    """
    Custom feature extractor that processes observation through skill-specific networks

    Architecture:
    - Movement subnet: Processes spatial/velocity information
    - Aiming subnet: Processes target direction/distance
    - Clicking subnet: Processes aim error and timing
    - Fusion layer: Combines all skills for final action
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        obs_dim = observation_space.shape[0]

        # Movement subnet (indices 0-11: positions, velocities, speeds)
        self.movement_net = nn.Sequential(
            nn.Linear(12, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
        )

        # Aiming subnet (indices 0-3: direction to target, distance)
        self.aiming_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
        )

        # Clicking subnet (indices 0-3, 12-13, 19: aim info, health, invuln)
        self.clicking_net = nn.Sequential(
            nn.Linear(7, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
        )

        # Fusion layer combines all skills
        self.fusion = nn.Sequential(
            nn.Linear(32 + 16 + 16, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract relevant features for each skill
        # Observation structure: [dir_to_target(3), dist(1), agent_vel(3), agent_speed(1),
        #                         target_vel(3), target_speed(1), agent_health(1), target_health(1),
        #                         on_ground(2), target_look(3), invuln(1)]

        # Movement features: all spatial and velocity info
        movement_features = torch.cat(
            [
                observations[:, 0:4],  # dir + dist
                observations[:, 4:12],  # velocities and speeds
            ],
            dim=1,
        )

        # Aiming features: target direction and distance
        aiming_features = observations[:, 0:4]

        # Clicking features: aim direction, health, invuln status
        clicking_features = torch.cat(
            [
                observations[:, 0:4],  # dir + dist (for aim error)
                observations[:, 12:14],  # health states
                observations[:, 19:20],  # invuln status
            ],
            dim=1,
        )

        # Process through skill-specific networks
        movement_out = self.movement_net(movement_features)
        aiming_out = self.aiming_net(aiming_features)
        clicking_out = self.clicking_net(clicking_features)

        # Fuse all skills
        combined = torch.cat([movement_out, aiming_out, clicking_out], dim=1)
        return self.fusion(combined)


class CombinedModel:
    """
    Combined model that integrates pre-trained movement, aiming, and clicking models
    """

    def __init__(
        self,
        movement_model_path=None,
        aiming_model_path=None,
        clicking_model_path=None,
        env_kwargs=None,
        model_kwargs=None,
    ):
        self.env_kwargs = env_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.n_envs = 4  # Number of parallel environments

        # Store model paths for reference
        self.movement_model_path = movement_model_path
        self.aiming_model_path = aiming_model_path
        self.clicking_model_path = clicking_model_path

        # Create vectorized training environment
        self.env = make_vec_env(
            lambda: CombinedEnv(**self.env_kwargs),
            n_envs=self.n_envs,
        )
        # Single eval environment
        self.eval_env = CombinedEnv(**self.env_kwargs)

        # Setup policy kwargs with custom feature extractor
        policy_kwargs = {
            "features_extractor_class": CombinedSkillNetwork,
            "features_extractor_kwargs": dict(features_dim=128),
            "net_arch": [dict(pi=[128, 64], vf=[128, 64])],
        }

        default_kwargs = {
            "learning_rate": 3e-4,
            "n_steps": 1024,  # 1024 steps per env * 4 envs = 4096 total
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

        # Create model
        self.model = PPO("MlpPolicy", self.env, verbose=1, **kwargs)

        # Load pre-trained weights if provided
        if any([movement_model_path, aiming_model_path, clicking_model_path]):
            self._load_pretrained_skills(
                movement_model_path, aiming_model_path, clicking_model_path
            )

        print(f"Created Combined PPO model with {self.n_envs} parallel environments")
        print(
            f"Loaded skills: Movement={movement_model_path is not None}, "
            f"Aiming={aiming_model_path is not None}, "
            f"Clicking={clicking_model_path is not None}"
        )

    def _load_pretrained_skills(
        self, movement_path=None, aiming_path=None, clicking_path=None
    ):
        """
        Load pre-trained weights from individual skill models and initialize combined model

        Note: This is a best-effort initialization. The architectures may not match perfectly,
        but we extract what we can from the pre-trained models.
        """
        print("\nInitializing from pre-trained skill models...")

        loaded_models = {}

        # Load each skill model if path provided
        if movement_path:
            try:
                from models.movement.baseline_model import MovementModel

                temp_movement = MovementModel()
                temp_movement.load(movement_path)
                loaded_models["movement"] = temp_movement.model
                print(f"✓ Loaded movement model from {movement_path}")
            except Exception as e:
                print(f"✗ Failed to load movement model: {e}")

        if aiming_path:
            try:
                from models.aiming.baseline_model import AimingModel

                temp_aiming = AimingModel()
                temp_aiming.load(aiming_path)
                loaded_models["aiming"] = temp_aiming.model
                print(f"✓ Loaded aiming model from {aiming_path}")
            except Exception as e:
                print(f"✗ Failed to load aiming model: {e}")

        if clicking_path:
            try:
                from models.clicking.baseline_model import ClickingModel

                temp_clicking = ClickingModel()
                temp_clicking.load(clicking_path)
                loaded_models["clicking"] = temp_clicking.model
                print(f"✓ Loaded clicking model from {clicking_path}")
            except Exception as e:
                print(f"✗ Failed to load clicking model: {e}")

        # Transfer knowledge from loaded models to combined model
        # This is a heuristic approach - we initialize parts of the network with pre-trained weights
        if loaded_models:
            print("\nTransferring knowledge to combined model...")
            try:
                # Get the feature extractor from combined model
                feature_extractor = self.model.policy.features_extractor

                # Initialize movement subnet with movement model weights if available
                if "movement" in loaded_models:
                    self._transfer_weights(
                        loaded_models["movement"],
                        feature_extractor.movement_net,
                        "movement",
                    )

                # Initialize aiming subnet with aiming model weights if available
                if "aiming" in loaded_models:
                    self._transfer_weights(
                        loaded_models["aiming"],
                        feature_extractor.aiming_net,
                        "aiming",
                    )

                # Initialize clicking subnet with clicking model weights if available
                if "clicking" in loaded_models:
                    self._transfer_weights(
                        loaded_models["clicking"],
                        feature_extractor.clicking_net,
                        "clicking",
                    )

                print("✓ Knowledge transfer complete")
            except Exception as e:
                print(f"⚠ Knowledge transfer partially failed: {e}")
                print("  Continuing with random initialization for failed parts")

    def _transfer_weights(self, source_model, target_subnet, skill_name):
        """
        Transfer weights from pre-trained model to subnet

        This attempts to copy compatible layers from the source model's
        feature extractor to the target subnet.
        """
        try:
            source_extractor = source_model.policy.features_extractor

            # Get source layers
            if hasattr(source_extractor, "children"):
                source_layers = list(source_extractor.children())
                print(source_layers)
            else:
                source_layers = [source_extractor]

            # Get target layers
            target_layers = list(target_subnet.children())

            # Transfer compatible layers
            transferred = 0
            for src_layer, tgt_layer in zip(source_layers, target_layers):
                if isinstance(src_layer, nn.Linear) and isinstance(
                    tgt_layer, nn.Linear
                ):
                    # Check if dimensions are compatible
                    if (
                        src_layer.weight.shape[1] <= tgt_layer.weight.shape[1]
                        and src_layer.weight.shape[0] == tgt_layer.weight.shape[0]
                    ):
                        # Copy weights (partial if input dims don't match)
                        with torch.no_grad():
                            tgt_layer.weight[:, : src_layer.weight.shape[1]].copy_(
                                src_layer.weight
                            )
                            tgt_layer.bias.copy_(src_layer.bias)
                        transferred += 1

            print(f"  {skill_name}: Transferred {transferred} layer(s)")
        except Exception as e:
            print(f"  {skill_name}: Transfer failed - {e}")

    def train(
        self, total_timesteps: int, eval_freq: int, eval_episodes: int, save_path: str
    ):
        """Train the combined model"""

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
        print(f"\nTraining combined model for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True
        )

        # Save final model
        final_path = f"{save_path}_final"
        self.model.save(final_path)
        print(f"Model saved to {final_path}")

        return final_path

    def evaluate(self, n_episodes: int, render: bool = False):
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")

        env = CombinedEnv(render_mode="human" if render else None)

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
        """Load a trained combined model"""
        # Recreate vectorized environment for loading
        env = make_vec_env(
            lambda: CombinedEnv(**self.env_kwargs),
            n_envs=self.n_envs,
        )
        self.model = PPO.load(path, env=env)
        self.env = env
        print(
            f"Loaded combined model from {path} with {self.n_envs} parallel environments"
        )

    def save(self, path):
        """Save the current model"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.save(path)
        print(f"Saved model to {path}")


def train_model(
    timesteps: int,
    movement_model=None,
    aiming_model=None,
    clicking_model=None,
):
    """
    Train the combined model, optionally starting from pre-trained skill models
    """
    print("=" * 70)
    print("COMBINED MODEL TRAINING")
    print("=" * 70)
    print(f"Total timesteps: {timesteps}")
    print(f"Movement model: {movement_model or 'None (random init)'}")
    print(f"Aiming model: {aiming_model or 'None (random init)'}")
    print(f"Clicking model: {clicking_model or 'None (random init)'}")
    print("=" * 70)

    model = CombinedModel(
        movement_model_path=movement_model,
        aiming_model_path=aiming_model,
        clicking_model_path=clicking_model,
    )

    # save_path = model.train(
    #     total_timesteps=timesteps,
    #     eval_freq=timesteps // 20,
    #     eval_episodes=10,
    #     save_path="best_baseline_ppo_combined",
    # )

    # Final evaluation
    print("\nFinal evaluation:")
    model.evaluate(n_episodes=20, render=False)

    save_path = "HIHIHIIHIHIHIIH"
    return model, save_path


if __name__ == "__main__":
    import sys

    # Parse arguments
    timesteps = 500_000 if len(sys.argv) <= 1 else int(sys.argv[1])

    # Optional: Load pre-trained skill models
    movement_model = "best_baseline_ppo_movement_final"
    aiming_model = "best_baseline_ppo_aiming_final"
    clicking_model = "best_baseline_ppo_clicking_final"

    train_model(
        timesteps=timesteps,
        movement_model=movement_model,
        aiming_model=aiming_model,
        clicking_model=clicking_model,
    )
