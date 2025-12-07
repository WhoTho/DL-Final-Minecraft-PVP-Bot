from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Type, Optional, Dict, Any, Tuple, cast
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

if TYPE_CHECKING:
    from environments.base_enviroment import BaseEnv


class BaseModel:
    """
    Base model class for training PPO agents.
    Handles environment setup, model creation, training, and saving.
    """

    def __init__(self) -> None:
        # Model and environment
        self.model: Optional[PPO] = None
        self.environment_class: Optional[Type["BaseEnv"]] = None
        self.env: Optional[Any] = None
        self.eval_env: Optional["BaseEnv"] = None
        self.env_kwargs: Dict[str, Any] = {}
        self.model_kwargs: Dict[str, Any] = {}
        self.n_envs: int = 0

        # Training configuration
        self.model_name: Optional[str] = None
        self.timestamp: Optional[str] = None
        self.save_dir: Optional[str] = None
        self.log_dir: Optional[str] = None
        self.save_path_timestamped: Optional[str] = None
        self.save_path_latest: Optional[str] = None
        self.log_path: Optional[str] = None

    def build(
        self,
        model_name: str,
        environment_class: Type["BaseEnv"],
        n_envs: int = 4,
        save_dir: str = "training_results",
        log_dir: str = "logs",
        env_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "BaseModel":
        """
        Build the model by setting up environments, configuration, and paths.

        Args:
            model_name: Name identifier for this model
            environment_class: Environment class to instantiate
            n_envs: Number of parallel environments
            save_dir: Directory to save model checkpoints
            log_dir: Directory for tensorboard logs
            env_kwargs: Keyword arguments for environment initialization
            model_kwargs: Keyword arguments for PPO model

        Returns:
            Self for method chaining
        """
        # Store configuration
        self.model_name = model_name
        self.environment_class = environment_class
        self.n_envs = n_envs
        self.env_kwargs = env_kwargs or {}
        self.save_dir = save_dir
        self.log_dir = log_dir

        # Setup directories
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup save paths
        self.save_path_timestamped = (
            f"{self.save_dir}/{self.model_name}_{self.timestamp}"
        )
        self.save_path_latest = f"{self.save_dir}/{self.model_name}_latest"
        self.log_path = f"{self.log_dir}/{self.model_name}_{self.timestamp}"

        env_class = cast(Type["BaseEnv"], self.environment_class)

        # Create training environment
        self.env = make_vec_env(
            lambda: env_class(**self.env_kwargs),
            n_envs=self.n_envs,
        )

        # Create evaluation environment
        self.eval_env = env_class(**self.env_kwargs)

        # Setup PPO hyperparameters
        default_kwargs: Dict[str, Any] = {
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
            "policy_kwargs": dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
            "device": "cpu",
        }

        self.model_kwargs = default_kwargs.copy()
        if model_kwargs:
            self.model_kwargs.update(model_kwargs)

        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=self.log_path,
            **self.model_kwargs,
        )

        # Log configuration
        self._log_build_info()

        return self

    def _log_build_info(self) -> None:
        """Log model build information"""
        assert self.environment_class is not None
        assert self.model_name is not None
        assert self.timestamp is not None
        assert self.save_dir is not None
        assert self.log_dir is not None
        assert self.save_path_timestamped is not None
        assert self.save_path_latest is not None
        assert self.log_path is not None

        print(f"\n{'='*70}")
        print(f"Model Configuration")
        print(f"{'='*70}")
        print(f"Model name:             {self.model_name}")
        print(f"Timestamp:              {self.timestamp}")
        print(f"Environment class:      {self.environment_class.__name__}")
        print(f"Parallel environments:  {self.n_envs}")
        print(f"Save directory:         {self.save_dir}")
        print(f"Log directory:          {self.log_dir}")
        print(f"Timestamped save path:  {self.save_path_timestamped}")
        print(f"Latest save path:       {self.save_path_latest}")
        print(f"Log path:               {self.log_path}")
        print(f"Environment kwargs:     {self.env_kwargs}")
        print(f"Model kwargs:           {self.model_kwargs}")
        print(f"{'='*70}\n")

    def train(
        self,
        total_timesteps: int,
        eval_freq: Optional[int] = None,
        eval_episodes: int = 10,
    ) -> str:
        """
        Train the model with evaluation callbacks.
        Requires build() to be called first to set up paths.

        Args:
            total_timesteps: Number of environment steps for training
            eval_freq: Evaluation frequency (steps between evals)
            eval_episodes: Number of episodes per evaluation

        Returns:
            Path to timestamped model checkpoint
        """
        # Ensure build() was called
        if self.timestamp is None or self.model is None or self.eval_env is None:
            raise ValueError("Must call build() before train()")

        # Calculate eval frequency if not provided
        if eval_freq is None:
            eval_freq = total_timesteps // 20

        assert self.save_path_timestamped is not None
        assert self.log_path is not None

        # Create evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=self.save_path_timestamped,
            log_path=self.log_path,
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        )

        # Train
        print(f"\nTraining {self.model_name} for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )

        # Save final model to latest path
        assert self.save_path_latest is not None
        self.model.save(self.save_path_latest)
        print(f"Latest model saved to {self.save_path_latest}")

        return self.save_path_timestamped

    def evaluate(self, n_episodes: int = 10, render: bool = False) -> Tuple[Any, Any]:
        """
        Evaluate the trained model on the environment.

        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render the environment

        Returns:
            Tuple of (mean_reward, std_reward)
        """
        if self.model is None or self.environment_class is None:
            raise ValueError("Model not trained or loaded yet. Call build() first.")

        # Create fresh environment for evaluation
        env = self.environment_class(**self.env_kwargs)

        mean_reward, std_reward = evaluate_policy(
            self.model,
            env,
            n_eval_episodes=n_episodes,
            deterministic=True,
            render=render,
        )

        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        """
        Make a prediction using the trained model.

        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy

        Returns:
            Predicted action
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def load(self, path: str) -> "BaseModel":
        """
        Load a trained model from disk.

        Args:
            path: Path to model file
        """
        self.model = PPO.load(path, device="cpu")
        print(f"Loaded model from {path}")
        return self

    def save(self, path: str) -> None:
        """
        Save the current model to disk.

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")

        self.model.save(path)
        print(f"Saved model to {path}")


def build_and_train_model(
    model_name: str,
    environment_class: Type["BaseEnv"],
    timesteps: int,
    n_envs: int = 4,
    save_dir: str = "training_results",
    log_dir: str = "logs",
    env_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[BaseModel, str]:
    """
    Build and train a model in one step.

    Args:
        model_name: Name identifier for the model
        environment_class: Environment class to use
        timesteps: Total training timesteps
        n_envs: Number of parallel environments
        save_dir: Directory to save checkpoints
        log_dir: Directory for tensorboard logs
        env_kwargs: Environment keyword arguments
        model_kwargs: PPO model keyword arguments

    Returns:
        Tuple of (trained_model, save_path)
    """
    model = BaseModel()
    model.build(
        model_name=model_name,
        environment_class=environment_class,
        n_envs=n_envs,
        save_dir=save_dir,
        log_dir=log_dir,
        env_kwargs=env_kwargs,
        model_kwargs=model_kwargs,
    )

    save_path = model.train(
        total_timesteps=timesteps,
        eval_freq=timesteps // 100,
        eval_episodes=10,
    )

    print("\nFinal evaluation:")
    model.evaluate(n_episodes=20, render=False)

    return model, save_path
