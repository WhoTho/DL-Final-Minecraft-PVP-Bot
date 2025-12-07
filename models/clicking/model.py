from models.base_model import build_and_train_model
from environments.clicking.environment import ClickingEnv

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        timesteps = int(sys.argv[1])
    else:
        timesteps = 100_000

    build_and_train_model(
        model_name="clicking",
        environment_class=ClickingEnv,
        timesteps=timesteps,
    )
