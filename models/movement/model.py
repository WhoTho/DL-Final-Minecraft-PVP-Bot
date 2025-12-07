from models.base_model import build_and_train_model
from environments.movement.environment import MovementEnv

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        timesteps = int(sys.argv[1])
    else:
        timesteps = 100_000

    build_and_train_model(
        model_name="movement",
        environment_class=MovementEnv,
        timesteps=timesteps,
        model_kwargs={
            "policy_kwargs": {
                "net_arch": {
                    "pi": [128, 128],
                    "vf": [128, 128],
                }
            },
        },
    )
