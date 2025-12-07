"""
Create datasets from trained PPO models for skill distillation.

This script:
1. Loads trained movement, aiming, and clicking models
2. Runs each model in its environment to collect (obs, action) pairs
3. Transforms observations to combined observation space
4. Saves datasets for supervised learning
"""

import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

from environments.movement.environment import MovementEnv
from environments.aiming.environment import AimingEnv
from environments.clicking.environment import ClickingEnv
from helpers import vec3, angles, world
from simulator.physics import GROUND_Y


def movement_obs_to_combined(obs, agent, target):
    """
    Convert movement observation to combined observation space.

    Movement obs (20 dims):
    [to_target_local(3), dist(1), agent_vel_local(3), agent_speed(1),
     target_vel_local(3), target_speed(1), agent_on_ground(1), target_on_ground(1),
     target_look_local(3), prev_target_look_local(3)]

    Combined obs (20 dims):
    [to_target_local(3), dist(1), agent_vel_local(3), agent_speed(1),
     target_vel_local(3), target_speed(1), agent_health(1), target_health(1),
     agent_on_ground(1), target_on_ground(1), target_look_local(3), invuln(1)]
    """
    # Movement env has 20 dims: matches combined, but different order
    # We need to reorder and add health/invuln info

    combined_obs = np.zeros(20, dtype=np.float32)

    # Copy spatial features (indices 0-13 are the same in both)
    combined_obs[0:4] = obs[0:4]  # to_target_local(3), dist(1)
    combined_obs[4:8] = obs[4:8]  # agent_vel_local(3), agent_speed(1)
    combined_obs[8:12] = obs[8:12]  # target_vel_local(3), target_speed(1)

    # Health (normalized) - use current entity health
    combined_obs[12] = agent.health / 20.0  # agent_health
    combined_obs[13] = target.health / 20.0  # target_health

    # Ground flags
    combined_obs[14] = obs[12]  # agent_on_ground
    combined_obs[15] = obs[13]  # target_on_ground

    # Target look direction
    combined_obs[16:19] = obs[14:17]  # target_look_local(3)

    # Invulnerability (normalized)
    combined_obs[19] = np.clip(target.invulnerablility_ticks / 10.0, 0.0, 1.0)

    return combined_obs


def aiming_obs_to_combined(obs, agent, target):
    """
    Convert aiming observation to combined observation space.

    Aiming obs (5 dims): [sin(yaw_diff), cos(yaw_diff), sin(pitch_diff), cos(pitch_diff), dist_norm]

    Combined obs (20 dims): Need to reconstruct full state from limited info
    """
    # Extract aim information
    sin_yaw = obs[0]
    cos_yaw = obs[1]
    sin_pitch = obs[2]
    cos_pitch = obs[3]
    distance_norm = obs[4]

    # Reconstruct yaw/pitch differences
    yaw_diff = np.arctan2(sin_yaw, cos_yaw)
    pitch_diff = np.arctan2(sin_pitch, cos_pitch)
    distance = distance_norm * 15.0  # Unnormalize

    # Compute direction to target in agent's local frame
    # In local frame, forward is (0, 0, -1), so we need to rotate by yaw_diff, pitch_diff
    # Simplified: use spherical coordinates
    to_target_local = vec3.from_list(
        [
            np.sin(yaw_diff) * np.cos(pitch_diff),
            np.sin(pitch_diff),
            -np.cos(yaw_diff) * np.cos(pitch_diff),
        ]
    )

    combined_obs = np.zeros(20, dtype=np.float32)

    # Direction and distance
    combined_obs[0:3] = to_target_local
    combined_obs[3] = distance / 10.0  # Normalize for combined

    # Velocities - use actual entity velocities if available
    forward, right, up = world.yaw_pitch_to_basis_vectors(agent.yaw, agent.pitch)

    agent_vel_dir, agent_speed = vec3.direction_and_length(agent.velocity)
    if agent_speed > 1e-6:
        agent_vel_local = world.world_to_local(agent_vel_dir, forward, right, up)
    else:
        agent_vel_local = vec3.zero()

    target_vel_dir, target_speed = vec3.direction_and_length(target.velocity)
    if target_speed > 1e-6:
        target_vel_local = world.world_to_local(target_vel_dir, forward, right, up)
    else:
        target_vel_local = vec3.zero()

    combined_obs[4:7] = agent_vel_local
    combined_obs[7] = agent_speed / 2.0
    combined_obs[8:11] = target_vel_local
    combined_obs[11] = target_speed / 2.0

    # Health
    combined_obs[12] = agent.health / 20.0
    combined_obs[13] = target.health / 20.0

    # Ground flags
    combined_obs[14] = 1.0 if agent.on_ground else -1.0
    combined_obs[15] = 1.0 if target.on_ground else -1.0

    # Target look direction (in agent's local frame)
    target_look_world = vec3.from_yaw_pitch(target.yaw, target.pitch)
    target_look_local = world.world_to_local(target_look_world, forward, right, up)
    combined_obs[16:19] = target_look_local

    # Invulnerability
    combined_obs[19] = np.clip(target.invulnerablility_ticks / 10.0, 0.0, 1.0)

    return np.clip(combined_obs, -1.0, 1.0)


def clicking_obs_to_combined(obs, agent, target):
    """
    Convert clicking observation to combined observation space.

    Clicking obs (9 dims): [yaw_err_norm, pitch_err_norm, dist_norm, in_range, invuln_status,
                            target_pos_x, target_pos_z, agent_yaw_norm, agent_pitch_norm]

    Combined obs (20 dims): Need to reconstruct from limited info
    """
    yaw_error_norm = obs[0]
    pitch_error_norm = obs[1]
    distance_norm = obs[2]
    in_range = obs[3]
    invuln_status = obs[4]
    target_pos_x = obs[5]
    target_pos_z = obs[6]
    agent_yaw_norm = obs[7]
    agent_pitch_norm = obs[8]

    # Reconstruct angles
    yaw_error = yaw_error_norm * np.pi
    pitch_error = pitch_error_norm * (np.pi / 2)
    distance = distance_norm * 10.0

    # Agent orientation
    agent_yaw = agent_yaw_norm * np.pi
    agent_pitch = agent_pitch_norm * (np.pi / 2)

    # Compute direction to target in agent's local frame
    # Target yaw/pitch from agent's perspective
    target_yaw = agent_yaw + yaw_error
    target_pitch = agent_pitch + pitch_error

    # Direction vector to target in world frame
    to_target_world = vec3.from_list(
        [
            distance * np.cos(target_pitch) * np.sin(target_yaw),
            distance * np.sin(target_pitch),
            -distance * np.cos(target_pitch) * np.cos(target_yaw),
        ]
    )

    # Transform to agent's local frame
    forward, right, up = world.yaw_pitch_to_basis_vectors(agent.yaw, agent.pitch)
    to_target_local_dir = vec3.normalize(to_target_world)
    to_target_local = world.world_to_local(to_target_local_dir, forward, right, up)

    combined_obs = np.zeros(20, dtype=np.float32)

    # Direction and distance
    combined_obs[0:3] = to_target_local
    combined_obs[3] = distance_norm  # Already normalized 0-1

    # Velocities - use actual entity velocities
    agent_vel_dir, agent_speed = vec3.direction_and_length(agent.velocity)
    if agent_speed > 1e-6:
        agent_vel_local = world.world_to_local(agent_vel_dir, forward, right, up)
    else:
        agent_vel_local = vec3.zero()

    target_vel_dir, target_speed = vec3.direction_and_length(target.velocity)
    if target_speed > 1e-6:
        target_vel_local = world.world_to_local(target_vel_dir, forward, right, up)
    else:
        target_vel_local = vec3.zero()

    combined_obs[4:7] = agent_vel_local
    combined_obs[7] = agent_speed / 2.0
    combined_obs[8:11] = target_vel_local
    combined_obs[11] = target_speed / 2.0

    # Health
    combined_obs[12] = agent.health / 20.0
    combined_obs[13] = target.health / 20.0

    # Ground flags (assume both on ground in clicking env)
    combined_obs[14] = 1.0 if agent.on_ground else -1.0
    combined_obs[15] = 1.0 if target.on_ground else -1.0

    # Target look direction
    target_look_world = vec3.from_yaw_pitch(target.yaw, target.pitch)
    target_look_local = world.world_to_local(target_look_world, forward, right, up)
    combined_obs[16:19] = target_look_local

    # Invulnerability
    combined_obs[19] = invuln_status

    return np.clip(combined_obs, -1.0, 1.0)


def movement_action_to_combined(action):
    """
    Convert movement action (6 dims) to combined action (9 dims).

    Movement: [w, a, s, d, space, sprint]
    Combined: [w, a, s, d, space, sprint, click, dyaw, dpitch]
    """
    combined_action = np.zeros(9, dtype=np.float32)
    combined_action[0:6] = action  # Copy movement actions
    # click=0, dyaw=0, dpitch=0 (no aiming/clicking in movement env)
    return combined_action


def aiming_action_to_combined(action):
    """
    Convert aiming action (2 dims) to combined action (9 dims).

    Aiming: [dyaw, dpitch]
    Combined: [w, a, s, d, space, sprint, click, dyaw, dpitch]
    """
    combined_action = np.zeros(9, dtype=np.float32)
    # No movement or clicking, only aiming
    combined_action[7:9] = action  # dyaw, dpitch
    return combined_action


def clicking_action_to_combined(action):
    """
    Convert clicking action (discrete 2) to combined action (9 dims).

    Clicking: 0 or 1 (no click or click)
    Combined: [w, a, s, d, space, sprint, click, dyaw, dpitch]
    """
    combined_action = np.zeros(9, dtype=np.float32)
    combined_action[6] = float(action)  # click action
    # No movement or aiming in clicking env (stationary agent)
    return combined_action


def collect_skill_data(
    model_path,
    env,
    obs_converter,
    action_converter,
    num_episodes,
    skill_name,
):
    """
    Collect dataset from a trained model.

    Returns:
        observations: List of observations in combined space
        actions: List of actions in combined space
    """
    from stable_baselines3 import PPO

    print(f"\nCollecting {skill_name} data from {model_path}")

    # Load model
    model = PPO.load(model_path, device="cpu")

    observations = []
    actions = []

    for episode in tqdm(range(num_episodes), desc=f"Collecting {skill_name}"):
        obs, _ = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)

            # Convert to combined space
            combined_obs = obs_converter(obs, env.agent, env.target)
            combined_action = action_converter(action)

            observations.append(combined_obs)
            actions.append(combined_action)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)

    print(f"Collected {len(observations)} samples from {skill_name}")

    return np.array(observations), np.array(actions)


def create_datasets(
    movement_model, aiming_model, clicking_model, num_episodes=100, save_dir="datasets"
):
    """
    Create datasets for all three skills and save them.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    datasets = {}

    # Movement dataset
    if movement_model:
        env = MovementEnv()
        obs, actions = collect_skill_data(
            movement_model,
            env,
            movement_obs_to_combined,
            movement_action_to_combined,
            num_episodes,
            "movement",
        )
        datasets["movement"] = {"observations": obs, "actions": actions}
        env.close()

    # Aiming dataset
    if aiming_model:
        env = AimingEnv()
        obs, actions = collect_skill_data(
            aiming_model,
            env,
            aiming_obs_to_combined,
            aiming_action_to_combined,
            num_episodes,
            "aiming",
        )
        datasets["aiming"] = {"observations": obs, "actions": actions}
        env.close()

    # Clicking dataset
    if clicking_model:
        env = ClickingEnv()
        obs, actions = collect_skill_data(
            clicking_model,
            env,
            clicking_obs_to_combined,
            clicking_action_to_combined,
            num_episodes,
            "clicking",
        )
        datasets["clicking"] = {"observations": obs, "actions": actions}
        env.close()

    # Save datasets
    for skill, data in datasets.items():
        save_path = save_dir / f"{skill}_dataset.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {skill} dataset to {save_path}")

    # Create combined dataset (all skills together)
    all_obs = np.concatenate([data["observations"] for data in datasets.values()])
    all_actions = np.concatenate([data["actions"] for data in datasets.values()])

    # Shuffle
    indices = np.random.permutation(len(all_obs))
    all_obs = all_obs[indices]
    all_actions = all_actions[indices]

    combined_dataset = {"observations": all_obs, "actions": all_actions}
    save_path = save_dir / "combined_dataset.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(combined_dataset, f)
    print(f"Saved combined dataset to {save_path}")
    print(f"Total samples: {len(all_obs)}")

    return datasets, combined_dataset


if __name__ == "__main__":
    import sys

    # Parse arguments
    num_episodes = 100 if len(sys.argv) <= 1 else int(sys.argv[1])

    # Model paths
    movement_model = "best_baseline_ppo_movement_final"
    aiming_model = "best_baseline_ppo_aiming_final"
    clicking_model = "best_baseline_ppo_clicking_final"

    print("=" * 70)
    print("CREATING SKILL DISTILLATION DATASETS")
    print("=" * 70)
    print(f"Episodes per skill: {num_episodes}")
    print(f"Movement model: {movement_model}")
    print(f"Aiming model: {aiming_model}")
    print(f"Clicking model: {clicking_model}")
    print("=" * 70)

    datasets, combined = create_datasets(
        movement_model, aiming_model, clicking_model, num_episodes=num_episodes
    )

    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    for skill, data in datasets.items():
        print(f"{skill.capitalize()}:")
        print(f"  Observations shape: {data['observations'].shape}")
        print(f"  Actions shape: {data['actions'].shape}")
    print(f"\nCombined:")
    print(f"  Observations shape: {combined['observations'].shape}")
    print(f"  Actions shape: {combined['actions'].shape}")
    print("=" * 70)
