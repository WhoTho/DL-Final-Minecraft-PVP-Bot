import numpy as np
import gymnasium as gym
from gymnasium import spaces
from helpers import vec3, angles
from simulator import combat
from helpers import world

MAX_DEGREES_PER_STEP = 360 / 0.5 / 20  # 360 degrees in 0.5s at 20 steps/s


class AimingEnv(gym.Env):
    """
    Player fixed at (0,0,0) with eye height 1.62.
    Target moves with velocity.
    Action: Δyaw, Δpitch (in radians)
    Observation:
        dx, dy, dz (relative target position)
        yaw, pitch (current angles in radians)
    Reward:
        - distance between aim direction and target direction
        + bonus for close alignment
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Player fixed position with eye height
        self.player_pos = vec3.from_list([0.0, 1.62, 0.0])

        # Yaw/pitch angles (in radians)
        self.yaw = 0.0
        self.pitch = 0.0

        # Target state
        self.target_pos = vec3.zero()
        self.target_vel = vec3.zero()

        # Action: Δyaw, Δpitch in radians (small adjustments)
        max_delta = np.radians(MAX_DEGREES_PER_STEP)
        self.action_space = spaces.Box(
            low=np.array([-max_delta, -max_delta], dtype=np.float32),
            high=np.array([max_delta, max_delta], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )

        # Episode length
        self.max_steps = 100
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random initial angles
        self.yaw = self.np_random.uniform(-np.pi, np.pi)
        self.pitch = self.np_random.uniform(-np.pi / 6, np.pi / 6)  # -30 to 30 degrees

        # Random target position around player (3-8 blocks away)
        distance = self.np_random.uniform(3.0, 8.0)
        target_yaw = self.np_random.uniform(-np.pi, np.pi)
        target_pitch = self.np_random.uniform(
            -np.pi / 12, np.pi / 12
        )  # slight vertical variance

        # Calculate target position
        target_direction = vec3.from_yaw_pitch(target_yaw, target_pitch)
        target_direction = vec3.scale(target_direction, distance)
        self.target_pos = vec3.add(self.player_pos, target_direction)

        # Add some height variation
        self.target_pos[1] = self.np_random.uniform(0.5, 2.5)

        # Random velocity (slow strafing motion)
        self.target_vel = vec3.from_list(
            [
                self.np_random.uniform(-0.05, 0.05),
                self.np_random.uniform(-0.02, 0.02),
                self.np_random.uniform(-0.05, 0.05),
            ]
        )

        self.current_step = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Apply Δyaw, Δpitch
        dyaw, dpitch = action
        self.yaw = angles.yaw_difference(0, self.yaw + dyaw)  # Wrap yaw properly
        self.pitch = np.clip(
            self.pitch + dpitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01
        )  # Clamp pitch

        # Compute reward
        reward = self._compute_reward(action)

        # Update target position
        self.target_pos = vec3.add(self.target_pos, self.target_vel)

        # Check termination
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """Get current observation"""
        forward, right, up = world.yaw_pitch_to_basis_vectors(self.yaw, self.pitch)

        agent_to_target_world = vec3.subtract(self.target_pos, self.player_pos)

        agent_to_target_local = world.world_to_local(
            agent_to_target_world, forward, right, up
        )
        agent_to_target_local_dir, agent_to_target_distance = vec3.direction_and_length(
            agent_to_target_local
        )

        target_velocity_local = world.world_to_local(
            self.target_vel, forward, right, up
        )
        target_velocity_local_dir, target_velocity_speed = vec3.direction_and_length(
            target_velocity_local
        )

        obs = np.array(
            [
                agent_to_target_local_dir[0],
                agent_to_target_local_dir[1],
                agent_to_target_local_dir[2],
                agent_to_target_distance / 10.0,  # normalize distance
                target_velocity_local_dir[0],
                target_velocity_local_dir[1],
                target_velocity_local_dir[2],
                target_velocity_speed / 0.1,  # normalize speed
            ],
            dtype=np.float32,
        )

        return np.clip(obs, -1.0, 1.0)

    def _compute_reward(self, action):
        """Reward based on how close the agent is aiming at the target, with jitter penalty."""
        # Direction to target
        direction_to_target = vec3.subtract(self.target_pos, self.player_pos)
        target_yaw, target_pitch, distance_to_target = angles.vec_to_yaw_pitch_distance(
            direction_to_target
        )

        # Angular error using proper angle difference functions
        yaw_error = abs(angles.yaw_difference(self.yaw, target_yaw))
        pitch_error = abs(angles.pitch_difference(self.pitch, target_pitch))
        # angular_error = yaw_error + pitch_error

        # Base reward is negative error, scaled and squared
        reward = -(yaw_error**2 + pitch_error**2) * 1.0  # Scale factor

        # Bonus if very close to perfect aim (within ~1.15 degrees)
        # if yaw_error < 0.02 and pitch_error < 0.02:
        #     reward += 2.0
        # elif yaw_error < 0.05 and pitch_error < 0.05:
        #     reward += 0.5

        # penalty for movement
        reward -= 0.1 * (abs(action[0]) + abs(action[1]))

        if yaw_error < np.radians(10) and pitch_error < np.radians(
            30
        ):  # quick check to avoid unnecessary calculations
            # reward for looking at closest axis
            target_aabb_min = vec3.subtract(
                self.target_pos, vec3.from_list([0.3, 1.62, 0.3])
            )
            target_aabb_max = vec3.add(
                self.target_pos, vec3.from_list([0.3, 1.8 - 1.62, 0.3])
            )
            look_dir = vec3.from_yaw_pitch(self.yaw, self.pitch)
            did_hit, distance_to_intersection = combat.line_intersects_aabb(
                self.player_pos,
                look_dir,
                target_aabb_min,
                target_aabb_max,
            )

            if did_hit:
                reward += (distance_to_target - distance_to_intersection + 1) * 5.0

        return reward

    def get_perfect_action(self):
        """Get the perfect action to aim at target (for testing/debugging)"""
        direction_to_target = vec3.subtract(self.target_pos, self.player_pos)
        target_yaw, target_pitch, _ = angles.vec_to_yaw_pitch_distance(
            direction_to_target
        )

        dyaw = angles.yaw_difference(self.yaw, target_yaw)
        dpitch = angles.pitch_difference(self.pitch, target_pitch)

        # Clamp to action space
        max_delta = np.radians(MAX_DEGREES_PER_STEP)
        dyaw = np.clip(dyaw, -max_delta, max_delta)
        dpitch = np.clip(dpitch, -max_delta, max_delta)

        return np.array([dyaw, dpitch], dtype=np.float32)

    def render(self):
        """Simple text-based render for debugging"""
        if self.render_mode == "human":
            direction_to_target = vec3.subtract(self.target_pos, self.player_pos)
            target_yaw, target_pitch, distance = angles.vec_to_yaw_pitch_distance(
                direction_to_target
            )

            yaw_error = abs(angles.yaw_difference(self.yaw, target_yaw))
            pitch_error = abs(angles.pitch_difference(self.pitch, target_pitch))

            print(f"Step: {self.current_step}")
            print(
                f"Current: yaw={np.degrees(self.yaw):.1f}°, pitch={np.degrees(self.pitch):.1f}°"
            )
            print(
                f"Target:  yaw={np.degrees(target_yaw):.1f}°, pitch={np.degrees(target_pitch):.1f}°"
            )
            print(
                f"Error:   yaw={np.degrees(yaw_error):.1f}°, pitch={np.degrees(pitch_error):.1f}°"
            )
            print(f"Distance: {distance:.2f}")
            print("-" * 50)
