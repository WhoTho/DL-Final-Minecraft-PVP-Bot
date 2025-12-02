import numpy as np
import gymnasium as gym
from gymnasium import spaces
from helpers import vec3, angles

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
            low=-np.radians(180), high=np.radians(180), shape=(2,), dtype=np.float32
        )

        # Episode length
        self.max_steps = 200
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
        reward = self._compute_reward()

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
        target_to_player = vec3.subtract(self.target_pos, self.player_pos)
        target_yaw, target_pitch, distance_to_target = angles.vec_to_yaw_pitch_distance(
            target_to_player
        )
        target_yaw_diff = angles.yaw_difference(self.yaw, target_yaw)
        target_pitch_diff = angles.pitch_difference(self.pitch, target_pitch)

        return np.array(
            [
                # distance_to_target,
                target_yaw_diff,
                target_pitch_diff,
            ],
            dtype=np.float32,
        )

    def _compute_reward(self):
        """Reward based on how close the agent is aiming at the target."""
        # Direction to target
        direction_to_target = vec3.subtract(self.target_pos, self.player_pos)
        target_yaw, target_pitch, _ = angles.vec_to_yaw_pitch_distance(
            direction_to_target
        )

        # Angular error using proper angle difference functions
        yaw_error = abs(angles.yaw_difference(self.yaw, target_yaw))
        pitch_error = abs(angles.pitch_difference(self.pitch, target_pitch))
        angular_error = yaw_error + pitch_error

        # MAX_ANGLE_ERROR = np.radians(180 + 90)  # Max possible error (yaw + pitch)

        # Reward is negative error, scaled and squared
        # reward = -(1 - (1 - (angular_error / MAX_ANGLE_ERROR)) ** 2)
        reward = -(yaw_error**2 + pitch_error**2) * 10.0  # Scale factor

        # Bonus if very close to perfect aim (within ~1.15 degrees)
        if yaw_error < 0.02 and pitch_error < 0.02:
            reward += 2.0
        elif yaw_error < 0.05 and pitch_error < 0.05:
            reward += 0.5

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
            print("-" * 40)
