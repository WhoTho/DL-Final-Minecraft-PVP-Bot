import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from helpers import vec3, angles, world
from simulator.objects import Entity
from simulator.physics import simulate, InputState, GROUND_Y
from simulator import combat
from environments.base_enviroment import (
    BaseEnv,
    MAX_ANGLE_PER_STEP,
)

MOVEMENT_TIMER_INTERVAL = 4  # Change movement input every 4 steps


class AimingEnv(BaseEnv):
    def __init__(self, render_mode=None):
        super().__init__(max_steps=200, render_mode=render_mode)

        self.prev_target_look_direction = vec3.from_yaw_pitch(0.0, 0.0)

        # Action space: Δyaw, Δpitch
        self.action_space = spaces.Box(
            low=np.array([-MAX_ANGLE_PER_STEP, -MAX_ANGLE_PER_STEP], dtype=np.float32),
            high=np.array([MAX_ANGLE_PER_STEP, MAX_ANGLE_PER_STEP], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        # Apply Δyaw, Δpitch
        dyaw, dpitch = action
        self.agent_input.yaw = angles.yaw_difference(0, self.agent.yaw + dyaw)
        self.agent_input.pitch = np.clip(
            self.agent.pitch + dpitch, -np.pi / 2, np.pi / 2
        )

        # Update movement inputs periodically
        if self.current_step % MOVEMENT_TIMER_INTERVAL == 0:
            self.generate_probabilistic_input(self.agent_input)
            self.generate_probabilistic_input(self.target_input)

            agent_distance_to_target = vec3.distance(
                self.agent.position, self.target.position
            )
            # move away if the agent is too close
            if agent_distance_to_target < 2.0:
                self.target_input.w = False
                self.target_input.s = True
                self.agent_input.w = False
                self.agent_input.s = True

        self.aim_target_at_agent()

        # Simulate physics
        simulate(self.agent, self.agent_input)
        simulate(self.target, self.target_input)

        # Compute reward
        reward = self._compute_reward(action)

        # Check termination
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}

    def _compute_reward(self, action):
        """Reward based on how close the agent is aiming at the target"""
        # Direction to target
        agent_to_target = vec3.subtract(
            vec3.subtract(self.target.position, vec3.from_list([0, 0.3, 0])),
            self.agent.position,
        )
        yaw_to_target, pitch_to_target, distance_to_target = (
            angles.vec_to_yaw_pitch_distance(agent_to_target)
        )

        # Angular error
        yaw_error = abs(angles.yaw_difference(self.agent.yaw, yaw_to_target)) / np.pi
        pitch_error = abs(
            angles.pitch_difference(self.agent.pitch, pitch_to_target)
        ) / (np.pi / 2)

        # Base reward is negative error
        reward = -(yaw_error**2 + pitch_error**2)

        # Penalty for movement
        reward -= 0.3 * (abs(action[0]) + abs(action[1]))

        # Bonus for hit
        # target_aabb_min, target_aabb_max = self.target.get_min_max_aabb()
        # look_dir = vec3.from_yaw_pitch(self.agent.yaw, self.agent.pitch)
        # did_hit, distance_to_intersection = combat.line_intersects_aabb(
        #     self.agent.get_eye_position(),
        #     look_dir,
        #     target_aabb_min,
        #     target_aabb_max,
        # )

        # if did_hit:
        #     reward += 0.5

        return reward

    def render(self):
        """Simple text-based render for debugging"""
        if self.render_mode == "human":
            direction_to_target = vec3.subtract(
                self.target.position, self.agent.position
            )
            target_yaw, target_pitch, distance = angles.vec_to_yaw_pitch_distance(
                direction_to_target
            )

            yaw_error = abs(angles.yaw_difference(self.agent.yaw, target_yaw))
            pitch_error = abs(angles.pitch_difference(self.agent.pitch, target_pitch))

            print(f"Step: {self.current_step}")
            print(
                f"Agent: yaw={np.degrees(self.agent.yaw):.1f}°, pitch={np.degrees(self.agent.pitch):.1f}°"
            )
            print(
                f"Target:  yaw={np.degrees(target_yaw):.1f}°, pitch={np.degrees(target_pitch):.1f}°"
            )
            print(
                f"Error:   yaw={np.degrees(yaw_error):.1f}°, pitch={np.degrees(pitch_error):.1f}°"
            )
            print(f"Distance: {distance:.2f}")
            print("-" * 50)
