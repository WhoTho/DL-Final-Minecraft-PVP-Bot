import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from helpers import vec3, angles, world
from simulator.objects import Entity
from simulator.physics import simulate, InputState, GROUND_Y
from simulator.combat import apply_knockback
from environments.base_enviroment import (
    BaseEnv,
    MAX_ANGLE_PER_STEP,
)

MOVEMENT_TIMER_INTERVAL = 4  # Change movement input every 4 steps


class MovementEnv(BaseEnv):
    def __init__(self, render_mode=None):
        super().__init__(max_steps=400, render_mode=render_mode)

        self.prev_distance = 0.0
        self.agent_yaw_velocity = 0.0
        self.target_yaw_velocity = 0.0
        self.target_last_knockback_at = 0

        # Action space: W, A, S, D, SPACE, SPRINT (6 discrete actions)
        self.action_space = spaces.MultiBinary(6)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset tracking variables
        self.prev_distance = vec3.distance(self.agent.position, self.target.position)
        self.agent_yaw_velocity = 0.0
        self.target_yaw_velocity = 0.0
        self.target_last_knockback_at = 0

        return self._get_observation(), {}

    def _action_to_input(self, action):
        """Convert discrete action to InputState"""
        input_state = InputState()

        if action[0]:
            input_state.w = True
        if action[1]:
            input_state.a = True
        if action[2]:
            input_state.s = True
        if action[3]:
            input_state.d = True
        if action[4]:
            input_state.space = True
        if action[5]:
            input_state.sprint = True

        # Update agent yaw to look at target (with limited angular velocity)
        agent_to_target = vec3.subtract(self.target.position, self.agent.position)
        if vec3.length(agent_to_target) > 1e-6:
            target_yaw, target_pitch, _ = angles.vec_to_yaw_pitch_distance(
                agent_to_target
            )
            yaw_diff = angles.yaw_difference(self.agent.yaw, target_yaw)

            # aiming in same direction speeds up turning slowly, opposite direction slows down quickly
            if np.sign(yaw_diff) == np.sign(self.agent_yaw_velocity):
                self.agent_yaw_velocity += yaw_diff * 0.2
            else:
                self.agent_yaw_velocity += yaw_diff * 0.6
            self.agent_yaw_velocity *= 0.8  # damping

            self.agent_yaw_velocity = np.clip(
                self.agent_yaw_velocity,
                -MAX_ANGLE_PER_STEP,
                MAX_ANGLE_PER_STEP,
            )
            input_state.yaw = angles.yaw_difference(
                0, self.agent.yaw + self.agent_yaw_velocity
            )
            input_state.pitch = target_pitch
        else:
            input_state.yaw = self.agent.yaw
            input_state.pitch = self.agent.pitch

        return input_state

    def _update_target_behavior(self):
        """Update target behavior based on curriculum stage and bit flags"""
        # Update movement keys
        if self.current_step % MOVEMENT_TIMER_INTERVAL == 0:
            self.generate_probabilistic_input(self.target_input)

        # Update knockback
        time_since_last_knockback = self.current_step - self.target_last_knockback_at
        if time_since_last_knockback >= 10:
            if (
                self.np_random.random() < 0.1
                and vec3.distance(self.agent.position, self.target.position) < 4
            ):  # 10% chance to apply knockback if within 4 blocks
                self.target_last_knockback_at = self.current_step
                # Apply knockback force
                agent_to_target = vec3.subtract(
                    self.target.position, self.agent.position
                )
                apply_knockback(
                    self.target,
                    agent_to_target[0],
                    agent_to_target[2],
                    is_sprint_hit=self.np_random.random() < 0.2,
                )

        # Update target yaw to look at agent (with limited angular velocity)
        agent_direction = vec3.subtract(self.agent.position, self.target.position)
        if vec3.length(agent_direction) > 1e-6:
            target_yaw, _, _ = angles.vec_to_yaw_pitch_distance(agent_direction)
            yaw_diff = angles.yaw_difference(self.target.yaw, target_yaw)

            # aiming in same direction speeds up turning slowly, opposite direction slows down quickly
            if np.sign(yaw_diff) == np.sign(self.target_yaw_velocity):
                self.target_yaw_velocity += yaw_diff * 0.2
            else:
                self.target_yaw_velocity += yaw_diff * 0.4
            self.target_yaw_velocity *= 0.9  # damping

            self.target_yaw_velocity = np.clip(
                self.target_yaw_velocity,
                -MAX_ANGLE_PER_STEP,
                MAX_ANGLE_PER_STEP,
            )
            self.target.yaw += self.target_yaw_velocity

        self.target_input.yaw = self.target.yaw
        self.target_input.pitch = self.target.pitch

    def step(self, action):
        # Convert action to input
        self.agent_input = self._action_to_input(action)

        # Update target behavior
        self._update_target_behavior()

        # Simulate physics for both entities
        simulate(self.agent, self.agent_input)
        simulate(self.target, self.target_input)

        # Calculate reward
        reward = self._calculate_reward(action)

        # Check termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        return self._get_observation(), reward, terminated, truncated, {}

    def _calculate_reward(self, current_action):
        """Calculate reward based on curriculum stage"""

        prev_distance = self.prev_distance
        current_distance = vec3.distance_xz(self.agent.position, self.target.position)

        # Reward getting closer, punish getting farther
        # Distance change is noramlly between 0 and 0.2
        reward = (prev_distance - current_distance) * 1.5

        # Tiny penalty for doing nothing
        if not any(current_action[:4]):  # no WASD
            reward -= 0.01

        # Tiny pentaly for jumping
        if current_action[4]:  # space
            reward -= 0.05

        # Reward being within optimal distance band
        optimal_distance_min = 2.0
        optimal_distance_max = 3.5
        if optimal_distance_min <= current_distance <= optimal_distance_max:
            reward += 0.5
        elif current_distance < optimal_distance_min:
            reward -= 0.5 * (optimal_distance_min - current_distance)
            if current_distance < 1:
                reward -= 0.5  # extra penalty for being too close
        else:  # current_distance > optimal_distance_max
            reward -= 0.5 * (current_distance - optimal_distance_max) * 0.2

        # Reward not being directly in targets aim if far away if within optimal distance
        if optimal_distance_min <= current_distance <= optimal_distance_max:
            target_to_agent = vec3.subtract(self.agent.position, self.target.position)
            target_yaw_to_agent, _, _ = angles.vec_to_yaw_pitch_distance(
                target_to_agent
            )
            yaw_diff = angles.yaw_difference(self.target.yaw, target_yaw_to_agent)
            reward += 0.02 * (
                abs(yaw_diff) / math.pi
            )  # more reward for being outside aim cone

        # update previous distance
        self.prev_distance = current_distance
        return reward

    def render(self):
        """Render environment state"""
        if self.render_mode == "human":
            distance = vec3.distance(self.agent.position, self.target.position)
            agent_speed = vec3.length(self.agent.velocity)

            print(f"Step: {self.current_step}/{self.max_steps}")
            print(
                f"Agent Position: ({self.agent.position[0]:.2f}, {self.agent.position[2]:.2f})"
            )
            print(
                f"Target Position: ({self.target.position[0]:.2f}, {self.target.position[2]:.2f})"
            )
            print(f"Distance: {distance:.2f}")
            print(f"Agent Speed: {agent_speed:.2f}")
            print(f"Agent On Ground: {self.agent.on_ground}")
            print("-" * 50)
