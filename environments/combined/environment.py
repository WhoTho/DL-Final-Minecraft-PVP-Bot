import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from helpers import vec3, angles, world
from simulator.objects import Entity
from simulator.physics import simulate, InputState, GROUND_Y
from simulator import combat
from environments.combined.simple_bot import SimpleBot

# from environments.combined.self_play_agent import SelfPlayAgent
from environments.base_enviroment import BaseEnv, MAX_ANGLE_PER_STEP


class CombinedEnv(BaseEnv):
    def __init__(self, render_mode=None):
        super().__init__(max_steps=400, render_mode=render_mode)

        # Initialize bot AI
        self.bot = SimpleBot()

        # Health tracking for reward calculation
        self.prev_agent_health = 0.0
        self.prev_target_health = 0.0

        # Action space for movement is separate - we'll use Box for continuous action
        # Combined: [w, a, s, d, space, sprint, click, dyaw, dpitch]

        self.action_space = spaces.Box(
            low=np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -MAX_ANGLE_PER_STEP,
                    -MAX_ANGLE_PER_STEP,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    MAX_ANGLE_PER_STEP,
                    MAX_ANGLE_PER_STEP,
                ],
                dtype=np.float32,
            ),
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bot.reset()

        # Initialize health tracking for reward calculation
        self.prev_agent_health = self.agent.health
        self.prev_target_health = self.target.health

        return self._get_observation(), {}

    def _action_to_input(self, action):
        """Convert continuous action to InputState"""
        input_state = InputState()

        input_state.w = bool(action[0] > 0.5)
        input_state.a = bool(action[1] > 0.5)
        input_state.s = bool(action[2] > 0.5)
        input_state.d = bool(action[3] > 0.5)
        input_state.space = bool(action[4] > 0.5)
        input_state.sprint = bool(action[5] > 0.5)

        input_state.click = bool(action[6] > 0.5)

        input_state.yaw = angles.yaw_difference(0, self.agent.yaw + action[7])
        input_state.pitch = np.clip(self.agent.pitch + action[8], -np.pi / 2, np.pi / 2)

        return input_state

    def step(self, action):
        # Convert agent action to input
        self.agent_input = self._action_to_input(action)

        # Update bot AI (bot also attacks)
        self.target_input = self.bot.get_bot_inputs(
            self.current_step, self.target, self.agent, self.np_random
        )

        # Agent attack if clicking
        if self.agent_input.click:
            if combat.try_attack(self.agent, [self.target]):
                self.target.health = max(0.0, self.target.health - 1.0)

        if self.target_input.click:
            if combat.try_attack(self.target, [self.agent]):
                self.agent.health = max(0.0, self.agent.health - 1.0)

        # Simulate physics for both entities
        simulate(self.agent, self.agent_input)
        simulate(self.target, self.target_input)

        # Decrement invulnerability timers
        if self.agent.invulnerablility_ticks > 0:
            self.agent.invulnerablility_ticks -= 1
        if self.target.invulnerablility_ticks > 0:
            self.target.invulnerablility_ticks -= 1

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        # Terminate if either player dies
        if self.agent.health <= 0 or self.target.health <= 0:
            terminated = True
        truncated = False

        return self._get_observation(), reward, terminated, truncated, {}

    def _calculate_reward(self):
        reward = 0.0

        # 1. Reward for damage dealt THIS STEP
        damage_this_step = self.prev_target_health - self.target.health
        reward += damage_this_step * 1.0

        # 2. Penalize damage taken THIS STEP
        damage_taken_this_step = self.prev_agent_health - self.agent.health
        reward -= damage_taken_this_step * 1.0

        # 3. Small reward for ending the fight
        if self.target.health <= 0:
            reward += 50.0  # win bonus

        if self.agent.health <= 0:
            reward -= 50.0  # loss penalty

        # Save for next step
        self.prev_target_health = self.target.health
        self.prev_agent_health = self.agent.health

        return reward

    def render(self):
        """Render environment state"""
        if self.render_mode == "human":
            distance = vec3.distance(self.agent.position, self.target.position)
            agent_speed = vec3.length(self.agent.velocity)

            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Agent Health: {self.agent.health:.1f}/{self.agent.max_health:.1f}")
            print(
                f"Target Health: {self.target.health:.1f}/{self.target.max_health:.1f}"
            )
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
