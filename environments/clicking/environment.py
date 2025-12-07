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


class ClickingEnv(BaseEnv):
    def __init__(self, render_mode=None):
        super().__init__(max_steps=600, render_mode=render_mode)

        # Tracking
        self.last_hit_tick = -1000  # When we last successfully hit the target
        self.target_health_on_step_start = self.target.health
        self.total_hits = 0
        self.total_wasted_clicks = 0

        # Action: Binary click (0 or 1)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset tracking
        self.current_step = 0
        self.last_hit_tick = -1000
        self.target_health_on_step_start = self.target.health
        self.total_hits = 0
        self.total_wasted_clicks = 0

        return self._get_observation(), {}

    def step(self, action):
        # action: 0 = don't click, 1 = click
        clicked = action == 1

        # Record target health before step to detect hits
        self.target_health_on_step_start = self.target.health

        # Update target movement
        if self.current_step % MOVEMENT_TIMER_INTERVAL == 0:
            self.generate_probabilistic_input(self.target_input)
            self.generate_probabilistic_input(self.agent_input)

        # Update target orientation
        self.aim_target_at_agent()

        # update agent orientation
        agent_to_target = vec3.subtract(self.target.position, self.agent.position)
        agent_yaw, agent_pitch, _ = angles.vec_to_yaw_pitch_distance(agent_to_target)
        self.agent_input.yaw = agent_yaw
        self.agent_input.pitch = agent_pitch

        # Simulate physics
        simulate(self.agent, self.agent_input)
        simulate(self.target, self.target_input)

        # Perform attack if agent clicks
        did_hit = False
        if clicked:
            did_hit = combat.try_attack(self.agent, [self.target])
            if did_hit:
                self.last_hit_tick = self.current_step
                self.total_hits += 1
            else:
                self.total_wasted_clicks += 1

        # Decrement target invulnerability
        if self.target.invulnerablility_ticks > 0:
            self.target.invulnerablility_ticks -= 1

        # Calculate reward
        reward = self._calculate_reward(clicked, did_hit)

        # Check termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        return self._get_observation(), reward, terminated, truncated, {}

    def _calculate_reward(self, clicked, hit):
        """
        Reward structure:
        - Successful hit: +1.0 (base) + bonus for good aim
        - Wasted click (in range but missed): -0.3
        - Wasted click (out of range): -0.5
        - Good no-click (out of range, didn't click): +0.1
        - Opportunity missed (in range + good aim, didn't click): -0.2
        """
        reward = 0.0

        # Get target info
        agent_to_target = vec3.subtract(self.target.position, self.agent.position)
        distance = vec3.length(agent_to_target)
        target_yaw, target_pitch, _ = angles.vec_to_yaw_pitch_distance(agent_to_target)
        yaw_error = abs(angles.yaw_difference(self.agent.yaw, target_yaw))
        pitch_error = abs(angles.pitch_difference(self.agent.pitch, target_pitch))

        # Thresholds
        aim_threshold_yaw = np.radians(15)  # 15 degree tolerance
        aim_threshold_pitch = np.radians(15)
        range_threshold = 3.0  # Max attack range (3 blocks)

        in_range = distance <= range_threshold
        good_aim = yaw_error <= aim_threshold_yaw and pitch_error <= aim_threshold_pitch
        invuln_ready = self.target.invulnerablility_ticks <= 0

        # Hit: best outcome
        if hit:
            reward += 1.0
            # Bonus for hitting with excellent aim
            if yaw_error < np.radians(5) and pitch_error < np.radians(5):
                reward += 0.5
            return reward

        # Clicked but didn't hit
        if clicked:
            if not in_range:
                # Wasted click: out of range
                reward -= 0.5
            elif invuln_ready and good_aim:
                # Hit should have happened - penalty for failure (might be rare due to physics)
                reward -= 0.2
            else:
                # Wasted click: in range but bad aim or still invuln
                reward -= 0.3
        else:
            # Didn't click
            if in_range and good_aim and invuln_ready:
                # Missed opportunity: could have hit
                reward -= 0.2
            elif not in_range:
                # Correctly didn't waste a click
                reward += 0.05

        return reward

    def render(self):
        """Text-based rendering for debugging"""
        if self.render_mode == "human":
            agent_to_target = vec3.subtract(self.target.position, self.agent.position)
            distance = vec3.length(agent_to_target)
            target_yaw, target_pitch, _ = angles.vec_to_yaw_pitch_distance(
                agent_to_target
            )

            yaw_error = abs(angles.yaw_difference(self.agent.yaw, target_yaw))
            pitch_error = abs(angles.pitch_difference(self.agent.pitch, target_pitch))

            print(f"Step: {self.current_step}/{self.max_steps}")
            print(
                f"Hits: {self.total_hits} | Wasted Clicks: {self.total_wasted_clicks}"
            )
            print(
                f"Agent: yaw={np.degrees(self.agent.yaw):.1f}°, pitch={np.degrees(self.agent.pitch):.1f}°"
            )
            print(
                f"Target: yaw={np.degrees(target_yaw):.1f}°, pitch={np.degrees(target_pitch):.1f}°"
            )
            print(
                f"Aim Error: yaw={np.degrees(yaw_error):.1f}°, pitch={np.degrees(pitch_error):.1f}°"
            )
            print(
                f"Distance: {distance:.2f} | Invuln Ticks: {self.target.invulnerablility_ticks}"
            )
            print("-" * 60)
