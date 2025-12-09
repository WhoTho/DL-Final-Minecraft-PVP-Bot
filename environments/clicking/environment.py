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

        return (
            self._get_observation(randomize_target_aim=True),
            reward,
            terminated,
            truncated,
            {},
        )

    def _calculate_reward(self, clicked, hit):
        """
        Reward structure using AABB intersection:
        - Successful hit: +1.0
        - Wasted click (could have hit but didn't): -0.3
        - Wasted click (couldn't hit - no intersection or invuln): -0.5
        - Opportunity missed (could have hit, didn't click): -0.2
        - Good no-click (couldn't hit, didn't click): +0.05
        """
        reward = 0.0

        # Use raycast to determine if we COULD hit (same logic as try_attack)
        eye_pos = self.agent.get_eye_position()
        look_dir = vec3.from_yaw_pitch(self.agent.yaw, self.agent.pitch)
        aabb_min, aabb_max = self.target.get_min_max_aabb()

        # Check AABB intersection
        intersects_aabb, distance = combat.line_intersects_aabb(
            eye_pos, look_dir, aabb_min, aabb_max
        )

        # Can hit if: intersects AABB, within reach, and not invulnerable
        can_hit = (
            intersects_aabb
            and distance <= self.agent.reach
            and self.target.invulnerablility_ticks <= 0
        )

        # Hit: best outcome
        if hit:
            reward += 1.0
            return reward

        # Clicked but didn't hit
        if clicked:
            if can_hit:
                # Should have hit but didn't (rare edge case, maybe physics timing)
                reward -= 0.3
            else:
                # Wasted click: couldn't have hit anyway
                reward -= 0.5
        else:
            # Didn't click
            if can_hit:
                # Missed opportunity: should have clicked
                reward -= 0.2
            else:
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
