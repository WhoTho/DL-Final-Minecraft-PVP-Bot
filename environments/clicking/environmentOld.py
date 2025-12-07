import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from helpers import vec3, angles, world
from simulator.objects import Entity
from simulator.physics import simulate, InputState, GROUND_Y
from simulator import combat

MAX_DEGREES_PER_STEP = 360 / 0.5 / 20  # 360 degrees in 0.5s at 20 steps/s

# Combat constants
INVULNERABILITY_TICKS = 10  # Ticks before can hit again
ATTACK_COOLDOWN = 5  # Optimal frames between attacks for DPS


class ClickingEnv(gym.Env):
    """
    Agent learns when to click (attack) a moving target.

    Key mechanics:
    - Clicking wastes time (slowdown + can't sprint)
    - Invulnerability cooldown: can't hit same target twice within 10 ticks
    - Agent must learn to click only when:
      1. Aim is good (yaw/pitch error < threshold)
      2. Target is in range (< 3 blocks)
      3. Invulnerability cooldown has passed

    Action: Binary click (0=no click, 1=click)
    Observation: Target position/distance, aim error, invulnerability status
    Reward: Successful hits (with aim bonus), penalties for wasted clicks
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Agent entity (stationary, just aims and clicks)
        self.agent = Entity(
            object_id=1,
            position=vec3.from_list([0.0, GROUND_Y, 0.0]),
            yaw=0.0,
            pitch=0.0,
            color=(0, 255, 0),
        )
        self.agent_input = InputState()

        # Target entity (moves randomly, tries to dodge)
        self.target = Entity(
            object_id=2,
            position=vec3.from_list([2.0, GROUND_Y, 2.0]),
            yaw=0.0,
            pitch=0.0,
            health=100.0,
            max_health=100.0,
            color=(255, 0, 0),
        )
        self.target_input = InputState()

        # Tracking
        self.last_hit_tick = -1000  # When we last successfully hit the target
        self.target_health_on_step_start = self.target.health
        self.total_hits = 0
        self.total_wasted_clicks = 0

        # Action: Binary click (0 or 1)
        self.action_space = spaces.Discrete(2)

        # Observation: (yaw_diff, pitch_diff, distance, in_range, invuln_status, target_pos_x, target_pos_z, agent_yaw, agent_pitch)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(9,), dtype=np.float32
        )

        # Episode parameters
        self.max_steps = 600  # 30 seconds at 20 FPS
        self.current_step = 0
        self.target_movement_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agent (stay at origin, look ahead)
        self.agent.position = vec3.from_list([0.0, GROUND_Y, 0.0])
        self.agent.yaw = 0.0
        self.agent.pitch = 0.0
        self.agent_input = InputState()

        # Reset target (start in front, moving randomly)
        angle = self.np_random.uniform(-np.pi / 4, np.pi / 4)  # In front of agent
        distance = self.np_random.uniform(2.0, 3.5)
        self.target.position = vec3.from_list(
            [
                distance * math.cos(angle),
                GROUND_Y,
                distance * math.sin(angle),
            ]
        )
        self.target.velocity = vec3.zero()
        self.target.yaw = self.np_random.uniform(-np.pi, np.pi)
        self.target.pitch = 0.0
        self.target.health = 100.0
        self.target_input = InputState()

        # Reset tracking
        self.current_step = 0
        self.last_hit_tick = -1000
        self.target_health_on_step_start = self.target.health
        self.total_hits = 0
        self.total_wasted_clicks = 0
        self.target_movement_timer = 0

        return self._get_observation(), {}

    def step(self, action):
        # action: 0 = don't click, 1 = click
        should_click = action == 1

        # Record target health before step to detect hits
        self.target_health_on_step_start = self.target.health

        # Update target movement every 5 ticks (evasive movement)
        if self.target_movement_timer % 5 == 0:
            # Evasive movement: strafe side to side, maybe back up
            if np.random.rand() < 0.8:
                # Move forward toward agent
                self.target_input.w = True
                self.target_input.s = False
            else:
                # Back up
                self.target_input.w = False
                self.target_input.s = True

            # Strafe to evade
            strafe_choice = self.np_random.choice(
                ["left", "right", "none"], p=[0.4, 0.4, 0.2]
            )
            if strafe_choice == "left":
                self.target_input.a = True
                self.target_input.d = False
            elif strafe_choice == "right":
                self.target_input.a = False
                self.target_input.d = True
            else:
                self.target_input.a = False
                self.target_input.d = False

            self.target_input.sprint = np.random.rand() < 0.5
            self.target_input.space = np.random.rand() < 0.1

        self.target_movement_timer += 1

        # Update target orientation
        target_to_agent = vec3.subtract(self.agent.position, self.target.position)
        target_yaw, target_pitch, _ = angles.vec_to_yaw_pitch_distance(target_to_agent)
        self.target.yaw = target_yaw
        self.target.pitch = target_pitch
        self.target_input.yaw = self.target.yaw
        self.target_input.pitch = self.target.pitch

        # update agent orientation
        agent_to_target = vec3.subtract(self.target.position, self.agent.position)
        agent_yaw, agent_pitch, _ = angles.vec_to_yaw_pitch_distance(agent_to_target)
        self.agent.yaw = agent_yaw
        self.agent.pitch = agent_pitch
        self.agent_input.yaw = self.agent.yaw
        self.agent_input.pitch = self.agent.pitch

        # Simulate target physics
        simulate(self.target, self.target_input.clone())

        # Perform attack if agent clicks
        did_hit = False
        if should_click:
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
        reward = self._calculate_reward(should_click, did_hit)

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

    def _get_observation(self):
        """
        Observation includes:
        1. Aim error (yaw diff, pitch diff)
        2. Distance to target (normalized)
        3. In range flag
        4. Invulnerability status
        5. Target relative position
        6. Agent orientation
        """
        agent_to_target = vec3.subtract(self.target.position, self.agent.position)
        distance = vec3.length(agent_to_target)
        target_yaw, target_pitch, _ = angles.vec_to_yaw_pitch_distance(agent_to_target)

        yaw_error = angles.yaw_difference(self.agent.yaw, target_yaw)
        pitch_error = angles.pitch_difference(self.agent.pitch, target_pitch)

        # Normalize errors to [-1, 1]
        yaw_error_norm = np.clip(yaw_error / np.pi, -1.0, 1.0)
        pitch_error_norm = np.clip(pitch_error / (np.pi / 2), -1.0, 1.0)
        distance_norm = np.clip(distance / 10.0, 0.0, 1.0)

        # In range and invulnerability status
        in_range = 1.0 if distance <= 3.0 else -1.0
        invuln_status = np.clip(self.target.invulnerablility_ticks / 10.0, 0.0, 1.0)

        # Target relative position (in world frame)
        target_pos_x = np.clip(self.target.position[0] / 10.0, -1.0, 1.0)
        target_pos_z = np.clip(self.target.position[2] / 10.0, -1.0, 1.0)

        # Agent orientation
        agent_yaw_norm = self.agent.yaw / np.pi
        agent_pitch_norm = self.agent.pitch / (np.pi / 2)

        obs = np.array(
            [
                yaw_error_norm,
                pitch_error_norm,
                distance_norm,
                in_range,
                invuln_status,
                target_pos_x,
                target_pos_z,
                agent_yaw_norm,
                agent_pitch_norm,
            ],
            dtype=np.float32,
        )

        return np.clip(obs, -1.0, 1.0)

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
