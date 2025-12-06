import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from helpers import vec3, angles, world
from simulator.objects import Entity
from simulator.physics import simulate, InputState, GROUND_Y
from simulator.combat import apply_knockback


# ==================== ACTION BIT FLAGS ====================
class ActionFlags:
    """Bit flags for target actions"""

    MOVE_FORWARD = 1 << 0
    MOVE_BACKWARD = 1 << 1
    MOVE_LEFT = 1 << 2
    MOVE_RIGHT = 1 << 3
    JUMP = 1 << 4
    KNOCKBACK = 1 << 5
    SPRINT = 1 << 6
    ALL = (
        MOVE_FORWARD
        | MOVE_BACKWARD
        | MOVE_LEFT
        | MOVE_RIGHT
        | JUMP
        | KNOCKBACK
        | SPRINT
    )


class TargetModes:
    """Target behavior modes using bit flag combinations"""

    # Basic modes
    IDLE = 0x000
    MOVE_TOWARD = ActionFlags.MOVE_FORWARD
    MOVE_AWAY = ActionFlags.MOVE_BACKWARD

    # Strafing variants
    MOVE_TOWARD_LEFT = ActionFlags.MOVE_FORWARD | ActionFlags.MOVE_LEFT
    MOVE_TOWARD_RIGHT = ActionFlags.MOVE_FORWARD | ActionFlags.MOVE_RIGHT

    # Everything (all possible actions)
    EVERYTHING = (
        ActionFlags.MOVE_FORWARD
        | ActionFlags.MOVE_BACKWARD
        | ActionFlags.MOVE_LEFT
        | ActionFlags.MOVE_RIGHT
        | ActionFlags.JUMP
        | ActionFlags.KNOCKBACK
        | ActionFlags.SPRINT
    )

    # Mode strings for debugging
    MODE_NAMES = {
        0x000: "IDLE",
        0x001: "MOVE_TOWARD",
        0x002: "MOVE_AWAY",
        0x041: "MOVE_TOWARD_SPRINT",  # Forward + Sprint
        0x042: "MOVE_AWAY_SPRINT",  # Backward + Sprint
    }


MOVEMENT_CYCLE_TIME = 4


class MovementEnv(gym.Env):
    """
    Movement-only environment with curriculum learning.
    Agent controls: W, A, S, D, SPACE, SPRINT
    Agent's yaw and pitch are fixed at (0, 0).
    Target behavior changes based on curriculum stage.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Environment parameters
        self.max_steps = 400
        self.current_step = 0

        # Agent setup (fixed position and orientation)
        self.agent = Entity(
            object_id=1,
            position=vec3.from_list([0.0, GROUND_Y, 0.0]),
            yaw=0.0,  # Fixed forward
            pitch=0.0,  # Fixed level
            color=(0, 255, 0),
        )
        self.agent_input = InputState()

        # Target setup
        self.target = Entity(
            object_id=2,
            position=vec3.from_list([5.0, GROUND_Y, 5.0]),
            yaw=0.0,
            pitch=0.0,
            color=(255, 0, 0),
        )
        self.target_input = InputState()

        # Target behavior state (bit flags)
        self.target_mode = TargetModes.IDLE
        self.target_yaw_velocity = 0.0
        self.target_max_yaw_velocity = math.radians(540)  # 540 degrees per second
        self.target_last_knockback_at = 0
        self.prev_target_look_direction = vec3.from_yaw_pitch(0.0, 0.0)

        # Action space: W, A, S, D, SPACE, SPRINT (6 discrete actions)
        self.action_space = spaces.MultiBinary(6)

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(20,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agent position and state
        self.agent.position = vec3.from_list(
            [
                self.np_random.uniform(-2.0, 2.0),
                GROUND_Y,
                self.np_random.uniform(-2.0, 2.0),
            ]
        )
        self.agent.velocity = vec3.zero()
        self.agent.yaw = np.random.uniform(-math.pi, math.pi)
        self.agent.pitch = np.random.uniform(-math.pi / 12, math.pi / 12)
        self.agent.on_ground = True
        self.agent_input = InputState()

        # Reset target position and state
        angle = self.np_random.uniform(0, 2 * math.pi)
        distance = self.np_random.uniform(6.0, 12.0)
        self.target.position = vec3.from_list(
            [
                self.agent.position[0] + distance * math.cos(angle),
                GROUND_Y,
                self.agent.position[2] + distance * math.sin(angle),
            ]
        )
        self.target.velocity = vec3.zero()
        self.target.on_ground = True
        self.target_input = InputState()
        self.target_last_knockback_at = 0
        self.prev_target_look_direction = vec3.from_yaw_pitch(
            self.target.yaw, self.target.pitch
        )

        # Reset tracking variables
        self.current_step = 0
        self.prev_distance = vec3.distance(self.agent.position, self.target.position)

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

        # Keep fixed orientation
        input_state.yaw = self.agent.yaw
        input_state.pitch = self.agent.pitch

        return input_state

    def _update_target_behavior(self, dt):
        """Update target behavior based on curriculum stage and bit flags"""
        # Update movement keys
        if self.current_step % MOVEMENT_CYCLE_TIME == 0:
            # forward or backward
            if np.random.rand() < 0.7:
                self.target_input.w = True
                self.target_input.s = False
            else:
                self.target_input.w = False
                self.target_input.s = True
            # left or right or neither
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

            # jump
            if np.random.rand() < 0.1:
                self.target_input.space = True
            else:
                self.target_input.space = False

            # sprint
            if np.random.rand() < 0.9:
                self.target_input.sprint = True
            else:
                self.target_input.sprint = False

        # Update knockback
        time_since_last_knockback = self.current_step - self.target_last_knockback_at
        if time_since_last_knockback >= 10:
            if (
                np.random.rand() < 0.1
                and vec3.distance_squared(self.agent.position, self.target.position)
                < 16.0
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
                    is_sprint_hit=np.random.rand() < 0.2,
                )

        # Update target yaw to look at agent (with limited angular velocity)
        agent_direction = vec3.subtract(self.agent.position, self.target.position)
        if vec3.length(agent_direction) > 1e-6:
            target_yaw, _, _ = angles.vec_to_yaw_pitch_distance(agent_direction)
            yaw_diff = angles.yaw_difference(self.target.yaw, target_yaw)

            # aiming in same direction speeds up turning slowly, opposite direction slows down quickly
            if np.sign(yaw_diff) == np.sign(self.target_yaw_velocity):
                self.target_yaw_velocity += yaw_diff * 0.3
            else:
                self.target_yaw_velocity += yaw_diff * 0.6
            self.target_yaw_velocity *= 0.8  # damping

            self.target_yaw_velocity = np.clip(
                self.target_yaw_velocity,
                -self.target_max_yaw_velocity * dt,
                self.target_max_yaw_velocity * dt,
            )
            self.target.yaw += self.target_yaw_velocity

        self.target_input.yaw = self.target.yaw
        self.target_input.pitch = self.target.pitch

    def step(self, action):
        # Convert action to input
        self.agent_input = self._action_to_input(action)

        # Store previous target look direction before update
        self.prev_target_look_direction = vec3.from_yaw_pitch(
            self.target.yaw, self.target.pitch
        )

        # Update target behavior
        dt = 1.0 / 20.0  # 20 TPS
        self._update_target_behavior(dt)

        # Simulate physics for both entities
        simulate(self.agent, self.agent_input)
        simulate(self.target, self.target_input.clone())  # prevent mutation issues

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

        # 1. Reward getting closer, punish getting farther
        # Distance change is noramlly between 0 and 0.2
        reward = (prev_distance - current_distance) * 1.5

        # 2. Tiny penalty for doing nothing
        if not any(current_action[:4]):  # no WASD
            reward -= 0.01

        # 3. Reward being within optimal distance band
        optimal_distance_min = 2.0
        optimal_distance_max = 3.5
        if optimal_distance_min <= current_distance <= optimal_distance_max:
            reward += 0.5
        elif current_distance < optimal_distance_min:
            reward -= 0.5 * (optimal_distance_min - current_distance)
        else:  # current_distance > optimal_distance_max
            reward -= 0.5 * (current_distance - optimal_distance_max) * 0.2

        # 4. Reward not being directly in targets aim if far away if within optimal distance
        if optimal_distance_min <= current_distance <= optimal_distance_max:
            target_to_agent = vec3.subtract(self.agent.position, self.target.position)
            target_yaw_to_agent, _, _ = angles.vec_to_yaw_pitch_distance(
                target_to_agent
            )
            yaw_diff = angles.yaw_difference(self.target.yaw, target_yaw_to_agent)
            reward += 0.05 * (
                abs(yaw_diff) / math.pi
            )  # more reward for being outside aim cone

        # update previous distance
        self.prev_distance = current_distance
        return reward
        # reward = 0.0
        # current_distance = vec3.distance_xz(self.agent.position, self.target.position)
        # # distance band
        # if self.optimal_distance_min <= current_distance <= self.optimal_distance_max:
        #     reward += self.reward_weights["distance_band"]
        # else:
        #     distance_error = min(
        #         abs(current_distance - self.optimal_distance_min),
        #         abs(current_distance - self.optimal_distance_max),
        #     )
        #     reward -= self.reward_weights["distance_band"] * distance_error

        # dodging
        # target_to_agent = vec3.subtract(self.agent.position, self.target.position)
        # target_yaw_to_agent, _, _ = angles.vec_to_yaw_pitch_distance(target_to_agent)
        # yaw_diff = angles.yaw_difference(self.target.yaw, target_yaw_to_agent)
        # cone_radius = math.radians(30)  # 30 degree cone
        # if abs(yaw_diff) < cone_radius:  # Within 30 degree cone
        #     # Reward dodging when in aim cone
        #     moving_laterally = (current_action[1] and not current_action[3]) or (
        #         not current_action[1] and current_action[3]
        #     )  # A xor D pressed
        #     if moving_laterally:
        #         reward += self.reward_weights["dodge"]
        # else:
        #     reward += self.reward_weights["dodge"] * 3

        return reward

        # reward = 0.0
        # current_distance = vec3.distance_xz(self.agent.position, self.target.position)

        # # Stage 1 rewards
        # if self.curriculum_stage >= 1:
        #     if current_distance < 4.0:
        #         reward += self.reward_weights["move_toward"]
        #     # Move toward target
        #     distance_change = self.prev_distance - current_distance
        #     reward += self.reward_weights["move_toward"] * distance_change

        #     # small pentalty for not pressing any movement keys
        #     if not any(current_action[:4]):  # W, A, S, D
        #         reward -= self.reward_weights["no_movement"]

        # # Stage 2 rewards
        # if self.curriculum_stage >= 2:
        #     # Distance band reward
        #     if (
        #         self.optimal_distance_min
        #         <= current_distance
        #         <= self.optimal_distance_max
        #     ):
        #         reward += self.reward_weights["distance_band"]
        #     else:
        #         distance_error = min(
        #             abs(current_distance - self.optimal_distance_min),
        #             abs(current_distance - self.optimal_distance_max),
        #         )
        #         reward -= self.reward_weights["distance_band"] * 0.1 * distance_error

        #     # Strafing reward (lateral movement)
        #     if (current_action[1] and not current_action[3]) or (
        #         not current_action[1] and current_action[3]
        #     ):  # A xor D pressed
        #         reward += self.reward_weights["strafing"]

        # # Stage 3 rewards
        # if self.curriculum_stage >= 3:
        #     # Penalty for being in target's aim cone
        #     target_to_agent = vec3.subtract(self.agent.position, self.target.position)
        #     target_yaw_to_agent, _, _ = angles.vec_to_yaw_pitch_distance(
        #         target_to_agent
        #     )
        #     yaw_diff = angles.yaw_difference(self.target.yaw, target_yaw_to_agent)
        #     cone_radius = math.radians(30)  # 30 degree cone
        #     if abs(yaw_diff) < cone_radius:  # Within 30 degree cone
        #         reward -= self.reward_weights["aim_penalty"] * (
        #             (cone_radius - abs(yaw_diff)) / cone_radius
        #         )

        #         # Reward dodging when in aim cone
        #         moving_laterally = (current_action[1] and not current_action[3]) or (
        #             not current_action[1] and current_action[3]
        #         )  # A xor D pressed
        #         if moving_laterally:
        #             reward += self.reward_weights["dodge"]

        # # update previous distance for next step
        # self.prev_distance = current_distance

        # return reward

    def _get_observation(self):
        """Get current observation state in agent's local coordinate frame"""
        # Create rotation matrix to transform world vectors to agent's local frame
        # Agent's forward direction
        forward, right, up = world.yaw_pitch_to_basis_vectors(
            self.agent.yaw, self.agent.pitch
        )

        # Direction to target
        to_target_world = vec3.subtract(self.target.position, self.agent.position)
        to_target_local_dir, distance_to_target = vec3.direction_and_length(
            to_target_world
        )
        to_target_local = world.world_to_local(to_target_local_dir, forward, right, up)

        # Agent velocity
        agent_vel_dir, agent_speed = vec3.direction_and_length(self.agent.velocity)
        if agent_speed > 1e-6:
            agent_vel_local = world.world_to_local(agent_vel_dir, forward, right, up)
        else:
            agent_vel_local = vec3.zero()

        # Target velocity
        target_vel_dir, target_speed = vec3.direction_and_length(self.target.velocity)
        if target_speed > 1e-6:
            target_vel_local = world.world_to_local(target_vel_dir, forward, right, up)
        else:
            target_vel_local = vec3.zero()

        # Target look direction
        target_look = vec3.from_yaw_pitch(self.target.yaw, self.target.pitch)
        target_look_local = world.world_to_local(target_look, forward, right, up)

        # Previous target look direction
        prev_target_look_local = world.world_to_local(
            self.prev_target_look_direction, forward, right, up
        )

        # Binary flags
        agent_on_ground = 1.0 if self.agent.on_ground else -1.0
        target_on_ground = 1.0 if self.target.on_ground else -1.0

        # Assemble observation
        obs = np.array(
            [
                # Direction to target (local frame)
                to_target_local[0],
                to_target_local[1],
                to_target_local[2],
                # Distance to target (normalized)
                distance_to_target / 10.0,
                # Agent velocity direction (local frame)
                agent_vel_local[0],
                agent_vel_local[1],
                agent_vel_local[2],
                # Agent speed (normalized)
                agent_speed / 2.0,
                # Target velocity direction (local frame)
                target_vel_local[0],
                target_vel_local[1],
                target_vel_local[2],
                # Target speed (normalized)
                target_speed / 2.0,
                # Ground contact flags
                agent_on_ground,
                target_on_ground,
                # Target look direction (local frame)
                target_look_local[0],
                target_look_local[1],
                target_look_local[2],
                # Previous target look direction (local frame)
                prev_target_look_local[0],
                prev_target_look_local[1],
                prev_target_look_local[2],
            ],
            dtype=np.float32,
        )

        return np.clip(obs, -1.0, 1.0)

    def render(self):
        """Render environment state"""
        if self.render_mode == "human":
            distance = vec3.distance(self.agent.position, self.target.position)
            agent_speed = vec3.length(self.agent.velocity)

            # Convert flags to human-readable format
            mode_name = TargetModes.MODE_NAMES.get(
                self.target_mode, f"MODE_{self.target_mode:03x}"
            )

            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Target Mode: {mode_name}")
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

    def get_info(self):
        """Get additional environment information"""
        mode_name = TargetModes.MODE_NAMES.get(
            self.target_mode, f"MODE_{self.target_mode:03x}"
        )
        return {
            "target_mode": mode_name,
            "target_mode_flags": hex(self.target_mode),
            "distance": vec3.distance(self.agent.position, self.target.position),
            "agent_speed": vec3.length(self.agent.velocity),
        }
