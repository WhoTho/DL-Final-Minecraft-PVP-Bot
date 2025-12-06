import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from helpers import vec3, angles
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


class MovementEnv(gym.Env):
    """
    Movement-only environment with curriculum learning.
    Agent controls: W, A, S, D, SPACE, SPRINT
    Agent's yaw and pitch are fixed at (0, 0).
    Target behavior changes based on curriculum stage.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None, curriculum_stage=1, episode_number=0):
        super().__init__()

        self.render_mode = render_mode
        self.curriculum_stage = curriculum_stage
        self.episode_number = episode_number

        # Environment parameters
        self.max_steps = 400  # 20 seconds at 20 TPS
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
        self.target_mode_timer = 0  # Counter for 4-tick update cycle
        self.target_yaw_velocity = 0.0
        self.target_max_yaw_velocity = math.radians(540)  # 540 degrees per second

        # Action space: W, A, S, D, SPACE, SPRINT (6 discrete actions)
        self.action_space = spaces.MultiBinary(6)

        # Observation space: [agent_x, agent_z, agent_vel_x, agent_vel_z,
        #                     target_x, target_z, target_vel_x, target_vel_z,
        #                     distance, relative_angle]
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(14,), dtype=np.float32
        )

        # Curriculum parameters
        # self.stage_transitions = {
        #     1: 50,  # Stage 1 -> 2 at episode 100
        #     2: 100,  # Stage 2 -> 3 at episode 200
        # }
        self.stage_transitions = {
            1: 0,  # Stage 1 -> 2 at episode 100
            2: 0,  # Stage 2 -> 3 at episode 200
        }

        # Reward weights
        self.reward_weights = {
            # Stage 1
            "move_toward": 1.0,
            "no_movement": 0.1,
            # Stage 2
            "distance_band": 3.0,
            "strafing": 0.5,
            # Stage 3
            "aim_penalty": 1.5,
            "dodge": 1.0,
        }

        # Distance parameters
        self.optimal_distance_min = 1.5
        self.optimal_distance_max = 3.5

    def _update_curriculum_stage(self):
        """Update curriculum stage based on episode number"""
        if self.episode_number < self.stage_transitions[1]:
            self.curriculum_stage = 1
        elif self.episode_number < self.stage_transitions[2]:
            self.curriculum_stage = 2
        else:
            self.curriculum_stage = 3

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._update_curriculum_stage()

        # Reset agent position and state
        self.agent.position = vec3.from_list(
            [
                self.np_random.uniform(-2.0, 2.0),
                GROUND_Y,
                self.np_random.uniform(-2.0, 2.0),
            ]
        )
        self.agent.velocity = vec3.zero()
        self.agent.yaw = 0.0  # Fixed forward
        self.agent.pitch = 0.0  # Fixed level
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

        # Reset target behavior
        self.target_mode = TargetModes.IDLE
        self.target_mode_timer = 0  # Reset 4 tick counter

        # Reset tracking variables
        self.current_step = 0
        self.prev_distance = vec3.distance(self.agent.position, self.target.position)
        self.prev_agent_velocity = vec3.zero()
        self.prev_velocity_direction = vec3.zero()

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
        input_state.yaw = 0.0
        input_state.pitch = 0.0

        return input_state

    def _update_target_behavior(self, dt):
        """Update target behavior based on curriculum stage and bit flags"""
        # Update mode timer - changes input state every 4 ticks (0.15 seconds at 20 TPS)
        self.target_mode_timer += 1

        if self.target_mode_timer >= 4:  # Update every 4 ticks
            self._randomize_target_actions()
            self.target_mode_timer = 0

        # Update target yaw to look at agent (with limited angular velocity)
        agent_direction = vec3.subtract(self.agent.position, self.target.position)
        if vec3.length(agent_direction) > 1e-6:
            target_yaw, _, _ = angles.vec_to_yaw_pitch_distance(agent_direction)
            yaw_diff = angles.yaw_difference(self.target.yaw, target_yaw)

            # aiming in same direction speeds up turning slowly, opposite direction slows down quickly
            if np.sign(yaw_diff) == np.sign(self.target_yaw_velocity):
                self.target_yaw_velocity += yaw_diff * 0.3
            else:
                self.target_yaw_velocity += yaw_diff * 0.8
            self.target_yaw_velocity *= 0.90
            self.target_yaw_velocity = np.clip(
                self.target_yaw_velocity,
                -self.target_max_yaw_velocity * dt,
                self.target_max_yaw_velocity * dt,
            )
            self.target.yaw += self.target_yaw_velocity

        # Execute target behavior based on current action flags
        self._execute_target_actions()

    def _get_allowed_action_flags(self):
        """Get action flags allowed in current curriculum stage"""
        if self.curriculum_stage == 1:
            # Stage 1: Basic movement only (forward/backward)
            return ActionFlags.MOVE_FORWARD | ActionFlags.MOVE_BACKWARD
        elif self.curriculum_stage == 2:
            # Stage 2: Movement + strafing
            return (
                ActionFlags.MOVE_FORWARD
                | ActionFlags.MOVE_BACKWARD
                | ActionFlags.MOVE_LEFT
                | ActionFlags.MOVE_RIGHT
            )
        else:  # Stage 3
            # Stage 3: All actions enabled
            return (
                ActionFlags.MOVE_FORWARD
                | ActionFlags.MOVE_BACKWARD
                | ActionFlags.MOVE_LEFT
                | ActionFlags.MOVE_RIGHT
                | ActionFlags.JUMP
                | ActionFlags.KNOCKBACK
                | ActionFlags.SPRINT
            )

    def _randomize_target_actions(self):
        """Randomly select exclusive target actions for the next 4 ticks"""
        allowed_flags = self._get_allowed_action_flags()
        self.target_mode = 0x000  # Start with no actions

        # Movement choice (mutually exclusive)
        movement_options = []
        if allowed_flags & ActionFlags.MOVE_FORWARD:
            movement_options.append(ActionFlags.MOVE_FORWARD)
        if allowed_flags & ActionFlags.MOVE_BACKWARD:
            movement_options.append(ActionFlags.MOVE_BACKWARD)

        if movement_options:
            self.target_mode |= self.np_random.choice(movement_options)

        # Strafing choice (mutually exclusive - left OR right OR neither)
        strafe_options = [0x000]  # Include option for no strafing
        if allowed_flags & ActionFlags.MOVE_LEFT:
            strafe_options.append(ActionFlags.MOVE_LEFT)
        if allowed_flags & ActionFlags.MOVE_RIGHT:
            strafe_options.append(ActionFlags.MOVE_RIGHT)

        self.target_mode |= self.np_random.choice(strafe_options)

        # Jump choice (on/off)
        if allowed_flags & ActionFlags.JUMP:
            if self.np_random.random() < 0.1:  # 10% chance to jump
                self.target_mode |= ActionFlags.JUMP

        # Knockback choice (on/off)
        if allowed_flags & ActionFlags.KNOCKBACK:
            if self.np_random.random() < 0.1:  # 10% chance to knockback
                self.target_mode |= ActionFlags.KNOCKBACK

        # Sprint choice (on/off)
        if allowed_flags & ActionFlags.SPRINT:
            if self.np_random.random() < 0.9:  # 90% chance to sprint
                self.target_mode |= ActionFlags.SPRINT

    def _execute_target_actions(self):
        """Execute target behavior based on current action flags"""
        self.target_input = InputState()

        # Check each action flag and apply corresponding input
        if self.target_mode & ActionFlags.MOVE_FORWARD:
            self.target_input.w = True
        if self.target_mode & ActionFlags.MOVE_BACKWARD:
            self.target_input.s = True
        if self.target_mode & ActionFlags.MOVE_LEFT:
            self.target_input.a = True
        if self.target_mode & ActionFlags.MOVE_RIGHT:
            self.target_input.d = True
        if self.target_mode & ActionFlags.JUMP:
            if self.target.on_ground:
                self.target_input.space = True
        if (
            self.target_mode & ActionFlags.KNOCKBACK and self.target_mode_timer == 0
        ):  # Only apply once at start of a cycle
            # Apply knockback force
            agent_to_target = vec3.subtract(self.target.position, self.agent.position)
            apply_knockback(
                self.target,
                agent_to_target[0],
                agent_to_target[2],
                is_sprint_hit=np.random.rand() < 0.2,
            )
        if self.target_mode & ActionFlags.SPRINT:
            self.target_input.sprint = True

        # Set target orientation
        self.target_input.yaw = self.target.yaw
        self.target_input.pitch = self.target.pitch

    def step(self, action):
        # Convert action to input
        self.agent_input = self._action_to_input(action)
        self.agent_input.yaw = 0.0  # Keep fixed orientation
        self.agent_input.pitch = 0.0

        # Update target behavior
        dt = 1.0 / 20.0  # 20 TPS
        self._update_target_behavior(dt)

        # Store previous state for reward calculation
        prev_agent_pos = vec3.copy(self.agent.position)
        self.prev_agent_velocity = vec3.copy(self.agent.velocity)

        # Simulate physics for both entities
        simulate(self.agent, self.agent_input)
        simulate(self.target, self.target_input)

        # Calculate reward
        reward = self._calculate_reward(action, prev_agent_pos)

        # Check termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        if terminated or truncated:
            self.episode_number += 1

        return self._get_observation(), reward, terminated, truncated, {}

    def _calculate_reward(self, current_action, prev_agent_pos):
        """Calculate reward based on curriculum stage"""

        prev_distance = self.prev_distance
        current_distance = vec3.distance_xz(self.agent.position, self.target.position)

        # 1. Reward getting closer, punish getting farther
        reward = (prev_distance - current_distance) * 5.0

        # 2. Tiny penalty for doing nothing
        if not any(current_action[:4]):  # no WASD
            reward -= 0.01

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
        """Get current observation state"""
        # Agent state
        agent_to_target = vec3.subtract(self.target.position, self.agent.position)
        yaw_to_target, pitch_to_target, distance_to_target = (
            angles.vec_to_yaw_pitch_distance(agent_to_target)
        )
        yaw_diff_to_target = angles.yaw_difference(self.agent.yaw, yaw_to_target)
        pitch_diff_to_target = angles.pitch_difference(
            self.agent.pitch, pitch_to_target
        )
        cos_yaw_diff = math.cos(yaw_diff_to_target)
        sin_yaw_diff = math.sin(yaw_diff_to_target)

        target_to_agent = vec3.subtract(self.agent.position, self.target.position)
        target_yaw, target_pitch, _ = angles.vec_to_yaw_pitch_distance(target_to_agent)
        target_yaw_diff_to_agent = angles.yaw_difference(self.target.yaw, target_yaw)
        target_pitch_diff_to_agent = angles.pitch_difference(
            self.target.pitch, target_pitch
        )

        agent_velocity_yaw, agent_velocity_pitch, agent_speed = (
            angles.vec_to_yaw_pitch_distance(self.agent.velocity)
        )
        agent_velocity_yaw_diff = angles.yaw_difference(
            self.agent.yaw, agent_velocity_yaw
        )
        agent_velocity_pitch_diff = angles.pitch_difference(
            self.agent.pitch, agent_velocity_pitch
        )
        agent_on_ground = 1.0 if self.agent.on_ground else 0.0

        target_velocity_yaw, target_velocity_pitch, target_speed = (
            angles.vec_to_yaw_pitch_distance(self.target.velocity)
        )
        target_velocity_yaw_diff = angles.yaw_difference(
            self.target.yaw, target_velocity_yaw
        )
        target_velocity_pitch_diff = angles.pitch_difference(
            self.target.pitch, target_velocity_pitch
        )
        target_on_ground = 1.0 if self.target.on_ground else 0.0

        obs = np.array(
            [
                cos_yaw_diff,
                sin_yaw_diff,
                pitch_diff_to_target / (np.pi / 2),
                distance_to_target / 10.0,
                target_yaw_diff_to_agent / np.pi,
                target_pitch_diff_to_agent / (np.pi / 2),
                agent_velocity_yaw_diff / np.pi,
                agent_velocity_pitch_diff / (np.pi / 2),
                agent_speed / 2.0,
                agent_on_ground,
                target_velocity_yaw_diff / np.pi,
                target_velocity_pitch_diff / (np.pi / 2),
                target_speed / 2.0,
                target_on_ground,
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
            print(f"Curriculum Stage: {self.curriculum_stage}")
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
            "curriculum_stage": self.curriculum_stage,
            "target_mode": mode_name,
            "target_mode_flags": hex(self.target_mode),
            "distance": vec3.distance(self.agent.position, self.target.position),
            "agent_speed": vec3.length(self.agent.velocity),
        }
