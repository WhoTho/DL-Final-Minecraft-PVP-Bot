import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from helpers import vec3, angles, world
from simulator.objects import Entity
from simulator.physics import simulate, InputState, GROUND_Y
from simulator import combat
from .simple_bot import SimpleBot

MAX_DELTA_ANGLE_PER_SECOND = 540

DEFAULT_HEALTH = 100.0


class CombinedEnv(gym.Env):
    """
    PvP combat environment.
    Agent controls: W, A, S, D, SPACE, SPRINT, yaw, pitch, click
    Target (bot) uses SimpleBot AI to chase and attack.
    Reward based on: damage dealt, health difference.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Environment parameters
        self.max_steps = 400
        self.current_step = 0

        # Agent setup
        self.agent = Entity(
            object_id=1,
            position=vec3.from_list([0.0, GROUND_Y, 0.0]),
            yaw=0.0,
            pitch=0.0,
            health=DEFAULT_HEALTH,
            max_health=DEFAULT_HEALTH,
            color=(0, 255, 0),
        )
        self.agent_input = InputState()

        # Target (bot) setup
        self.target = Entity(
            object_id=2,
            position=vec3.from_list([5.0, GROUND_Y, 5.0]),
            yaw=0.0,
            pitch=0.0,
            health=DEFAULT_HEALTH,
            max_health=DEFAULT_HEALTH,
            color=(255, 0, 0),
        )

        # Initialize bot AI
        self.bot = SimpleBot(bot_id=2, target_id=1)
        self.target_input = InputState()

        # Track initial health for damage calculation
        self.agent_initial_health = self.agent.health
        self.target_initial_health = self.target.health
        self.agent_should_click = False

        # Action space: W, A, S, D, SPACE, SPRINT (6 discrete binary + 2 continuous for yaw/pitch)
        # For simplicity, using MultiBinary(6) for movement and discrete for look
        self.action_space = spaces.Box(
            low=np.array([-np.pi, -np.pi / 2], dtype=np.float32),
            high=np.array([np.pi, np.pi / 2], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # Action space for movement is separate - we'll use Box for continuous action
        # Combined: [w, a, s, d, space, sprint, click, dyaw, dpitch]

        delta_angle_per_step = (
            math.radians(MAX_DELTA_ANGLE_PER_SECOND) / 20.0
        )  # Assuming 20 FPS

        self.action_space = spaces.Box(
            low=np.array(
                [0, 0, 0, 0, 0, 0, 0, -delta_angle_per_step, -delta_angle_per_step],
                dtype=np.float32,
            ),
            high=np.array(
                [1, 1, 1, 1, 1, 1, 1, delta_angle_per_step, delta_angle_per_step],
                dtype=np.float32,
            ),
            shape=(9,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(20,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agent
        self.agent.position = vec3.from_list([0.0, GROUND_Y, 0.0])
        self.agent.velocity = vec3.zero()
        self.agent.yaw = 0.0
        self.agent.pitch = 0.0
        self.agent.health = DEFAULT_HEALTH
        self.agent.on_ground = True
        self.agent.invulnerablility_ticks = 0
        self.agent_input = InputState()

        # Reset target
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
        self.target.yaw = 0.0
        self.target.pitch = 0.0
        self.target.health = DEFAULT_HEALTH
        self.target.on_ground = True
        self.target.invulnerablility_ticks = 0
        self.target_input = InputState()

        # Reset tracking variables
        self.current_step = 0
        self.agent_initial_health = self.agent.health
        self.target_initial_health = self.target.health
        self.bot.step_count = 0

        return self._get_observation(), {}

    def _action_to_input(self, action):
        """Convert continuous action to InputState"""
        input_state = InputState()

        # Movement keys (0-5)
        input_state.w = action[0] > 0.5
        input_state.a = action[1] > 0.5
        input_state.s = action[2] > 0.5
        input_state.d = action[3] > 0.5
        input_state.space = action[4] > 0.5
        input_state.sprint = action[5] > 0.5

        # Click (6) - store for processing in step()
        input_state.click = action[6] > 0.5

        # Look angles (7-8)
        self.agent.yaw += action[7]
        self.agent.pitch += action[8]
        # Clamp pitch
        self.agent.pitch = np.clip(self.agent.pitch, -np.pi / 2, np.pi / 2)

        input_state.yaw = self.agent.yaw
        input_state.pitch = self.agent.pitch

        return input_state

    def step(self, action):
        # Convert agent action to input
        self.agent_input = self._action_to_input(action)

        # Update bot AI (bot also attacks)
        self.target_input = self.bot.update(
            self.target, self.agent, [self.agent, self.target]
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
        simulate(self.target, self.target_input.clone())

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
        """
        Calculate reward based on:
        1. Damage dealt to opponent
        2. Health advantage
        """
        reward = 0.0

        # Reward for dealing damage
        agent_damage_dealt = self.agent_initial_health - self.target.health
        if agent_damage_dealt > 0:
            reward += agent_damage_dealt * 0.5

        # Reward for health advantage
        health_difference = (self.agent.health - self.target.health) / 20.0
        reward += health_difference * 0.3

        # Small penalty for taking damage
        agent_damage_taken = self.agent_initial_health - self.agent.health
        if agent_damage_taken > 0:
            reward -= agent_damage_taken * 0.25

        return reward

    def _get_observation(self):
        """Get current observation state in agent's local coordinate frame"""
        # Create rotation matrix to transform world vectors to agent's local frame
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

        # Health states (normalized)
        agent_health_norm = self.agent.health / 20.0
        target_health_norm = self.target.health / 20.0

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
                # Health states
                agent_health_norm,
                target_health_norm,
                # Ground contact flags
                agent_on_ground,
                target_on_ground,
                # Target look direction (local frame)
                target_look_local[0],
                target_look_local[1],
                target_look_local[2],
                # Invulnerability status (normalized)
                self.agent.invulnerablility_ticks / 10.0,
            ],
            dtype=np.float32,
        )

        return np.clip(obs, -1.0, 1.0)

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
