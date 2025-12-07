import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from helpers import vec3, angles, world
from simulator.objects import Entity
from simulator.physics import simulate, InputState, GROUND_Y

TPS = 20  # Simulation ticks per second
DEFAULT_HEALTH = 100.0
MAX_ANGLE_PER_STEP = np.radians(720 / TPS)


class BaseEnv(gym.Env):
    """
    Base environment class for Minecraft-like PvP simulation.
    Provides common functionality for resetting and stepping through the environment.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, max_steps: int, render_mode=None):
        super(BaseEnv, self).__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        self.agent = Entity(
            object_id=1,
            position=vec3.from_list([0.0, GROUND_Y, 0.0]),
            velocity=vec3.zero(),
            yaw=0.0,
            pitch=0.0,
            color=(0, 255, 0),
            health=DEFAULT_HEALTH,
            max_health=DEFAULT_HEALTH,
        )
        self.agent_input = InputState()

        self.target = Entity(
            object_id=2,
            position=vec3.from_list([5.0, GROUND_Y, 5.0]),
            velocity=vec3.zero(),
            yaw=0.0,
            pitch=0.0,
            color=(255, 0, 0),
            health=DEFAULT_HEALTH,
            max_health=DEFAULT_HEALTH,
        )
        self.target_input = InputState()

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(18,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state and returns an initial observation.
        """

        super().reset(seed=seed)

        # Reset agent state
        self.agent.position = vec3.from_list(
            [self.np_random.uniform(-1, 1), GROUND_Y, self.np_random.uniform(-1, 1)]
        )
        self.agent.on_ground = True
        self.agent.is_sprinting = False
        self.agent.invulnerablility_ticks = 0
        self.agent.velocity = vec3.zero()
        self.agent.yaw = self.np_random.uniform(-np.pi, np.pi)
        self.agent.pitch = self.np_random.uniform(-np.pi / 12, np.pi / 12)
        self.agent.health = DEFAULT_HEALTH
        self.agent_input = InputState()

        # Reset target state
        angle = self.np_random.uniform(0, 2 * np.pi)
        distance = self.np_random.uniform(5, 10)
        self.target.position = vec3.from_list(
            [
                self.agent.position[0] + distance * math.cos(angle),
                GROUND_Y,
                self.agent.position[2] + distance * math.sin(angle),
            ]
        )
        self.target.on_ground = True
        self.target.is_sprinting = False
        self.target.invulnerablility_ticks = 0
        self.target.velocity = vec3.zero()
        self.target.yaw = self.np_random.uniform(-np.pi, np.pi)
        self.target.pitch = self.np_random.uniform(-np.pi / 12, np.pi / 12)
        self.target.health = DEFAULT_HEALTH
        self.target_input = InputState()

        self.current_step = 0

        return [], {}

    def step(self, action):
        raise NotImplementedError("Step method not implemented in BaseEnv")

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
                # Invulnerability status (normalized)
                self.agent.invulnerablility_ticks / 20.0,
            ],
            dtype=np.float32,
        )

        return np.clip(obs, -1.0, 1.0)

    def render(self):
        raise NotImplementedError("Render method not implemented in BaseEnv")

    def generate_probabilistic_input(self, input_state: InputState) -> None:
        """
        Generate probabilistic input for the entity based on the given InputState.
        This simulates human-like variability in inputs.

        Args:
            input_state: The InputState object to modify.
            rng: A numpy random generator for reproducibility.
        """
        # Forward/backward movement
        if self.np_random.random() < 0.8:
            input_state.w = True
            input_state.s = False
        else:
            input_state.w = False
            input_state.s = True

        strafe_choice = self.np_random.choice(
            ["left", "right", "none"], p=[0.4, 0.4, 0.2]
        )
        if strafe_choice == "left":
            input_state.a = True
            input_state.d = False
        elif strafe_choice == "right":
            input_state.a = False
            input_state.d = True
        else:
            input_state.a = False
            input_state.d = False

        input_state.sprint = self.np_random.random() < 0.9

        # jump occasionally
        input_state.space = self.np_random.random() < 0.1

    def aim_target_at_agent(self) -> None:
        """
        Adjust the target entity's yaw and pitch to look at the agent.
        """
        target_to_agent = vec3.subtract(self.agent.position, self.target.position)
        yaw, pitch, _ = angles.vec_to_yaw_pitch_distance(target_to_agent)
        self.target_input.yaw = yaw
        self.target_input.pitch = pitch
