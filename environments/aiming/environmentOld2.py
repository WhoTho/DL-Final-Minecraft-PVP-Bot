import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from helpers import vec3, angles, world
from simulator.objects import Entity
from simulator.physics import simulate, InputState, GROUND_Y
from simulator import combat

MAX_DEGREES_PER_STEP = 360 / 0.5 / 20  # 360 degrees in 0.5s at 20 steps/s


class AimingEnv(gym.Env):
    """
    Agent entity that moves around randomly, tries to aim at target.
    Target entity that moves around randomly.
    Action: Δyaw, Δpitch (in radians)
    Observation: Relative target direction, distance, velocities in agent's local frame
    Reward: Based on aim accuracy
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Agent entity (starts at origin, can move)
        self.agent = Entity(
            object_id=1,
            position=vec3.from_list([0.0, GROUND_Y, 0.0]),
            yaw=0.0,
            pitch=0.0,
            color=(0, 255, 0),
        )
        self.agent_input = InputState()

        # Target entity (starts offset, can move)
        self.target = Entity(
            object_id=2,
            position=vec3.from_list([5.0, GROUND_Y, 5.0]),
            yaw=0.0,
            pitch=0.0,
            color=(255, 0, 0),
        )
        self.target_input = InputState()

        # Target behavior state
        self.target_movement_timer = 0
        self.agent_movement_timer = 0
        self.prev_target_look_direction = vec3.from_yaw_pitch(0.0, 0.0)

        # Action: Δyaw, Δpitch in radians (small adjustments)
        max_delta = np.radians(MAX_DEGREES_PER_STEP)
        self.action_space = spaces.Box(
            low=np.array([-max_delta, -max_delta], dtype=np.float32),
            high=np.array([max_delta, max_delta], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(5,), dtype=np.float32
        )

        # Episode length
        self.max_steps = 200
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agent
        self.agent.position = vec3.from_list(
            [
                self.np_random.uniform(-2.0, 2.0),
                GROUND_Y,
                self.np_random.uniform(-2.0, 2.0),
            ]
        )
        self.agent.velocity = vec3.zero()
        self.agent.yaw = self.np_random.uniform(-np.pi, np.pi)
        self.agent.pitch = self.np_random.uniform(-np.pi / 12, np.pi / 12)
        self.agent.on_ground = True
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
        self.target.yaw = self.np_random.uniform(-np.pi, np.pi)
        self.target.pitch = self.np_random.uniform(-np.pi / 12, np.pi / 12)
        self.target.on_ground = True
        self.target_input = InputState()

        self.target_movement_timer = 0
        self.agent_movement_timer = 0
        self.current_step = 0

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        # Apply Δyaw, Δpitch
        dyaw, dpitch = action
        self.agent.yaw = angles.yaw_difference(0, self.agent.yaw + dyaw)
        self.agent.pitch = np.clip(
            self.agent.pitch + dpitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01
        )

        # Update agent movement input every 4 steps
        if self.agent_movement_timer % 4 == 0:
            # Random movement for agent
            if np.random.rand() < 0.6:
                self.agent_input.w = True
                self.agent_input.s = False
            else:
                self.agent_input.w = False
                self.agent_input.s = True

            strafe_choice = self.np_random.choice(
                ["left", "right", "none"], p=[0.3, 0.3, 0.4]
            )
            if strafe_choice == "left":
                self.agent_input.a = True
                self.agent_input.d = False
            elif strafe_choice == "right":
                self.agent_input.a = False
                self.agent_input.d = True
            else:
                self.agent_input.a = False
                self.agent_input.d = False

            self.agent_input.sprint = np.random.rand() < 0.5

        self.agent_movement_timer += 1

        # Update target movement input every 4 steps
        if self.target_movement_timer % 4 == 0:
            # Random movement for target
            if np.random.rand() < 0.7:
                self.target_input.w = True
                self.target_input.s = False
            else:
                self.target_input.w = False
                self.target_input.s = True

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

            self.target_input.sprint = np.random.rand() < 0.9

            # jump occasionally
            self.target_input.space = np.random.rand() < 0.1

        self.target_movement_timer += 1

        # Update agent/target orientation inputs
        self.agent_input.yaw = self.agent.yaw
        self.agent_input.pitch = self.agent.pitch
        self.target_input.yaw = self.target.yaw
        self.target_input.pitch = self.target.pitch

        # Simulate physics
        simulate(self.agent, self.agent_input)
        simulate(self.target, self.target_input.clone())

        # Compute reward
        reward = self._compute_reward(action)

        # Check termination
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        """Get current observation in agent's local frame"""
        agent_to_target = vec3.subtract(self.target.position, self.agent.position)
        yaw, pitch, distance = angles.vec_to_yaw_pitch_distance(agent_to_target)
        yaw_diff = angles.yaw_difference(self.agent.yaw, yaw)
        pitch_diff = angles.pitch_difference(self.agent.pitch, pitch)
        return np.clip(
            np.array(
                [
                    math.sin(yaw_diff),
                    math.cos(yaw_diff),
                    math.sin(pitch_diff),
                    math.cos(pitch_diff),
                    distance / 15.0,  # normalize distance
                ],
                dtype=np.float32,
            ),
            -1.0,
            1.0,
        )
        # forward, right, up = world.yaw_pitch_to_basis_vectors(
        #     self.agent.yaw, self.agent.pitch
        # )

        # # Direction to target
        # agent_to_target_world = vec3.subtract(self.target.position, self.agent.position)
        # agent_to_target_local = world.world_to_local(
        #     agent_to_target_world, forward, right, up
        # )
        # agent_to_target_local_dir, agent_to_target_distance = vec3.direction_and_length(
        #     agent_to_target_local
        # )

        # # Target velocity
        # target_velocity_local = world.world_to_local(
        #     self.target.velocity, forward, right, up
        # )
        # target_velocity_local_dir, target_velocity_speed = vec3.direction_and_length(
        #     target_velocity_local
        # )

        # # Agent velocity
        # agent_velocity_local = world.world_to_local(
        #     self.agent.velocity, forward, right, up
        # )
        # agent_velocity_local_dir, agent_velocity_speed = vec3.direction_and_length(
        #     agent_velocity_local
        # )

        # obs = np.array(
        #     [
        #         agent_to_target_local_dir[0],
        #         agent_to_target_local_dir[1],
        #         agent_to_target_local_dir[2],
        #         agent_to_target_distance / 10.0,  # normalize distance
        #         target_velocity_local_dir[0],
        #         target_velocity_local_dir[1],
        #         target_velocity_local_dir[2],
        #         target_velocity_speed / 2.0,  # normalize speed
        #         agent_velocity_local_dir[0],
        #         agent_velocity_local_dir[1],
        #         agent_velocity_local_dir[2],
        #         agent_velocity_speed / 2.0,  # normalize speed
        #     ],
        #     dtype=np.float32,
        # )

        return np.clip(obs, -1.0, 1.0)

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
        reward -= 0.5 * (abs(action[0]) + abs(action[1]))

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
        #     reward += (distance_to_target - distance_to_intersection + 2) * 0.01

        return reward

    def get_perfect_action(self):
        """Get the perfect action to aim at target"""
        direction_to_target = vec3.subtract(self.target.position, self.agent.position)
        target_yaw, target_pitch, _ = angles.vec_to_yaw_pitch_distance(
            direction_to_target
        )

        dyaw = angles.yaw_difference(self.agent.yaw, target_yaw)
        dpitch = angles.pitch_difference(self.agent.pitch, target_pitch)

        # Clamp to action space
        max_delta = np.radians(MAX_DEGREES_PER_STEP)
        dyaw = np.clip(dyaw, -max_delta, max_delta)
        dpitch = np.clip(dpitch, -max_delta, max_delta)

        return np.array([dyaw, dpitch], dtype=np.float32)

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
