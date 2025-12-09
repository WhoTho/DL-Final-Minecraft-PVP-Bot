from models.base_model import BaseModel
import numpy as np
from helpers import vec3, angles, world
from simulator.physics import InputState
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.objects import Entity

model = BaseModel().load("training_results/distilled_latest.zip")


class SelfPlayAgent:
    """An agent that uses a pre-trained model to act in the CombinedEnv."""

    def __init__(self):
        self.model = model

    def reset(self):
        pass

    def get_bot_inputs(
        self,
        current_step: int,
        bot_entity: "Entity",
        target_entity: "Entity",
        rng: np.random.Generator,
    ) -> InputState:
        obs = self._create_observation(bot_entity, target_entity)
        action = self.model.predict(obs, deterministic=True)
        bot_input = InputState()
        # Map continuous action to InputState
        bot_input.w = action[0] > 0.5
        bot_input.a = action[1] > 0.5
        bot_input.s = action[2] > 0.5
        bot_input.d = action[3] > 0.5
        bot_input.space = action[4] > 0.5
        bot_input.sprint = action[5] > 0.5
        bot_input.click = action[6] > 0.5
        bot_input.yaw = angles.yaw_difference(0, bot_entity.yaw + action[7])
        bot_input.pitch = np.clip(bot_entity.pitch + action[8], -np.pi / 2, np.pi / 2)
        return bot_input

    def _create_observation(
        self,
        bot_entity: "Entity",
        target_entity: "Entity",
    ) -> np.ndarray:
        # Create rotation matrix to transform world vectors to agent's local frame
        forward, right, up = world.yaw_pitch_to_basis_vectors(
            bot_entity.yaw, bot_entity.pitch
        )

        # Direction to target
        to_target_world = vec3.subtract(target_entity.position, bot_entity.position)
        to_target_local_dir, distance_to_target = vec3.direction_and_length(
            to_target_world
        )
        to_target_local = world.world_to_local(to_target_local_dir, forward, right, up)

        # Agent velocity
        agent_vel_dir, agent_speed = vec3.direction_and_length(bot_entity.velocity)
        if agent_speed > 1e-6:
            agent_vel_local = world.world_to_local(agent_vel_dir, forward, right, up)
        else:
            agent_vel_local = vec3.zero()

        # Target velocity
        target_vel_dir, target_speed = vec3.direction_and_length(target_entity.velocity)
        if target_speed > 1e-6:
            target_vel_local = world.world_to_local(target_vel_dir, forward, right, up)
        else:
            target_vel_local = vec3.zero()

        # Target look direction
        target_look = vec3.from_yaw_pitch(target_entity.yaw, target_entity.pitch)
        target_look_local = world.world_to_local(target_look, forward, right, up)

        # Binary flags
        agent_on_ground = 1.0 if bot_entity.on_ground else -1.0
        target_on_ground = 1.0 if target_entity.on_ground else -1.0

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
                bot_entity.invulnerablility_ticks / 20.0,
            ],
            dtype=np.float32,
        )

        return np.clip(obs, -1.0, 1.0)
