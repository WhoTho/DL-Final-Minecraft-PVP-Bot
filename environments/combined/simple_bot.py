# simple_bot.py
import math
from helpers import vec3, angles
import numpy as np
from simulator.physics import InputState
from typing import TYPE_CHECKING
from environments.base_enviroment import TPS, MAX_ANGLE_PER_STEP

if TYPE_CHECKING:
    from simulator.objects import Entity

ATTACK_CPS = 6
ATTACK_INTERVAL = TPS // ATTACK_CPS
STRAFE_INTERVAL = 8


class SimpleBot:
    """A simple PvP bot that chases and attacks using only local entity data"""

    def __init__(self):
        self.bot_yaw_velocity = 0.0
        self.current_strafe_direction = "none"

    def reset(self):
        self.bot_yaw_velocity = 0.0
        self.current_strafe_direction = "none"

    def get_bot_inputs(
        self,
        current_step: int,
        bot_entity: "Entity",
        target_entity: "Entity",
        rng: np.random.Generator,
    ) -> InputState:
        """
        Update the bot AI to chase and attack the target player.

        Args:
            bot_entity: Entity object for the bot
            target_entity: Entity object for the target
            all_entities: List of all entities in the environment for raycast
        """
        bot_input = InputState()

        if not bot_entity or not target_entity:
            return bot_input

        # Calculate direction to target (eye to eye)
        bot_eye_pos = bot_entity.get_eye_position()
        target_eye_pos = target_entity.get_eye_position()

        direction_to_target = vec3.subtract(target_eye_pos, bot_eye_pos)
        distance_to_target = vec3.length(direction_to_target)

        # Look directly at target's eyes
        if distance_to_target > 0.1:  # Avoid division by zero
            delta_yaw = self._calculate_delta_yaw(bot_entity, direction_to_target)
            _, target_pitch, _ = angles.vec_to_yaw_pitch_distance(direction_to_target)
            bot_input.yaw = angles.yaw_difference(0, bot_entity.yaw + delta_yaw)
            bot_input.pitch = np.clip(target_pitch, -np.pi / 2, np.pi / 2)
        else:
            bot_input.yaw = 0.0
            bot_input.pitch = 0.0

        # Movement logic: sprint run towards player
        if distance_to_target > 1.5:
            bot_input.w = True
            bot_input.sprint = True
        else:
            bot_input.w = False
            bot_input.sprint = False

        # Strafing logic
        if distance_to_target > 5.0:
            self.current_strafe_direction = "none"
        elif current_step % STRAFE_INTERVAL == 0:
            if rng.random() < 0.2:  # 20% chance to change strafe
                self.current_strafe_direction = rng.choice(
                    ["left", "right", "none"], p=[0.4, 0.4, 0.2]
                )
        if self.current_strafe_direction == "left":
            bot_input.a = True
            bot_input.d = False
        elif self.current_strafe_direction == "right":
            bot_input.a = False
            bot_input.d = True
        else:
            bot_input.a = False
            bot_input.d = False

        # Auto-click
        should_attack = (
            distance_to_target <= bot_entity.reach + 1
            and current_step % ATTACK_INTERVAL == 0
        )

        bot_input.space = False
        bot_input.click = should_attack

        return bot_input

    def _calculate_delta_yaw(
        self, bot_entity: "Entity", bot_to_target: vec3.VEC3
    ) -> float:
        """Calculate the smallest delta yaw to face the target"""
        target_yaw, _, _ = angles.vec_to_yaw_pitch_distance(bot_to_target)
        yaw_diff = angles.yaw_difference(bot_entity.yaw, target_yaw)

        # aiming in same direction speeds up turning slowly, opposite direction slows down quickly
        # if np.sign(yaw_diff) == np.sign(self.bot_yaw_velocity):
        #     self.bot_yaw_velocity += yaw_diff * 0.3
        # else:
        #     self.bot_yaw_velocity += yaw_diff * 0.8
        # self.bot_yaw_velocity *= 0.8  # damping
        self.bot_yaw_velocity = 0.7 * self.bot_yaw_velocity + 0.3 * yaw_diff

        self.bot_yaw_velocity = np.clip(
            self.bot_yaw_velocity,
            -MAX_ANGLE_PER_STEP,
            MAX_ANGLE_PER_STEP,
        )
        return self.bot_yaw_velocity
