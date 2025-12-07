# simple_bot.py
import math
from helpers import vec3, angles
import numpy as np
from simulator.physics import InputState
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.objects import Entity


class SimpleBot:
    """A simple PvP bot that chases and attacks using only local entity data"""

    def __init__(self, bot_id: int, target_id: int, step_count: int = 0):
        self.bot_id = bot_id
        self.target_id = target_id
        self.attack_interval = 3  # Attack every 3 ticks
        self.step_count = step_count

    def update(
        self,
        bot_entity: "Entity",
        target_entity: "Entity",
        all_entities: list["Entity"],
    ) -> InputState:
        """
        Update the bot AI to chase and attack the target player.

        Args:
            bot_entity: Entity object for the bot
            target_entity: Entity object for the target
            all_entities: List of all entities in the environment for raycast
        """
        if not bot_entity or not target_entity:
            return InputState()

        # Calculate direction to target (eye to eye)
        bot_eye_pos = bot_entity.get_eye_position()
        target_eye_pos = target_entity.get_eye_position()

        direction_to_target = vec3.subtract(target_eye_pos, bot_eye_pos)
        distance_to_target = vec3.length(direction_to_target)

        # Look directly at target's eyes
        if distance_to_target > 0.1:  # Avoid division by zero
            yaw, pitch, _ = angles.vec_to_yaw_pitch_distance(direction_to_target)
            bot_entity.yaw = yaw
            bot_entity.pitch = pitch

        # Movement logic: sprint run towards player
        move_forward = distance_to_target > 1.5  # Stop when very close

        # Auto-click every 3 ticks when in range
        should_attack = (
            distance_to_target <= bot_entity.reach
            and self.step_count % self.attack_interval == 0
        )

        # Perform attack if needed
        # if should_attack:
        #     combat.try_attack(bot_entity, all_entities)

        # Create input state for bot

        bot_input = InputState()
        bot_input.w = move_forward
        bot_input.a = False
        bot_input.s = False
        bot_input.d = False
        bot_input.sprint = move_forward
        bot_input.space = False
        bot_input.yaw = bot_entity.yaw
        bot_input.pitch = bot_entity.pitch
        bot_input.click = should_attack

        self.step_count += 1

        return bot_input
