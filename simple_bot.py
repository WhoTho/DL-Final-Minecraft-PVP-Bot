# simple_bot.py
import math
from simulator.server import MinecraftSimulationServer
from helpers import vec3, angles


class SimpleBot:
    """A simple PvP bot that chases and attacks the target player"""

    def __init__(self, bot_id: int, target_id: int):
        self.bot_id = bot_id
        self.target_id = target_id
        self.attack_interval = 2  # Attack every 2 ticks

    def update(self, server: MinecraftSimulationServer):
        return
        """Update the bot AI to chase and attack the target player"""
        bot = server.get_entity(self.bot_id)
        target = server.get_entity(self.target_id)

        if not bot or not target:
            return

        # Calculate direction to target (eye to eye)
        bot_eye_pos = vec3.add(bot.position, vec3.from_list([0, bot.eye_height, 0]))
        target_eye_pos = vec3.add(
            target.position, vec3.from_list([0, target.eye_height, 0])
        )

        direction_to_target = vec3.subtract(target_eye_pos, bot_eye_pos)
        distance_to_target = vec3.length(direction_to_target)

        # Look directly at target's eyes
        if distance_to_target > 0.1:  # Avoid division by zero
            yaw, pitch, _ = angles.vec_to_yaw_pitch_distance(direction_to_target)
            bot.yaw = yaw
            bot.pitch = pitch

        # Movement logic: sprint run towards player
        move_forward = distance_to_target > 1.5  # Stop when very close
        should_attack = (
            distance_to_target <= bot.reach
            and server.tick_count % self.attack_interval == 0
        )

        # Create input for bot
        bot_input = {
            "w": move_forward,  # Move forward when chasing
            "a": False,
            "s": False,
            "d": False,
            "sprint": move_forward,  # Always sprint when moving
            "space": False,  # Could add jump logic later
            "click": should_attack,  # Attack every other tick when in range
            "yaw": bot.yaw,
            "pitch": bot.pitch,
        }

        # Send input to server
        server.take_input(self.bot_id, bot_input)

        return {
            "distance": distance_to_target,
            "moving": move_forward,
            "attacking": should_attack,
            "looking_at": (bot.yaw, bot.pitch),
        }
