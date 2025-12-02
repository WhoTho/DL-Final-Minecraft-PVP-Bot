# visual_aiming_demo.py
"""
Visual aiming demo using the existing game renderer
"""

import time
import pygame
import math
import numpy as np
from renderer.cameras import FirstPersonCamera
from renderer.renderer import Renderer3D
from renderer.objects import render_entity
from simulator.objects import Entity
from enviroments.aiming.enviroment import AimingEnv
from models.aiming.model import AimingAgent
from models.aiming.baseline_model import BaselineAimingModel
from helpers import vec3
from helpers.angles import (
    vec_to_yaw_pitch_distance,
    yaw_difference,
    pitch_difference,
)


class VisualAimingDemo:
    """Visual demo of the aiming environment using the game renderer"""

    def __init__(self, model_path=None):
        self.WIDTH, self.HEIGHT = 1000, 700

        # Initialize renderer and camera
        self.renderer = Renderer3D(self.WIDTH, self.HEIGHT, "Aiming Training Demo")
        self.cam = FirstPersonCamera(self.WIDTH, self.HEIGHT)

        # Initialize environment
        self.env = AimingEnv()
        self.mode = "perfect"  # "perfect", "random", "ai"

        # Initialize agent (optional)
        self.agent = None
        if model_path:
            try:
                # self.agent = AimingAgent()
                # self.agent.load(model_path)
                self.agent = BaselineAimingModel()
                self.agent.load(model_path)

                self.mode = "ai"
                print(f"Loaded trained model: {model_path}")
            except FileNotFoundError:
                print(f"Model {model_path} not found. Using perfect aiming.")

        # Demo state
        self.state = None
        self.auto_step = True
        self.step_delay = 0.05  # seconds
        self.last_step_time = 0

        # Visual elements
        self.target_entity = Entity(
            object_id=999,
            position=vec3.zero(),
        )

        self.crosshair_color = (255, 255, 255)

    def reset_episode(self):
        """Reset the environment for a new episode"""
        self.state, _ = self.env.reset()
        self.update_visuals()

    def update_visuals(self):
        """Update visual elements based on current environment state"""
        # Update camera position (fixed at player position)
        self.cam.set_position(self.env.player_pos)
        self.cam.set_yaw_pitch(self.env.yaw, self.env.pitch)

        # Create target entity for rendering
        target_pos = self.env.target_pos
        self.target_entity.position = vec3.subtract(
            target_pos, vec3.from_list([0, 1.62, 0])
        )

        # Update crosshair color based on accuracy
        direction_to_target = vec3.subtract(self.env.target_pos, self.env.player_pos)

        target_yaw, target_pitch, _ = vec_to_yaw_pitch_distance(direction_to_target)

        yaw_error = abs(yaw_difference(self.env.yaw, target_yaw))
        pitch_error = abs(pitch_difference(self.env.pitch, target_pitch))

        # Green when accurate, red when inaccurate
        if yaw_error < 0.05 and pitch_error < 0.05:  # ~3 degrees
            self.crosshair_color = (0, 255, 0)  # Green
        elif yaw_error < 0.1 and pitch_error < 0.1:  # ~6 degrees
            self.crosshair_color = (255, 255, 0)  # Yellow
        else:
            self.crosshair_color = (255, 255, 255)  # White

    def get_next_action(self):
        """Get next action based on current mode"""
        if self.mode == "perfect":
            return self.env.get_perfect_action()
        elif self.mode == "random":
            return self.env.action_space.sample()
        elif self.mode == "ai" and self.agent:
            action = self.agent.predict(self.state, deterministic=True)
            # action, _ = self.agent.act(self.state, training=False)
            return action
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def step_environment(self):
        """Take one step in the environment"""
        action = self.get_next_action()
        self.state, reward, terminated, truncated, _ = self.env.step(action)

        if not self.auto_step:
            print("Current state:", self.state)
            print(
                f"Action taken: Δyaw={math.degrees(action[0]):.2f}°, Δpitch={math.degrees(action[1]):.2f}°, Reward: {reward:.3f}"
            )

        if terminated or truncated:
            self.reset_episode()

        self.update_visuals()
        return reward

    def draw_crosshair(self):
        """Draw crosshair at screen center"""
        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
        size = 10

        # Draw crosshair lines
        pygame.draw.line(
            self.renderer.screen,
            self.crosshair_color,
            (center_x - size, center_y),
            (center_x + size, center_y),
            2,
        )
        pygame.draw.line(
            self.renderer.screen,
            self.crosshair_color,
            (center_x, center_y - size),
            (center_x, center_y + size),
            2,
        )

        # Draw center dot
        pygame.draw.circle(
            self.renderer.screen, self.crosshair_color, (center_x, center_y), 2
        )

    def draw_gui(self):
        """Draw GUI information"""
        # Get current aiming info
        direction_to_target = vec3.subtract(self.env.target_pos, self.env.player_pos)

        target_yaw, target_pitch, distance = vec_to_yaw_pitch_distance(
            direction_to_target
        )

        yaw_error = yaw_difference(self.env.yaw, target_yaw)
        pitch_error = pitch_difference(self.env.pitch, target_pitch)

        gui_text = [
            f"Mode: {self.mode.title()} | Auto: {self.auto_step}",
            f"Current Aim: yaw={math.degrees(self.env.yaw):.1f}°, pitch={math.degrees(self.env.pitch):.1f}°",
            f"Target:      yaw={math.degrees(target_yaw):.1f}°, pitch={math.degrees(target_pitch):.1f}°",
            f"Error:       yaw={math.degrees(yaw_error):.1f}°, pitch={math.degrees(pitch_error):.1f}°",
            f"Distance: {distance:.2f}",
            f"Step: {self.env.current_step}/{self.env.max_steps}",
            "",
            "Controls:",
            "1/2/3 = Perfect/Random/AI mode",
            "Space = Manual step",
            "A = Toggle auto-step",
            "R = Reset episode",
            "Q = Quit",
        ]

        self.renderer.draw_gui_lines(gui_text, (10, 10), color=(255, 255, 255))

    def run(self):
        """Main demo loop"""
        running = True

        # Initialize first episode
        self.reset_episode()

        while running:
            current_time = pygame.time.get_ticks() / 1000.0

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_1:
                        self.mode = "perfect"
                        print("Mode: Perfect Aiming")
                    elif event.key == pygame.K_2:
                        self.mode = "random"
                        print("Mode: Random Aiming")
                    elif event.key == pygame.K_3:
                        self.mode = "ai"
                        print("Mode: AI Aiming")
                    elif event.key == pygame.K_a:
                        self.auto_step = not self.auto_step
                        print(f"Auto-step: {self.auto_step}")
                    elif event.key == pygame.K_r:
                        self.reset_episode()
                        print("Episode reset")
                    elif event.key == pygame.K_SPACE:
                        if not self.auto_step:
                            reward = self.step_environment()
                            print(f"Step reward: {reward:.3f}")

            # Auto-step if enabled
            if self.auto_step and current_time - self.last_step_time > self.step_delay:
                self.step_environment()
                self.last_step_time = current_time

            # Render
            self.renderer.begin_frame()
            self.renderer.draw_ground_grid(self.cam)

            # Draw target
            if self.target_entity:
                render_entity(self.target_entity, self.renderer, self.cam)

            # Draw crosshair
            self.draw_crosshair()

            # Draw GUI
            self.draw_gui()

            self.renderer.finish_frame()

        pygame.quit()


def main():
    """Main function to run visual aiming demo"""
    import sys

    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"Using model: {model_path}")
    else:
        print("No model specified. Use: python visual_aiming_demo.py [model_path]")
        print("Running with perfect aiming demo only.")

    demo = VisualAimingDemo(model_path)
    demo.run()


if __name__ == "__main__":
    main()
