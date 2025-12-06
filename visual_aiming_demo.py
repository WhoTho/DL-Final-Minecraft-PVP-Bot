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
from environments.aiming.environment import AimingEnv
from models.aiming.baseline_model import AimingModel
from helpers import vec3, world, angles


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
                self.agent = AimingModel()
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
        self.agent_entity = Entity(
            object_id=998,
            position=vec3.zero(),
            color=(0, 255, 0),
        )

        self.target_entity = Entity(
            object_id=999,
            position=vec3.zero(),
            color=(255, 0, 0),
        )

        self.crosshair_color = (255, 255, 255)

    def reset_episode(self):
        """Reset the environment for a new episode"""
        self.state, _ = self.env.reset()
        self.update_visuals()

    def update_visuals(self):
        """Update visual elements based on current environment state"""
        # Update camera position at agent's eye level
        agent_eye_pos = vec3.add(self.env.agent.position, vec3.from_list([0, 1.62, 0]))
        self.cam.set_position(agent_eye_pos)
        self.cam.set_yaw_pitch(self.env.agent.yaw, self.env.agent.pitch)

        # Update agent entity visuals
        self.agent_entity.position = self.env.agent.position
        self.agent_entity.yaw = self.env.agent.yaw
        self.agent_entity.pitch = self.env.agent.pitch

        # Update target entity visuals
        self.target_entity.position = self.env.target.position
        self.target_entity.yaw = self.env.target.yaw
        self.target_entity.pitch = self.env.target.pitch

        # Update crosshair color based on accuracy
        direction_to_target = vec3.subtract(self.env.target.position, agent_eye_pos)

        target_yaw, target_pitch, _ = angles.vec_to_yaw_pitch_distance(
            direction_to_target
        )

        yaw_error = abs(angles.yaw_difference(self.env.agent.yaw, target_yaw))
        pitch_error = abs(angles.pitch_difference(self.env.agent.pitch, target_pitch))

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
        agent_eye_pos = vec3.add(self.env.agent.position, vec3.from_list([0, 1.62, 0]))
        direction_to_target = vec3.subtract(self.env.target.position, agent_eye_pos)

        target_yaw, target_pitch, distance = angles.vec_to_yaw_pitch_distance(
            direction_to_target
        )

        yaw_error = angles.yaw_difference(self.env.agent.yaw, target_yaw)
        pitch_error = angles.pitch_difference(self.env.agent.pitch, target_pitch)

        agent_speed = vec3.length(self.env.agent.velocity)
        target_speed = vec3.length(self.env.target.velocity)

        gui_text = [
            f"Mode: {self.mode.title()} | Auto: {self.auto_step}",
            "",
            "Agent:",
            f"  Pos: ({self.env.agent.position[0]:.1f}, {self.env.agent.position[2]:.1f})",
            f"  Aim: yaw={math.degrees(self.env.agent.yaw):.1f}°, pitch={math.degrees(self.env.agent.pitch):.1f}°",
            f"  Speed: {agent_speed:.2f}",
            "",
            "Target:",
            f"  Pos: ({self.env.target.position[0]:.1f}, {self.env.target.position[2]:.1f})",
            f"  Dir: yaw={math.degrees(target_yaw):.1f}°, pitch={math.degrees(target_pitch):.1f}°",
            f"  Speed: {target_speed:.2f}",
            "",
            "Error:",
            f"  Yaw: {math.degrees(yaw_error):.1f}° | Pitch: {math.degrees(pitch_error):.1f}°",
            f"  Distance: {distance:.2f}",
            "",
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
                            # self.env.agent.pitch = 0
                            # self.agent_entity.pitch = 0
                            # reward = self.step_environment()
                            # self.env.agent.pitch = 0
                            # self.agent_entity.pitch = 0
                            # print(f"Step reward: {reward:.3f}")
                            # forward, right, up = world.yaw_pitch_to_basis_vectors(
                            #     self.env.agent.yaw, self.env.agent.pitch
                            # )
                            # agent_to_target_world = vec3.subtract(
                            #     self.env.target.position, self.env.agent.position
                            # )
                            # agent_to_target_local = world.world_to_local(
                            #     agent_to_target_world, forward, right, up
                            # )
                            # agent_to_target_local_dir, _ = vec3.direction_and_length(
                            #     agent_to_target_local
                            # )
                            # local_yaw_diff, local_pitch_diff, _ = (
                            #     angles.vec_to_yaw_pitch_distance(agent_to_target_local)
                            # )
                            # yaw_world, pitch_world, _ = (
                            #     angles.vec_to_yaw_pitch_distance(agent_to_target_world)
                            # )
                            # yaw_diff_world = angles.yaw_difference(
                            #     self.env.agent.yaw, yaw_world
                            # )
                            # pitch_diff_world = angles.pitch_difference(
                            #     self.env.agent.pitch, pitch_world
                            # )
                            # yaw_diff_local, pitch_diff_local, _ = (
                            #     angles.vec_to_yaw_pitch_distance(agent_to_target_local)
                            # )
                            # print("Current observation:", self.state)
                            # print(
                            #     "Agent to target (local degrees):",
                            #     tuple(
                            #         math.degrees(x) for x in agent_to_target_local_dir
                            #     ),
                            # )
                            # print(
                            #     "Yaw/Pitch to target (local from dir):",
                            #     (
                            #         math.degrees(local_yaw_diff),
                            #         math.degrees(local_pitch_diff),
                            #     ),
                            # )
                            # print(
                            #     "Yaw/Pitch difference (world degrees):",
                            #     (
                            #         math.degrees(yaw_diff_world),
                            #         math.degrees(pitch_diff_world),
                            #     ),
                            # )
                            # print(
                            #     "Yaw/Pitch difference (local degrees):",
                            #     (
                            #         math.degrees(yaw_diff_local),
                            #         math.degrees(pitch_diff_local),
                            #     ),
                            # )

            # Auto-step if enabled
            if self.auto_step and current_time - self.last_step_time > self.step_delay:
                self.step_environment()
                self.last_step_time = current_time

            # Render
            self.renderer.begin_frame()
            self.renderer.draw_ground_grid(self.cam)

            # Draw agent and target entities
            render_entity(self.agent_entity, self.renderer, self.cam)
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
