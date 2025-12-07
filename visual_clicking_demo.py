# visual_clicking_demo.py
"""
Visual clicking demo using the existing game renderer
"""

import time
import pygame
import math
import numpy as np
from renderer.cameras import FirstPersonCamera
from renderer.renderer import Renderer3D
from renderer.objects import render_entity
from simulator.objects import Entity
from environments.clicking.environment import ClickingEnv
from models.clicking.baseline_model import ClickingModel
from helpers import vec3, angles


class VisualClickingDemo:
    """Visual demo of the clicking environment using the game renderer"""

    def __init__(self, model_path=None):
        self.WIDTH, self.HEIGHT = 1000, 700

        # Initialize renderer and camera
        self.renderer = Renderer3D(self.WIDTH, self.HEIGHT, "Clicking Training Demo")
        self.cam = FirstPersonCamera(self.WIDTH, self.HEIGHT)

        # Initialize environment
        self.env = ClickingEnv()
        self.mode = "random"  # "random", "ai", "manual"

        # Initialize agent (optional)
        self.agent = None
        if model_path:
            try:
                self.agent = ClickingModel()
                self.agent.load(model_path)
                self.mode = "ai"
                print(f"Loaded trained model: {model_path}")
            except FileNotFoundError:
                print(f"Model {model_path} not found. Using random clicking.")

        # Demo state
        self.state = None
        self.auto_step = True
        self.step_delay = 0.05  # seconds
        self.last_step_time = 0
        self.manual_click = False

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

        # Tracking for visualization
        self.last_hits = []  # List of (step, confidence) for recent hits
        self.last_action = 0

    def reset_episode(self):
        """Reset the environment for a new episode"""
        self.state, _ = self.env.reset()
        self.last_hits = []
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

    def get_next_action(self):
        """Get next action based on current mode"""
        if self.mode == "manual":
            return 1 if self.manual_click else 0
        elif self.mode == "random":
            return self.env.action_space.sample()
        elif self.mode == "ai" and self.agent:
            action = self.agent.predict(self.state, deterministic=True)
            return action
        else:
            return 0

    def step_environment(self):
        """Take one step in the environment"""
        action = self.get_next_action()
        self.last_action = action
        self.state, reward, terminated, truncated, _ = self.env.step(action)

        if not self.auto_step:
            action_str = "CLICK" if action == 1 else "no-click"
            print(
                f"Action: {action_str}, Reward: {reward:.3f}, Hits: {self.env.total_hits}"
            )

        if terminated or truncated:
            self.reset_episode()

        self.update_visuals()
        return reward

    def draw_crosshair(self):
        """Draw aiming reticle at screen center"""
        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
        size = 15

        # Get aim quality for color
        agent_to_target = vec3.subtract(
            self.env.target.position, self.env.agent.position
        )
        distance = vec3.length(agent_to_target)
        target_yaw, target_pitch, _ = angles.vec_to_yaw_pitch_distance(agent_to_target)

        yaw_error = abs(angles.yaw_difference(self.env.agent.yaw, target_yaw))
        pitch_error = abs(angles.pitch_difference(self.env.agent.pitch, target_pitch))

        # Color based on aim quality and range
        aim_threshold_yaw = np.radians(15)
        aim_threshold_pitch = np.radians(15)
        in_range = distance <= 3.0

        good_aim = yaw_error <= aim_threshold_yaw and pitch_error <= aim_threshold_pitch
        invuln_ready = self.env.target.invulnerablility_ticks <= 0

        if good_aim and in_range and invuln_ready:
            color = (0, 255, 0)  # Green - READY TO CLICK
        elif good_aim and in_range:
            color = (255, 255, 0)  # Yellow - Good aim but invuln
        elif in_range:
            color = (255, 165, 0)  # Orange - In range but bad aim
        else:
            color = (255, 255, 255)  # White - Out of range

        # Draw crosshair
        pygame.draw.line(
            self.renderer.screen,
            color,
            (center_x - size, center_y),
            (center_x + size, center_y),
            2,
        )
        pygame.draw.line(
            self.renderer.screen,
            color,
            (center_x, center_y - size),
            (center_x, center_y + size),
            2,
        )

        # Draw center dot
        pygame.draw.circle(self.renderer.screen, color, (center_x, center_y), 3)

        # Draw hit indicator if recently hit
        if self.env.target.health < self.env.target.max_health:
            pygame.draw.circle(
                self.renderer.screen,
                (255, 0, 0),
                (center_x, center_y),
                size + 5,
                2,
            )

    def draw_gui(self):
        """Draw GUI information"""
        # Get current aiming info
        agent_to_target = vec3.subtract(
            self.env.target.position, self.env.agent.position
        )
        distance = vec3.length(agent_to_target)
        target_yaw, target_pitch, _ = angles.vec_to_yaw_pitch_distance(agent_to_target)

        yaw_error = angles.yaw_difference(self.env.agent.yaw, target_yaw)
        pitch_error = angles.pitch_difference(self.env.agent.pitch, target_pitch)

        # Determine if should click
        aim_threshold_yaw = np.radians(15)
        aim_threshold_pitch = np.radians(15)
        in_range = distance <= 3.0
        good_aim = (
            abs(yaw_error) <= aim_threshold_yaw
            and abs(pitch_error) <= aim_threshold_pitch
        )
        invuln_ready = self.env.target.invulnerablility_ticks <= 0
        should_click = good_aim and in_range and invuln_ready

        # Hit efficiency
        total_actions = self.env.total_hits + self.env.total_wasted_clicks
        efficiency = (
            (self.env.total_hits / total_actions * 100) if total_actions > 0 else 0.0
        )

        gui_text = [
            f"Mode: {self.mode.upper()} | Auto: {self.auto_step}",
            f"Last Action: {'CLICK ✓' if self.last_action == 1 else 'no-click'}",
            "",
            "Stats:",
            f"  Total Hits: {self.env.total_hits}",
            f"  Wasted Clicks: {self.env.total_wasted_clicks}",
            f"  Efficiency: {efficiency:.1f}%",
            "",
            "Target Status:",
            f"  Health: {self.env.target.health:.0f}/{self.env.target.max_health:.0f}",
            f"  Distance: {distance:.2f} blocks",
            f"  Invuln Ticks: {self.env.target.invulnerablility_ticks}",
            "",
            "Aim Quality:",
            f"  Yaw Error: {math.degrees(yaw_error):.1f}° (threshold: 15°)",
            f"  Pitch Error: {math.degrees(pitch_error):.1f}° (threshold: 15°)",
            f"  In Range: {'✓' if in_range else '✗'}",
            f"  Good Aim: {'✓' if good_aim else '✗'}",
            f"  Ready to Click: {'✓ YES' if should_click else '✗ NO'}",
            "",
            f"Step: {self.env.current_step}/{self.env.max_steps}",
            "",
            "Controls:",
            "1/2/3 = Random/AI/Manual mode",
            "Space = Click (manual mode)",
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
                        self.mode = "random"
                        print("Mode: Random Clicking")
                    elif event.key == pygame.K_2:
                        self.mode = "ai"
                        print("Mode: AI Clicking")
                    elif event.key == pygame.K_3:
                        self.mode = "manual"
                        print("Mode: Manual Clicking")
                    elif event.key == pygame.K_a:
                        self.auto_step = not self.auto_step
                        print(f"Auto-step: {self.auto_step}")
                    elif event.key == pygame.K_r:
                        self.reset_episode()
                        print("Episode reset")
                    elif event.key == pygame.K_SPACE:
                        self.manual_click = True
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        self.manual_click = False

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

            # Draw distance line
            agent_2d = self.cam.world_to_screen(self.env.agent.position)
            target_2d = self.cam.world_to_screen(self.env.target.position)
            if agent_2d and target_2d:
                pygame.draw.line(
                    self.renderer.screen, (100, 100, 100), agent_2d, target_2d, 1
                )

            # Draw crosshair
            self.draw_crosshair()

            # Draw GUI
            self.draw_gui()

            self.renderer.finish_frame()

        pygame.quit()


def main():
    """Main function to run visual clicking demo"""
    import sys

    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"Using model: {model_path}")
    else:
        print("No model specified. Use: python visual_clicking_demo.py [model_path]")
        print("Running with random clicking demo.")

    demo = VisualClickingDemo(model_path)
    demo.run()


if __name__ == "__main__":
    main()
