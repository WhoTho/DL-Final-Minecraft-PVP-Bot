# visual_combined_demo.py
"""
Visual combined demo - full PvP combat with movement + aiming + clicking
"""

import time
import pygame
import math
import numpy as np
from renderer.cameras import FirstPersonCamera
from renderer.renderer import Renderer3D
from renderer.objects import render_entity
from simulator.objects import Entity
from environments.combined.environment import CombinedEnv
from models.combined.baseline_model import CombinedModel
from helpers import vec3, angles


class VisualCombinedDemo:
    """Visual demo of the combined PvP environment"""

    def __init__(self, model_path=None):
        self.WIDTH, self.HEIGHT = 1200, 800

        # Initialize renderer and camera
        self.renderer = Renderer3D(self.WIDTH, self.HEIGHT, "PvP Combat Demo")
        self.cam = FirstPersonCamera(self.WIDTH, self.HEIGHT)

        # Initialize environment
        self.env = CombinedEnv()
        self.mode = "random"  # "random", "ai", "manual"

        # Initialize agent (optional)
        self.agent = None
        if model_path:
            try:
                self.agent = CombinedModel()
                self.agent.load(model_path)
                self.mode = "ai"
                print(f"Loaded trained model: {model_path}")
            except FileNotFoundError:
                print(f"Model {model_path} not found. Using random actions.")

        # Demo state
        self.state = None
        self.auto_step = True
        self.step_delay = 0.05  # seconds
        self.last_step_time = 0

        # Manual controls
        self.keys_pressed = set()
        self.mouse_captured = False
        self.just_captured = 0
        self.manual_click = False

        # Mouse control setup
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)

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

        # Camera mode
        self.camera_mode = "first_person"  # "first_person" or "third_person"
        self.camera_distance = 8.0
        self.camera_height = 3.0

        # Combat stats
        self.episode_stats = {
            "hits_dealt": 0,
            "hits_taken": 0,
            "total_reward": 0.0,
        }

    def reset_episode(self):
        """Reset the environment for a new episode"""
        self.state, _ = self.env.reset()
        self.episode_stats = {
            "hits_dealt": 0,
            "hits_taken": 0,
            "total_reward": 0.0,
        }
        self.update_visuals()

    def update_visuals(self):
        """Update visual elements based on current environment state"""
        # Update entity positions and orientations
        self.agent_entity.position = self.env.agent.position
        self.agent_entity.yaw = self.env.agent.yaw
        self.agent_entity.pitch = self.env.agent.pitch

        self.target_entity.position = self.env.target.position
        self.target_entity.yaw = self.env.target.yaw
        self.target_entity.pitch = self.env.target.pitch

        # Update camera
        if self.camera_mode == "first_person":
            # First person view from agent's eyes
            agent_eye_pos = vec3.add(
                self.env.agent.position, vec3.from_list([0, 1.62, 0])
            )
            self.cam.set_position(agent_eye_pos)
            self.cam.set_yaw_pitch(self.env.agent.yaw, self.env.agent.pitch)
        else:
            # Third person view behind and above agent
            offset_dir = vec3.from_yaw_pitch(self.env.agent.yaw + math.pi, 0.3)
            offset = vec3.scale(offset_dir, self.camera_distance)
            camera_pos = vec3.add(
                self.env.agent.position,
                vec3.add(offset, vec3.from_list([0, self.camera_height, 0])),
            )
            self.cam.set_position(camera_pos)

            # Look at agent
            look_dir = vec3.subtract(self.env.agent.position, camera_pos)
            if vec3.length(look_dir) > 1e-6:
                yaw, pitch, _ = angles.vec_to_yaw_pitch_distance(look_dir)
                self.cam.set_yaw_pitch(yaw, pitch)

    def get_next_action(self):
        """Get next action based on current mode"""
        if self.mode == "manual":
            return self._get_manual_action()
        elif self.mode == "random":
            return self.env.action_space.sample()
        elif self.mode == "ai" and self.agent:
            action = self.agent.predict(self.state, deterministic=True)
            return action
        else:
            return np.zeros(9, dtype=np.float32)

    def _get_manual_action(self):
        """Get action from keyboard and mouse input"""
        # Action: [w, a, s, d, space, sprint, click, dyaw, dpitch]
        action = np.zeros(9, dtype=np.float32)

        # Movement keys
        action[0] = 1.0 if pygame.K_w in self.keys_pressed else 0.0
        action[1] = 1.0 if pygame.K_a in self.keys_pressed else 0.0
        action[2] = 1.0 if pygame.K_s in self.keys_pressed else 0.0
        action[3] = 1.0 if pygame.K_d in self.keys_pressed else 0.0
        action[4] = 1.0 if pygame.K_SPACE in self.keys_pressed else 0.0
        action[5] = 1.0 if pygame.K_LSHIFT in self.keys_pressed else 0.0

        # Click action (attack)
        action[6] = 1.0 if self.manual_click else 0.0
        if self.manual_click:
            self.manual_click = False  # Reset after using

        # Mouse look (only if captured)
        if self.mouse_captured:
            dx, dy = pygame.mouse.get_rel()
            if dx != 0 or dy != 0:
                if self.just_captured > 0:
                    # Discard large jump on capture
                    self.just_captured -= 1
                else:
                    mouse_sensitivity = 0.0025
                    action[7] = -dx * mouse_sensitivity
                    action[8] = dy * mouse_sensitivity
        else:
            # Fallback to arrow keys if mouse not captured
            keyboard_sensitivity = 0.05
            if pygame.K_LEFT in self.keys_pressed:
                action[7] = -keyboard_sensitivity
            if pygame.K_RIGHT in self.keys_pressed:
                action[7] = keyboard_sensitivity
            if pygame.K_UP in self.keys_pressed:
                action[8] = keyboard_sensitivity
            if pygame.K_DOWN in self.keys_pressed:
                action[8] = -keyboard_sensitivity

        return action

    def capture_mouse(self):
        """Capture mouse for FPS-style control"""
        self.mouse_captured = True
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        self.just_captured = 2

    def release_mouse(self):
        """Release mouse capture"""
        self.mouse_captured = False
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        self.just_captured = 0

    def step_environment(self):
        """Take one step in the environment"""
        prev_agent_health = self.env.agent.health
        prev_target_health = self.env.target.health

        action = self.get_next_action()
        self.state, reward, terminated, truncated, info = self.env.step(action)

        # Update combat stats
        self.episode_stats["total_reward"] += reward
        if self.env.agent.health < prev_agent_health:
            self.episode_stats["hits_taken"] += 1
        if self.env.target.health < prev_target_health:
            self.episode_stats["hits_dealt"] += 1

        if not self.auto_step:
            print(
                f"Reward: {reward:.3f}, Total: {self.episode_stats['total_reward']:.1f}"
            )

        if terminated or truncated:
            winner = (
                "AGENT" if self.env.agent.health > self.env.target.health else "BOT"
            )
            print(f"\n{'='*60}")
            print(f"EPISODE ENDED - Winner: {winner}")
            print(
                f"Agent Health: {self.env.agent.health:.1f}/{self.env.agent.max_health:.1f}"
            )
            print(
                f"Target Health: {self.env.target.health:.1f}/{self.env.target.max_health:.1f}"
            )
            print(f"Hits Dealt: {self.episode_stats['hits_dealt']}")
            print(f"Hits Taken: {self.episode_stats['hits_taken']}")
            print(f"Total Reward: {self.episode_stats['total_reward']:.1f}")
            print(f"{'='*60}\n")
            self.reset_episode()

        self.update_visuals()
        return reward

    def draw_health_bars(self):
        """Draw health bars for both entities"""
        # Agent health bar (bottom center)
        bar_width = 200
        bar_height = 20
        bar_x = self.WIDTH // 2 - bar_width // 2
        bar_y = self.HEIGHT - 40

        # Background
        pygame.draw.rect(
            self.renderer.screen,
            (50, 50, 50),
            (bar_x, bar_y, bar_width, bar_height),
        )

        # Health
        agent_health_ratio = self.env.agent.health / self.env.agent.max_health
        health_width = int(bar_width * agent_health_ratio)
        pygame.draw.rect(
            self.renderer.screen,
            (0, 255, 0),
            (bar_x, bar_y, health_width, bar_height),
        )

        # Border
        pygame.draw.rect(
            self.renderer.screen,
            (255, 255, 255),
            (bar_x, bar_y, bar_width, bar_height),
            2,
        )

        # Text
        font = pygame.font.Font(None, 24)
        text = font.render(
            f"Agent: {self.env.agent.health:.0f}/{self.env.agent.max_health:.0f}",
            True,
            (255, 255, 255),
        )
        self.renderer.screen.blit(text, (bar_x + bar_width + 10, bar_y))

        # Target health bar (top center)
        bar_y = 20

        # Background
        pygame.draw.rect(
            self.renderer.screen,
            (50, 50, 50),
            (bar_x, bar_y, bar_width, bar_height),
        )

        # Health
        target_health_ratio = self.env.target.health / self.env.target.max_health
        health_width = int(bar_width * target_health_ratio)
        pygame.draw.rect(
            self.renderer.screen,
            (255, 0, 0),
            (bar_x, bar_y, health_width, bar_height),
        )

        # Border
        pygame.draw.rect(
            self.renderer.screen,
            (255, 255, 255),
            (bar_x, bar_y, bar_width, bar_height),
            2,
        )

        # Text
        text = font.render(
            f"Target: {self.env.target.health:.0f}/{self.env.target.max_health:.0f}",
            True,
            (255, 255, 255),
        )
        self.renderer.screen.blit(text, (bar_x + bar_width + 10, bar_y))

    def draw_crosshair(self):
        """Draw aiming crosshair"""
        if self.camera_mode != "first_person":
            return

        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
        size = 10

        # Color based on target in crosshair
        agent_to_target = vec3.subtract(
            self.env.target.position, self.env.agent.position
        )
        target_yaw, target_pitch, distance = angles.vec_to_yaw_pitch_distance(
            agent_to_target
        )

        yaw_error = abs(angles.yaw_difference(self.env.agent.yaw, target_yaw))
        pitch_error = abs(angles.pitch_difference(self.env.agent.pitch, target_pitch))

        if yaw_error < 0.1 and pitch_error < 0.1 and distance <= 3.0:
            color = (255, 0, 0)  # Red - on target
        else:
            color = (255, 255, 255)  # White - default

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

    def draw_gui(self):
        """Draw GUI information"""
        distance = vec3.distance(self.env.agent.position, self.env.target.position)

        gui_text = [
            f"Mode: {self.mode.upper()} | Camera: {self.camera_mode} | Auto: {self.auto_step}",
            "",
            f"Step: {self.env.current_step}/{self.env.max_steps}",
            f"Distance: {distance:.2f}m",
            "",
            "Combat Stats:",
            f"  Hits Dealt: {self.episode_stats['hits_dealt']}",
            f"  Hits Taken: {self.episode_stats['hits_taken']}",
            f"  Total Reward: {self.episode_stats['total_reward']:.1f}",
            "",
            "Agent Invuln: "
            + ("✓" if self.env.agent.invulnerablility_ticks > 0 else "✗"),
            "Target Invuln: "
            + ("✓" if self.env.target.invulnerablility_ticks > 0 else "✗"),
            "",
            "Controls:",
            "1/2/3 = Random/AI/Manual",
            "C = Toggle camera",
            "P = Toggle auto-step",
            "R = Reset",
            "ESC = Release mouse",
            "Q = Quit",
            "",
            "Manual Mode:",
            "WASD = Move",
            "SPACE = Jump",
            "SHIFT = Sprint",
            "Mouse = Look (click to capture)",
            "Left Click = Attack",
            "Arrows = Look (fallback)",
        ]

        self.renderer.draw_gui_lines(gui_text, (10, 60), color=(255, 255, 255))

    def run(self):
        """Main demo loop"""
        running = True
        self.reset_episode()

        while running:
            current_time = pygame.time.get_ticks() / 1000.0

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        if not self.mouse_captured and self.mode == "manual":
                            self.capture_mouse()
                        elif self.mouse_captured:
                            # Clicking in manual mode - will trigger attack in environment
                            self.manual_click = True
                elif event.type == pygame.KEYDOWN:
                    self.keys_pressed.add(event.key)

                    if event.key == pygame.K_ESCAPE:
                        if self.mouse_captured:
                            self.release_mouse()
                    elif event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_1:
                        self.mode = "random"
                        self.release_mouse()
                        print("Mode: Random Actions")
                    elif event.key == pygame.K_2:
                        self.mode = "ai"
                        self.release_mouse()
                        print("Mode: AI Control")
                    elif event.key == pygame.K_3:
                        self.mode = "manual"
                        print("Mode: Manual Control")
                        print("Click to capture mouse for aiming")
                    elif event.key == pygame.K_c:
                        self.camera_mode = (
                            "third_person"
                            if self.camera_mode == "first_person"
                            else "first_person"
                        )
                        print(f"Camera: {self.camera_mode}")
                        self.update_visuals()
                    elif event.key == pygame.K_p:
                        self.auto_step = not self.auto_step
                        print(f"Auto-step: {self.auto_step}")
                    elif event.key == pygame.K_r:
                        self.reset_episode()
                        print("Episode reset")
                elif event.type == pygame.KEYUP:
                    self.keys_pressed.discard(event.key)

            # Auto-step if enabled
            if self.auto_step and current_time - self.last_step_time > self.step_delay:
                self.step_environment()
                self.last_step_time = current_time

            # Render
            self.renderer.begin_frame()
            self.renderer.draw_ground_grid(self.cam)

            # Draw entities
            render_entity(self.agent_entity, self.renderer, self.cam)
            render_entity(self.target_entity, self.renderer, self.cam)

            # Draw health bars
            self.draw_health_bars()

            # Draw crosshair (first person only)
            self.draw_crosshair()

            # Draw GUI
            self.draw_gui()

            self.renderer.finish_frame()

        pygame.quit()


def main():
    """Main function to run visual combined demo"""
    import sys

    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"Using model: {model_path}")
    else:
        print("No model specified. Use: python visual_combined_demo.py [model_path]")
        print("Running with random actions.")

    demo = VisualCombinedDemo(model_path)
    demo.run()


if __name__ == "__main__":
    main()
