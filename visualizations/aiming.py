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
from models.base_model import BaseModel
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
        self.mode = "manual"  # "manual", "random", "ai"

        # Initialize agent (optional)
        self.agent = None
        if model_path:
            self.agent = BaseModel().load(model_path)

            self.mode = "ai"
            print(f"Loaded trained model: {model_path}")

        # Demo state
        self.state = None
        self.auto_step = True
        self.step_delay = 0.05  # seconds
        self.last_step_time = 0
        self.last_reward = 0.0

        # Mouse control
        self.mouse_captured = False
        self.just_captured = 0
        self.mouse_sensitivity = 0.003

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

        self.agent_entity.copy(self.env.agent)
        self.target_entity.copy(self.env.target)

        # Update camera position at agent's eye level
        agent_eye_pos = self.agent_entity.get_eye_position()
        self.cam.set_position(agent_eye_pos)
        self.cam.set_yaw_pitch(self.agent_entity.yaw, self.agent_entity.pitch)

        # Update crosshair color based on accuracy
        agent_to_target = vec3.subtract(
            self.target_entity.position, self.agent_entity.position
        )

        target_yaw, target_pitch, _ = angles.vec_to_yaw_pitch_distance(agent_to_target)

        yaw_error = abs(angles.yaw_difference(self.agent_entity.yaw, target_yaw))
        pitch_error = abs(
            angles.pitch_difference(self.agent_entity.pitch, target_pitch)
        )

        # Green when accurate, red when inaccurate
        if yaw_error < 0.05 and pitch_error < 0.05:  # ~3 degrees
            self.crosshair_color = (0, 255, 0)  # Green
        elif yaw_error < 0.1 and pitch_error < 0.1:  # ~6 degrees
            self.crosshair_color = (255, 255, 0)  # Yellow
        else:
            self.crosshair_color = (255, 255, 255)  # White

    def _get_manual_action(self):
        """Get manual action from mouse movement"""
        action = np.zeros(2, dtype=np.float32)

        if self.mouse_captured:
            dx, dy = pygame.mouse.get_rel()
            if dx != 0 or dy != 0:
                if self.just_captured > 0:
                    # Discard initial jump on capture
                    self.just_captured -= 1
                else:
                    action[0] = -dx * self.mouse_sensitivity  # dyaw
                    action[1] = dy * self.mouse_sensitivity  # dpitch
        else:
            # Fallback to arrow keys
            keyboard_sensitivity = 0.05
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = -keyboard_sensitivity
            if keys[pygame.K_RIGHT]:
                action[0] = keyboard_sensitivity
            if keys[pygame.K_UP]:
                action[1] = keyboard_sensitivity
            if keys[pygame.K_DOWN]:
                action[1] = -keyboard_sensitivity

        return action

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
            raise ValueError(f"Unknown mode: {self.mode}")

    def step_environment(self):
        """Take one step in the environment"""
        action = self.get_next_action()
        self.state, reward, terminated, truncated, _ = self.env.step(action)
        self.last_reward = reward

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
            f"Last Reward: {self.last_reward:.3f}",
            "",
            "Controls:",
            "1/2/3 = Manual/Random/AI mode",
            "Mouse: Left click to capture (manual)",
            "ESC = Release mouse",
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
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and self.mode == "manual":
                        if not self.mouse_captured:
                            self.capture_mouse()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_1:
                        self.mode = "manual"
                        print("Mode: Manual (mouse look)")
                    elif event.key == pygame.K_2:
                        self.mode = "random"
                        if self.mouse_captured:
                            self.release_mouse()
                        print("Mode: Random Aiming")
                    elif event.key == pygame.K_3:
                        self.mode = "ai"
                        if self.mouse_captured:
                            self.release_mouse()
                        print("Mode: AI Aiming")
                    elif event.key == pygame.K_ESCAPE:
                        if self.mouse_captured:
                            self.release_mouse()
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

            # Draw agent and target entities
            render_entity(self.agent_entity, self.renderer, self.cam)
            render_entity(self.target_entity, self.renderer, self.cam)

            # Draw crosshair
            self.draw_crosshair()

            # Draw GUI
            self.draw_gui()

            self.renderer.finish_frame()

        pygame.quit()

    def capture_mouse(self):
        """Capture mouse for manual FPS-style look"""
        self.mouse_captured = True
        self.just_captured = 2
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()  # reset relative movement

    def release_mouse(self):
        """Release mouse capture"""
        self.mouse_captured = False
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)


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
