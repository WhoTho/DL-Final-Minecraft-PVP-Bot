# visual_movement_demo.py
"""
Visual movement demo using the existing game renderer
"""

import time
import pygame
import math
import numpy as np
from renderer.cameras import FirstPersonCamera
from renderer.renderer import Renderer3D
from renderer.objects import render_entity
from simulator.objects import Entity
from environments.movement.environment import MovementEnv
from helpers import vec3
from helpers.angles import vec_to_yaw_pitch_distance
from models.base_model import BaseModel


class VisualMovementDemo:
    """Visual demo of the movement environment using the game renderer"""

    def __init__(self, model_path=None):
        self.WIDTH, self.HEIGHT = 1200, 800

        # Initialize renderer and camera
        self.renderer = Renderer3D(self.WIDTH, self.HEIGHT, "Movement Training Demo")
        self.cam = FirstPersonCamera(self.WIDTH, self.HEIGHT)

        # Initialize environment
        self.env = MovementEnv(render_mode="human")
        self.mode = "manual"  # "manual", "random", "ai"

        # Initialize agent (optional)
        self.agent = None
        if model_path:
            self.agent = BaseModel().load(model_path)
            self.mode = "ai"

        # Demo state
        self.state = None
        self.auto_step = True
        self.step_delay = 0.05  # seconds
        self.last_step_time = 0
        self.current_action = 0  # Current manual action

        # Visual elements
        self.agent_entity = Entity(
            object_id=998, position=vec3.zero(), color=(0, 255, 0)  # Green for agent
        )

        self.target_entity = Entity(
            object_id=999, position=vec3.zero(), color=(255, 0, 0)  # Red for target
        )

        # Camera settings
        self.camera_height = 10.0
        self.camera_distance = 10.0
        self.follow_agent = True

        # Manual controls state
        self.keys_pressed = set()

    def reset_episode(self):
        """Reset the environment for a new episode"""
        self.state, _ = self.env.reset()
        self.update_visuals()

    def update_visuals(self):
        """Update visual elements based on current environment state"""
        # Update entity positions
        self.agent_entity.position = self.env.agent.position
        self.target_entity.position = self.env.target.position

        # Update entity orientation (for visual reference)
        self.agent_entity.yaw = self.env.agent.yaw
        self.agent_entity.pitch = self.env.agent.pitch
        self.target_entity.yaw = self.env.target.yaw
        self.target_entity.pitch = self.env.target.pitch

        # Update camera to follow the action
        if self.follow_agent:
            # Position camera above and behind the midpoint between agent and target
            midpoint = vec3.scale(
                vec3.add(self.env.agent.position, self.env.target.position), 0.5
            )
            camera_pos = vec3.add(
                midpoint, vec3.from_list([0, self.camera_height, -self.camera_distance])
            )
            self.cam.set_position(camera_pos)

            # Look down at the midpoint
            look_direction = vec3.subtract(midpoint, camera_pos)
            if vec3.length(look_direction) > 1e-6:
                yaw, pitch, _ = vec_to_yaw_pitch_distance(look_direction)
                self.cam.set_yaw_pitch(yaw, pitch)
        else:
            agent_to_target = vec3.subtract(
                self.env.target.position, self.env.agent.position
            )
            yaw, pitch, _ = vec_to_yaw_pitch_distance(agent_to_target)
            self.env.agent.yaw = yaw
            self.env.agent.pitch = 0
            self.agent_entity.yaw = yaw
            self.agent_entity.pitch = 0
            # Fixed camera at agent position
            self.cam.set_position(
                vec3.add(
                    self.env.agent.position,
                    vec3.from_list([0, 1.62, 0]),
                )
            )
            self.cam.set_yaw_pitch(self.env.agent.yaw, self.env.agent.pitch)

    def get_next_action(self):
        """Get next action based on current mode"""
        if self.mode == "manual":
            return self._get_manual_action()
        elif self.mode == "random":
            return self.env.action_space.sample()
        elif self.mode == "ai" and self.agent:
            action = self.agent.predict(self.state, deterministic=True)
            return action
            # raise NotImplementedError("AI agent not yet implemented")
        else:
            return [False, False, False, False, False, False]  # No action

    def _get_manual_action(self):
        """Get action from keyboard input"""
        # Priority order: W, A, S, D, SPACE, SPRINT
        return [
            bool(pygame.K_w in self.keys_pressed),
            bool(pygame.K_a in self.keys_pressed),
            bool(pygame.K_s in self.keys_pressed),
            bool(pygame.K_d in self.keys_pressed),
            bool(pygame.K_SPACE in self.keys_pressed),
            bool(pygame.K_r in self.keys_pressed),
        ]

    def step_environment(self):
        """Take one step in the environment"""
        action = self.get_next_action()
        prev_state = self.state.copy() if self.state is not None else None
        self.state, reward, terminated, truncated, info = self.env.step(action)

        if not self.auto_step:
            actionString = (
                " ".join(
                    [
                        name
                        for name, pressed in zip(
                            ["W", "A", "S", "D", "SPACE", "SPRINT"], action
                        )
                        if pressed
                    ]
                )
                or "NOTHING"
            )
            print(f"Action: {actionString}, Reward: {reward:.3f}")
            print(f"Info: {info}")

        if terminated or truncated:
            print(f"Episode ended. Final reward: {reward:.3f}")
            print(f"Final info: {info}")
            self.reset_episode()

        self.update_visuals()
        return reward

    def draw_trajectory_lines(self):
        """Draw lines showing movement directions"""
        # Agent velocity vector
        agent_vel = self.env.agent.velocity
        if vec3.length(agent_vel) > 0.1:
            start_pos = self.env.agent.position
            end_pos = vec3.add(
                start_pos, vec3.scale(agent_vel, 3.0)
            )  # Scale for visibility

            start_2d = self.cam.world_to_screen(start_pos)
            end_2d = self.cam.world_to_screen(end_pos)

            if start_2d and end_2d:
                pygame.draw.line(self.renderer.screen, (0, 255, 0), start_2d, end_2d, 3)
                # Arrow head
                pygame.draw.circle(self.renderer.screen, (0, 255, 0), end_2d, 5)

        # Target velocity vector
        target_vel = self.env.target.velocity
        if vec3.length(target_vel) > 0.1:
            start_pos = self.env.target.position
            end_pos = vec3.add(start_pos, vec3.scale(target_vel, 3.0))

            start_2d = self.cam.world_to_screen(start_pos)
            end_2d = self.cam.world_to_screen(end_pos)

            if start_2d and end_2d:
                pygame.draw.line(self.renderer.screen, (255, 0, 0), start_2d, end_2d, 3)
                # Arrow head
                pygame.draw.circle(self.renderer.screen, (255, 0, 0), end_2d, 5)

        # Line connecting agent and target
        agent_2d = self.cam.world_to_screen(self.env.agent.position)
        target_2d = self.cam.world_to_screen(self.env.target.position)

        if agent_2d and target_2d:
            pygame.draw.line(
                self.renderer.screen, (100, 100, 100), agent_2d, target_2d, 1
            )

    # def draw_distance_circles(self):
    #     """Draw optimal distance range circles around target"""
    #     # Distance circles disabled - attributes no longer in environment
    #     pass

    # def draw_target_aim_cone(self):
    #     """Draw target's aim cone if in stage 3"""
    #     # Aim cone drawing disabled - curriculum_stage no longer in environment
    #     pass

    def draw_gui(self):
        """Draw GUI information"""
        distance = vec3.distance(self.env.agent.position, self.env.target.position)
        agent_speed = vec3.length(self.env.agent.velocity)
        target_speed = vec3.length(self.env.target.velocity)

        gui_text = [
            f"Mode: {self.mode.title()} | Auto: {self.auto_step}",
            "",
            f"Step: {self.env.current_step}/{self.env.max_steps}",
            f"Distance: {distance:.2f}m",
            f"Agent Speed: {agent_speed:.2f}",
            f"Target Speed: {target_speed:.2f}",
            "",
            f"Agent Pos: ({self.env.agent.position[0]:.1f}, {self.env.agent.position[2]:.1f})",
            f"Target Pos: ({self.env.target.position[0]:.1f}, {self.env.target.position[2]:.1f})",
            "",
            "Manual Controls:",
            "WASD = Move",
            "SPACE = Jump",
            "SHIFT+W = Sprint",
            "",
            "Demo Controls:",
            "1/2/3 = Manual/Random/AI mode",
            "F = Toggle follow agent",
            "TAB = Toggle auto-step",
            "R = Reset episode",
            "ENTER = Manual step",
            "Q = Quit",
            "",
            "Agent Input State:",
            f"         W: {'X' if self.env.agent_input.w else '-'}",
            f"  A: {'X' if self.env.agent_input.a else '-'} | S: {'X' if self.env.agent_input.s else '-'} | D: {'X' if self.env.agent_input.d else '-'}",
            f"  SPACE: {'X' if self.env.agent_input.space else '-'} | SPRINT: {'X' if self.env.agent_input.sprint else '-'}",
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
                    self.keys_pressed.add(event.key)

                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_1:
                        self.mode = "manual"
                        print("Mode: Manual Control")
                    elif event.key == pygame.K_2:
                        self.mode = "random"
                        print("Mode: Random Actions")
                    elif event.key == pygame.K_3:
                        self.mode = "ai"
                        print("Mode: AI Control")
                    elif event.key == pygame.K_f:
                        self.follow_agent = not self.follow_agent
                        print(f"Follow agent: {self.follow_agent}")
                    elif event.key == pygame.K_TAB:
                        self.auto_step = not self.auto_step
                        print(f"Auto-step: {self.auto_step}")
                    elif event.key == pygame.K_r:
                        self.reset_episode()
                        print(f"Episode reset")
                    elif event.key == pygame.K_RETURN:
                        if not self.auto_step:
                            reward = self.step_environment()
                            print(f"Manual step reward: {reward:.3f}")
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

            # Draw visual helpers
            self.draw_trajectory_lines()

            # Draw GUI
            self.draw_gui()

            self.renderer.finish_frame()

        pygame.quit()


def main():
    """Main function to run visual movement demo"""
    import sys

    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"Using model: {model_path}")
    else:
        print("No model specified. Use: python visual_movement_demo.py [model_path]")
        print("Running with manual/random controls only.")

    demo = VisualMovementDemo(model_path)
    demo.run()


if __name__ == "__main__":
    main()
