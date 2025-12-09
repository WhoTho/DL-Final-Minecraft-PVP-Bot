# game_demo.py
import pygame
import math
from renderer.cameras import FirstPersonCamera
from renderer.renderer import Renderer3D
from renderer.objects import render_entity
from simulator.server import MinecraftSimulationServer
from simple_bot import SimpleBot


def main():
    # WIDTH, HEIGHT = 2000, 1400
    WIDTH, HEIGHT = 1000, 700
    center_x, center_y = WIDTH // 2, HEIGHT // 2

    renderer = Renderer3D(WIDTH, HEIGHT, "Minecraft PvP Demo")
    cam = FirstPersonCamera(WIDTH, HEIGHT)

    # Create server and add entities
    server = MinecraftSimulationServer()

    # Player 1 (human player) - red
    player1 = server.add_entity(1, 0, 0, 0, color=(255, 0, 0))

    # Player 2 (AI bot) - blue
    player2 = server.add_entity(2, 5, 0, 5, color=(0, 0, 255))

    # Create bot AI
    bot = SimpleBot(bot_id=2, target_id=1)

    # Some blocks for environment
    # blocks = [
    #     RectPrism(2, 0, 0, 1, 1, 1.5, (200, 50, 50)),
    #     RectPrism(-3, 0, 2, 1, 1, 2, (50, 200, 150)),
    #     RectPrism(8, 0, 8, 2, 2, 3, (100, 100, 200)),
    #     RectPrism(-5, 0, -5, 1, 1, 1, (200, 200, 50)),
    # ]

    # pygame.init()

    mouse_captured = False
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)

    # Helper functions for mouse capture
    def capture_mouse():
        nonlocal mouse_captured
        mouse_captured = True
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

    def release_mouse():
        nonlocal mouse_captured
        mouse_captured = False
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)

    running = True
    just_captured = 0
    # clock = pygame.time.Clock()

    # Game timing - run server at 20 TPS (50ms per tick)
    server_tick_time = server.tick_duration * 1000  # in milliseconds
    last_server_tick_at = pygame.time.get_ticks()
    last_time_at = last_server_tick_at
    time_accumulator = 0.0
    bot_status = None  # Store bot status for GUI display

    while running:
        current_time = pygame.time.get_ticks()
        time_accumulator += current_time - last_time_at
        last_time_at = current_time

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if not mouse_captured:
                        just_captured = 2
                        capture_mouse()
                    else:
                        # Send click input to server
                        server.take_input(1, {"click": True})
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    just_captured = 0
                    release_mouse()

        # Get current key states
        keys = pygame.key.get_pressed()
        keymap = {
            "w": keys[pygame.K_w],
            "s": keys[pygame.K_s],
            "a": keys[pygame.K_a],
            "d": keys[pygame.K_d],
            "q": keys[pygame.K_q],
            "space": keys[pygame.K_SPACE],
            "r": keys[pygame.K_r],
        }

        if keymap["q"]:
            running = False

        # Handle mouse look
        if mouse_captured:
            dx, dy = pygame.mouse.get_rel()
            if dx != 0 or dy != 0:
                if just_captured > 0:
                    # Discard large jump on capture
                    just_captured -= 1
                else:
                    # Update player1's yaw and pitch based on mouse movement
                    mouse_sensitivity = 0.0025

                    player1.yaw += -dx * mouse_sensitivity
                    player1.pitch += dy * mouse_sensitivity

                    # Clamp pitch
                    max_pitch = 1.45  # ~83 degrees
                    if player1.pitch > max_pitch:
                        player1.pitch = max_pitch
                    if player1.pitch < -max_pitch:
                        player1.pitch = -max_pitch

        # Prepare input for server
        input_map = {
            "w": keymap["w"],
            "a": keymap["a"],
            "s": keymap["s"],
            "d": keymap["d"],
            "sprint": keymap["r"],
            "space": keymap["space"],
            "yaw": player1.yaw,
            "pitch": player1.pitch,
        }

        # Send input to server for player1
        server.take_input(1, input_map)

        while time_accumulator >= server_tick_time:
            time_accumulator -= server_tick_time

            # Update bot AI before server tick
            bot_status = bot.update(server)

            server.step()
            last_server_tick_at = current_time

        # Update camera to match player1's position and orientation
        # Camera position should be at eye level
        eye_pos = player1.position.copy()
        eye_pos[1] += player1.eye_height  # Add eye height
        cam.set_position(eye_pos)
        cam.set_yaw_pitch(player1.yaw, player1.pitch)

        # Get entities for rendering (excluding player1 since it's the camera)
        render_entities = list(server.entities.values())

        # Rendering
        renderer.begin_frame()
        renderer.draw_ground_grid(cam)

        # Draw GUI info
        player1_info = server.get_entity_info(1)
        player2_info = server.get_entity_info(2)

        gui_text = []
        if player1_info:
            gui_text.extend(
                [
                    f"Player 1 - Health: {player1_info['health']:.1f} Pos: ({player1_info['pos'][0]:.1f}, {player1_info['pos'][1]:.1f}, {player1_info['pos'][2]:.1f}) Yaw: {player1_info['yaw']:.2f} Pitch: {player1_info['pitch']:.2f}",
                    f"Velocity: ({player1_info['velocity'][0]:.2f}, {player1_info['velocity'][1]:.2f}, {player1_info['velocity'][2]:.2f}) On Ground: {player1_info['on_ground']}",
                ]
            )

        if player2_info:
            if player1_info:
                distance = math.sqrt(
                    sum(
                        (a - b) ** 2
                        for a, b in zip(player1_info["pos"], player2_info["pos"])
                    )
                )
                gui_text.append(
                    f"Bot - Health: {player2_info['health']:.1f} Distance: {distance:.1f}"
                )
            else:
                gui_text.append(f"Bot - Health: {player2_info['health']:.1f}")

            # Add bot AI status if available
            if bot_status:
                gui_text.append(
                    f"Bot AI - Moving: {bot_status['moving']} Attacking: {bot_status['attacking']} Distance: {bot_status['distance']:.1f}"
                )

        gui_text.extend(
            [
                f"FPS: {int(renderer.clock.get_fps())} | Server Tick: {server.tick_count}",
                f"Controls: WASD=move, Space=jump, Mouse=look, Click=attack, Esc=release mouse, Q=quit",
            ]
        )

        renderer.draw_gui_lines(gui_text, (10, 10))

        # Draw blocks
        # for block in blocks:
        #     block.draw(renderer, cam)

        # Draw other entities (not player1)
        for entity in render_entities:
            if entity.object_id != 1:
                render_entity(entity, renderer, cam)

        renderer.draw_crosshair()
        renderer.finish_frame()

    pygame.quit()


if __name__ == "__main__":
    main()
