# rendering/objects.py

import pygame
import math
import numpy as np
from .core import RenderObject
from typing import TYPE_CHECKING
from helpers import vec3

if TYPE_CHECKING:
    from simulator.objects import Entity, RectPrism
    from renderer.cameras import FirstPersonCamera
    from renderer.renderer import Renderer3D


def render_rect_prism(
    rect_prism: "RectPrism",
    renderer: "Renderer3D",
    cam: "FirstPersonCamera",
    positionIsXZCenter=False,
):
    x, y, z = rect_prism.position
    w, d, h = rect_prism.width, rect_prism.depth, rect_prism.height
    color = rect_prism.color

    if positionIsXZCenter:
        x -= w / 2
        z -= d / 2

    corners = [
        (x, y, z),
        (x + w, y, z),
        (x + w, y, z + d),
        (x, y, z + d),
        (x, y + h, z),
        (x + w, y + h, z),
        (x + w, y + h, z + d),
        (x, y + h, z + d),
    ]

    # Project
    pts = []
    for c in corners:
        cam_pos = cam.world_to_camera(c)
        sp = cam.camera_to_screen(cam_pos)
        pts.append(sp)

    # If any face is None, skip
    if any(p is None for p in pts):
        return

    # Draw edges (wireframe)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # bottom
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # top
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # vertical
    ]

    for a, b in edges:
        pygame.draw.line(renderer.screen, color, pts[a], pts[b], 2)


def render_entity(entity: "Entity", renderer: "Renderer3D", cam: "FirstPersonCamera"):
    render_rect_prism(entity, renderer, cam, positionIsXZCenter=True)
    eye_pos_x = entity.position[0]
    eye_pos_y = entity.position[1] + entity.eye_height
    eye_pos_z = entity.position[2]

    # end_x = eye_pos_x + math.sin(entity.yaw)
    # end_y = eye_pos_y + math.sin(entity.pitch)
    # end_z = eye_pos_z + math.cos(entity.yaw)

    look_vector_xyz = vec3.from_yaw_pitch(entity.yaw, entity.pitch)
    end_x = eye_pos_x + look_vector_xyz[0]
    end_y = eye_pos_y + look_vector_xyz[1]
    end_z = eye_pos_z + look_vector_xyz[2]

    start_cam = cam.camera_to_screen(
        cam.world_to_camera((eye_pos_x, eye_pos_y, eye_pos_z))
    )
    end_cam = cam.camera_to_screen(cam.world_to_camera((end_x, end_y, end_z)))
    if start_cam and end_cam:
        pygame.draw.line(renderer.screen, (255, 255, 0), start_cam, end_cam, 3)
