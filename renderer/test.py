import pygame
import math
import numpy as np


def draw_ground(screen, cam, half=200, color=(180, 160, 120)):
    cx, cy, cz = cam.position

    pts = [
        (cx - half, 0, cz - half),
        (cx + half, 0, cz - half),
        (cx + half, 0, cz + half),
        (cx - half, 0, cz + half),
    ]

    pts2 = []
    for p in pts:
        cp = cam.world_to_camera(p)
        sp = cam.camera_to_screen(cp)
        if sp is None:  # if any corner is behind camera, skip ground
            return
        pts2.append(sp)

    pygame.draw.polygon(screen, color, pts2)


EPS_NEAR = 0.01


def project_clipped_line_world(cam, w1, w2):
    """
    Project the visible portion of world-space segment [w1,w2] using cam.
    Returns (s1, s2) screen points or None if nothing visible.
    This:
     - computes c1/c2 = world_to_camera(w1/w2)
     - if both in front: project both
     - if both behind: return None
     - if crosses near plane: find t where camera_z = EPS_NEAR,
       build W_clip = W1 + t*(W2-W1), then project the visible endpoints
       via world_to_camera(world_point) -> camera_to_screen(cam_point)
    """
    c1 = cam.world_to_camera(w1)
    c2 = cam.world_to_camera(w2)

    z1 = float(c1[2])
    z2 = float(c2[2])

    # both behind
    if z1 < EPS_NEAR and z2 < EPS_NEAR:
        return None

    # both in front
    if z1 >= EPS_NEAR and z2 >= EPS_NEAR:
        s1 = cam.camera_to_screen(c1)
        s2 = cam.camera_to_screen(c2)
        if s1 is None or s2 is None:
            return None
        return s1, s2

    # crossing case: compute t where camera_z(t) = EPS_NEAR
    # camera_z(t) = z1 + t*(z2 - z1)  =>  t = (EPS - z1) / (z2 - z1)
    denom = z2 - z1
    if abs(denom) < 1e-12:
        # numerical fallback: segment nearly parallel to near plane
        # if either endpoint in front, project that tiny bit
        if z1 >= EPS_NEAR:
            s1 = cam.camera_to_screen(c1)
            if s1 is None:
                return None
            return s1, s1
        if z2 >= EPS_NEAR:
            s2 = cam.camera_to_screen(c2)
            if s2 is None:
                return None
            return s2, s2
        return None

    t = (EPS_NEAR - z1) / denom
    t = max(0.0, min(1.0, t))

    # build clipped world point and then re-run world->camera->screen
    w1 = np.array(w1, dtype=float)
    w2 = np.array(w2, dtype=float)
    W_clip = (1.0 - t) * w1 + t * w2

    # which side is visible?
    if z1 < EPS_NEAR and z2 >= EPS_NEAR:
        # part visible is [W_clip, w2]
        c_clip = cam.world_to_camera(tuple(W_clip))
        s_clip = cam.camera_to_screen(c_clip)
        s2 = cam.camera_to_screen(cam.world_to_camera(tuple(w2)))
        if s_clip is None or s2 is None:
            return None
        return s_clip, s2

    if z1 >= EPS_NEAR and z2 < EPS_NEAR:
        # visible is [w1, W_clip]
        s1 = cam.camera_to_screen(cam.world_to_camera(tuple(w1)))
        c_clip = cam.world_to_camera(tuple(W_clip))
        s_clip = cam.camera_to_screen(c_clip)
        if s1 is None or s_clip is None:
            return None
        return s1, s_clip

    # fallback
    return None


def draw_ground_grid(
    screen, cam, radius=30, spacing=1.0, color=(80, 80, 80), thickness=2
):

    cx, _, cz = cam.position

    # lines parallel to X (changing Z)
    for i in range(-radius, radius + 1):
        wz = math.floor(cz) + i * spacing

        p1 = (cx - radius * spacing, 0, wz)
        p2 = (cx + radius * spacing, 0, wz)

        seg = project_clipped_line_world(cam, p1, p2)
        if seg:
            pygame.draw.line(screen, color, seg[0], seg[1], thickness)

    # lines parallel to Z (changing X)
    for i in range(-radius, radius + 1):
        wx = math.floor(cx) + i * spacing

        p1 = (wx, 0, cz - radius * spacing)
        p2 = (wx, 0, cz + radius * spacing)

        seg = project_clipped_line_world(cam, p1, p2)
        if seg:
            pygame.draw.line(screen, color, seg[0], seg[1], thickness)
