import math
import numpy as np
from helpers import vec3


class FirstPersonCamera:
    def __init__(self, width, height, fov=90):
        self.width = width
        self.height = height
        self.position = vec3.from_list([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.pitch = 0.0
        self.fov = fov

        # precomputed
        self.fov_tan = math.tan(math.radians(self.fov) / 2)
        self.world_up = vec3.from_list([0.0, 1.0, 0.0])
        self.up = self.world_up
        self.forward = vec3.from_list([0.0, 0.0, 1.0])
        self.right = vec3.from_list([1.0, 0.0, 0.0])

    def set_yaw_pitch(self, yaw, pitch):
        self.yaw = yaw
        self.pitch = pitch

        self.forward = vec3.from_yaw_pitch(self.yaw, self.pitch)
        self.right = vec3.cross(self.world_up, self.forward)
        self.right = vec3.normalize(self.right)
        self.up = vec3.cross(self.forward, self.right)
        self.up = vec3.normalize(self.up)

    def set_position(self, position):
        self.position = vec3.from_list(position)

    # def world_to_camera(self, world_pos):
    #     # transform world_pos into camera-local coordinates (x_right, y_up, z_forward)
    #     rel = np.array(world_pos, dtype=float) - self.pos

    #     # rotate by -yaw around Y axis
    #     cy = math.cos(-self.yaw)
    #     sy = math.sin(-self.yaw)
    #     rx = rel[0] * cy - rel[2] * sy
    #     rz = rel[0] * sy + rel[2] * cy

    #     # rotate by -pitch around X axis
    #     cp = math.cos(-self.pitch)
    #     sp = math.sin(-self.pitch)
    #     ry = rel[1] * cp - rz * sp
    #     rz = rel[1] * sp + rz * cp

    #     return np.array([rx, ry, rz], dtype=float)

    # def world_to_camera(self, world_pos):
    #     rel = np.array(world_pos, dtype=float) - self.pos

    #     up = np.array([0, 1, 0])

    #     forward = vec3.from_yaw_pitch(self.yaw, self.pitch)
    #     right = np.cross(up, forward)
    #     print(forward, right)
    #     right /= np.linalg.norm(right)

    #     # camera-space coordinates are dot-products onto basis
    #     cx = np.dot(rel, right)
    #     cy = np.dot(rel, up)
    #     cz = np.dot(rel, forward)

    #     return np.array([cx, cy, cz], dtype=float)

    def world_to_camera(self, world_pos):
        rel = np.array(world_pos, dtype=float) - self.position

        # up_world = np.array([0, 1, 0])

        # forward = vec3.from_yaw_pitch(self.yaw, self.pitch)

        # # RIGHT-HANDED: right = up × forward
        # right = np.cross(up_world, forward)
        # right /= np.linalg.norm(right)

        # # then up = forward × right
        # up = np.cross(forward, right)

        # dot products give camera space
        cx = np.dot(rel, self.right)
        cy = np.dot(rel, self.up)
        cz = np.dot(rel, self.forward)

        return np.array([cx, cy, cz], dtype=float)

    def camera_to_screen(self, cam_pos):
        x, y, z = cam_pos
        if z <= 0.001:
            return None
        f = (self.width / 2) / self.fov_tan
        sx = self.width / 2 + (x * f / z)
        sy = self.height / 2 - (y * f / z)
        return (int(sx), int(sy))

    def world_to_screen(self, world_pos):
        cam_pos = self.world_to_camera(world_pos)
        return self.camera_to_screen(cam_pos)

    # def camera_to_world(self, cam_pos):
    #     x, y, z = cam_pos

    #     up_world = np.array([0, 1, 0])

    #     forward = vec3.from_yaw_pitch(self.yaw, self.pitch)

    #     # RIGHT-HANDED: right = up × forward
    #     right = np.cross(up_world, forward)
    #     right /= np.linalg.norm(right)

    #     # then up = forward × right
    #     up = np.cross(forward, right)

    #     world_pos = self.position + right * x + up * y + forward * z

    #     return world_pos
