from simulator.objects import Entity
from helpers import vec3
import numpy as np
import math
from dataclasses import dataclass

GROUND_Y = 0.0

GRAVITY = 0.08
AIRDRAG = 0.9800000190734863
AIRBORNE_INERTIA = 0.91
AIRBORNE_ACCELERATION = 0.02


PLAYER_BASE_SPEED = 0.1
SPRINT_SPEED = 0.3
JUMP_VELOCITY = 0.42


NEGLIGIBLE_VELOCITY = 0.005
DEFAULT_SLIPPERINESS = 0.6
SPRINT_TIME_TRIGGER_COOLDOWN = 2


def move_entity(entity: Entity, dx: float, dy: float, dz: float):
    entity.position[0] += dx
    entity.position[1] += dy
    entity.position[2] += dz

    # simple ground collision
    into_ground = GROUND_Y - entity.position[1]
    if into_ground > 0:
        entity.position[1] += into_ground
        entity.velocity[1] = 0.0

    # on ground iff we are touching ground and moving downwards or stationary vertically
    entity.on_ground = into_ground >= 0 and dy <= 0


def apply_heading(entity: Entity, strafe: float, forward: float, multiplier: float):
    speed = (strafe * strafe + forward * forward) ** 0.5
    if speed < 0.01:
        return
    speed = multiplier / max(speed, 1)

    strafe *= speed
    forward *= speed

    forward_vec = vec3.from_yaw_pitch(entity.yaw, 0)  # pitch ignored for walking

    right_vec = np.array(
        [forward_vec[2], 0, -forward_vec[0]]
    )  # rotate forward 90° right
    # or: right = np.cross(forward, np.array([0,1,0]))

    entity.velocity += forward * forward_vec
    entity.velocity += right_vec * strafe
    # yaw_cos = math.cos(entity.yaw)
    # yaw_sin = math.sin(entity.yaw)

    # # used to be entity.velocity[0] += strafe * yaw_cos - forward * yaw_sin
    # entity.velocity[0] += forward * yaw_sin + strafe * yaw_cos
    # entity.velocity[2] += forward * yaw_cos + strafe * yaw_sin


def move_entity_with_heading(entity: Entity, strafe: float, forward: float):
    acceleration = AIRBORNE_ACCELERATION
    inertia = AIRBORNE_INERTIA

    if entity.on_ground:
        entity_speed_attribute = PLAYER_BASE_SPEED
        if entity.is_sprinting:
            entity_speed_attribute *= 1 + SPRINT_SPEED

        # inertia = DEFAULT_SLIPPERINESS * 0.91  # AIRBORNE_INERTIA?
        # acceleration = entity_speed_attribute * (
        #     0.1627714 / (inertia * inertia * inertia)
        # )
        # same math lowkey
        inertia = DEFAULT_SLIPPERINESS * 0.91
        slipperiness = DEFAULT_SLIPPERINESS
        acceleration = entity_speed_attribute * (
            0.216 / (slipperiness * slipperiness * slipperiness)
        )
        if acceleration < 0:
            acceleration = 0  # acceleration should not be negative

    apply_heading(entity, strafe, forward, acceleration)

    move_entity(entity, entity.velocity[0], entity.velocity[1], entity.velocity[2])

    if not entity.on_ground:  # maybe remove?
        entity.velocity[1] -= GRAVITY
        entity.velocity[1] *= AIRDRAG

    entity.velocity[0] *= inertia
    entity.velocity[2] *= inertia


def negate_small_velocities(entity: Entity):
    if abs(entity.velocity[0]) < NEGLIGIBLE_VELOCITY:
        entity.velocity[0] = 0.0
    if abs(entity.velocity[1]) < NEGLIGIBLE_VELOCITY:
        entity.velocity[1] = 0.0
    if abs(entity.velocity[2]) < NEGLIGIBLE_VELOCITY:
        entity.velocity[2] = 0.0


@dataclass
class InputState:
    """Input state for an entity"""

    w: bool = False
    a: bool = False
    s: bool = False
    d: bool = False
    sprint: bool = False
    space: bool = False
    click: bool = False
    yaw: float = 0.0
    pitch: float = 0.0

    def clone(self) -> "InputState":
        return InputState(
            w=self.w,
            a=self.a,
            s=self.s,
            d=self.d,
            sprint=self.sprint,
            space=self.space,
            click=self.click,
            yaw=self.yaw,
            pitch=self.pitch,
        )


def simulate(entity: Entity, controls: InputState):
    # update entity state based on controls
    entity.yaw = controls.yaw
    entity.pitch = controls.pitch
    entity.is_sprinting = controls.sprint

    negate_small_velocities(entity)

    strafe = 0.0
    forward = 0.0

    strafe = (float(controls.d) - float(controls.a)) * 0.98
    forward = (float(controls.w) - float(controls.s)) * 0.98

    if controls.space:
        if entity.on_ground:
            entity.velocity[1] = JUMP_VELOCITY
            if entity.is_sprinting and forward > 0:
                entity.velocity[0] += math.sin(entity.yaw) * 0.2
                entity.velocity[2] += math.cos(entity.yaw) * 0.2

    move_entity_with_heading(entity, strafe, forward)
