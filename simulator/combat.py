import math
from helpers import vec3
from simulator.objects import Entity

DAMAGE_AMOUNT = 0
ATTACK_SLOWDOWN_FACTOR = 0.6
INVULNERABILITY_TICKS_AFTER_HIT = 10  # 20 ticks = 1 second so 10 ticks = 0.5 seconds

KNOCKBACK_STRENGTH = 0.4
SPRINTING_KNOCKBACK_STRENGTH = 1.0
VERTICAL_KNOCKBACK = 0.4


def line_intersects_aabb(
    start: vec3.VEC3, direction: vec3.VEC3, aabb_min: vec3.VEC3, aabb_max: vec3.VEC3
) -> tuple[bool, float]:
    """
    Check if a ray starting at 'start' and going in 'direction' intersects the AABB defined by aabb_min and aabb_max.
    Returns (hit: bool, distance: float)
    """
    tmin = (
        (aabb_min[0] - start[0]) / direction[0] if direction[0] != 0 else float("-inf")
    )
    tmax = (
        (aabb_max[0] - start[0]) / direction[0] if direction[0] != 0 else float("inf")
    )
    if tmin > tmax:
        tmin, tmax = tmax, tmin

    tymin = (
        (aabb_min[1] - start[1]) / direction[1] if direction[1] != 0 else float("-inf")
    )
    tymax = (
        (aabb_max[1] - start[1]) / direction[1] if direction[1] != 0 else float("inf")
    )
    if tymin > tymax:
        tymin, tymax = tymax, tymin

    if (tmin > tymax) or (tymin > tmax):
        return (False, float("inf"))

    if tymin > tmin:
        tmin = tymin
    if tymax < tmax:
        tmax = tymax

    tzmin = (
        (aabb_min[2] - start[2]) / direction[2] if direction[2] != 0 else float("-inf")
    )
    tzmax = (
        (aabb_max[2] - start[2]) / direction[2] if direction[2] != 0 else float("inf")
    )
    if tzmin > tzmax:
        tzmin, tzmax = tzmax, tzmin

    if (tmin > tzmax) or (tzmin > tmax):
        return (False, float("inf"))

    if tzmin > tmin:
        tmin = tzmin
    if tzmax < tmax:
        tmax = tzmax

    if tmax < 0:
        return (False, float("inf"))

    return (True, tmin if tmin >= 0 else tmax)


def raycast(attacker: Entity, entities: list[Entity]):
    """
    Perform a raycast from the attacker's eye position in the look direction,
    up to the attacker's reach distance. Return the first entity hit, or None.
    """
    eye_pos = attacker.get_eye_position()
    look_dir = vec3.from_yaw_pitch(attacker.yaw, attacker.pitch)

    closest_entity = None
    closest_dist = float("inf")

    for entity in entities:
        if entity.object_id == attacker.object_id:
            continue  # skip self
        aabb_min, aabb_max = entity.get_min_max_aabb()
        hit, dist = line_intersects_aabb(eye_pos, look_dir, aabb_min, aabb_max)
        if hit and dist < closest_dist and dist <= attacker.reach:
            closest_dist = dist
            closest_entity = entity

    return closest_entity


def apply_knockback(
    target: Entity, ratio_x: float, ratio_z: float, is_sprint_hit: bool
):
    """
    Apply knockback to the target entity based on the attack direction and whether the attacker is sprinting.
    """
    strength = SPRINTING_KNOCKBACK_STRENGTH if is_sprint_hit else KNOCKBACK_STRENGTH

    f = math.sqrt(ratio_x * ratio_x + ratio_z * ratio_z)
    if f >= 0.0001:
        target.velocity[0] /= 2.0
        target.velocity[1] /= 2.0
        target.velocity[2] /= 2.0

        target.velocity[0] += (ratio_x / f) * strength
        target.velocity[2] += (ratio_z / f) * strength

        target.velocity[1] += VERTICAL_KNOCKBACK
        if target.velocity[1] > VERTICAL_KNOCKBACK:
            target.velocity[1] = VERTICAL_KNOCKBACK


def try_attack(attacker: Entity, entities: list[Entity]) -> bool:
    """
    Attempt to attack an entity within reach. Returns True if an entity was hit.
    """
    target = raycast(attacker, entities)
    if target is None:
        return False

    if target.invulnerablility_ticks > 0:
        # hit slowdown and sprint cancel
        attacker.velocity[0] *= ATTACK_SLOWDOWN_FACTOR
        attacker.velocity[2] *= ATTACK_SLOWDOWN_FACTOR
        attacker.is_sprinting = False
        return False

    target.health -= DAMAGE_AMOUNT
    target.invulnerablility_ticks = INVULNERABILITY_TICKS_AFTER_HIT

    ratio_x = target.position[0] - attacker.position[0]
    ratio_z = target.position[2] - attacker.position[2]

    apply_knockback(target, ratio_x, ratio_z, attacker.is_sprinting)

    # hit slowdown and sprint cancel
    attacker.velocity[0] *= ATTACK_SLOWDOWN_FACTOR
    attacker.velocity[2] *= ATTACK_SLOWDOWN_FACTOR
    attacker.is_sprinting = False

    return True
