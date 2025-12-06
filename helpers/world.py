from helpers import vec3


def yaw_pitch_to_basis_vectors(
    yaw: float, pitch: float
) -> tuple[vec3.VEC3, vec3.VEC3, vec3.VEC3]:
    """Convert yaw and pitch angles to world forward, right, and up vectors."""
    forward = vec3.from_yaw_pitch(yaw, pitch)
    up_world = vec3.from_list([0, 1, 0])

    # RIGHT-HANDED: right = up × forward
    right = vec3.cross(up_world, forward)
    right = vec3.normalize(right)

    # then up = forward × right
    up = vec3.cross(forward, right)
    up = vec3.normalize(up)

    return forward, right, up


def world_to_local(
    world_pos: vec3.VEC3, forward: vec3.VEC3, right: vec3.VEC3, up: vec3.VEC3
) -> vec3.VEC3:
    """Convert a world position to local coordinates given basis vectors."""
    return vec3.from_list(
        [
            vec3.dot(world_pos, right),
            vec3.dot(world_pos, up),
            vec3.dot(world_pos, forward),
        ]
    )
