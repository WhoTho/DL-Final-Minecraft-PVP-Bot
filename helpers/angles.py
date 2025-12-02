from typing import TYPE_CHECKING
import math

if TYPE_CHECKING:
    from helpers import vec3


def vec_to_yaw_pitch_distance(vec: "vec3.VEC3") -> tuple[float, float, float]:
    """
    Convert a direction vector to yaw and pitch angles (in radians), and return its length.
    Yaw=0 means facing +Z, pitch=0 means level.
    """
    x, y, z = vec
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0:
        return 0.0, 0.0, 0.0
    yaw = math.atan2(x, z)  # yaw right means +X
    pitch = math.asin(y / length)  # pitch up means +Y
    return yaw, pitch, length


def yaw_difference(start_yaw: float, end_yaw: float) -> float:
    """
    Compute the smallest difference between two yaw angles (in radians).
    The result is in the range [-pi, pi].
    """
    diff = end_yaw - start_yaw
    while diff < -math.pi:
        diff += 2 * math.pi
    while diff > math.pi:
        diff -= 2 * math.pi
    return diff


def pitch_difference(start_pitch: float, end_pitch: float) -> float:
    """
    Compute the smallest difference between two pitch angles (in radians).
    The result is in the range [-pi, pi].
    """
    diff = end_pitch - start_pitch
    while diff < -math.pi:
        diff += 2 * math.pi
    while diff > math.pi:
        diff -= 2 * math.pi
    return diff
