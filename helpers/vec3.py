import numpy as np
import numpy.typing as npt
from typing import TypeAlias
import math

# A 3-component float vector
VEC3: TypeAlias = npt.NDArray[np.float64]


# --- constructors -------------------------------------------------------------


def zero() -> VEC3:
    return np.array([0.0, 0.0, 0.0], dtype=float)


def from_list(lst: list[float]) -> VEC3:
    return np.array(lst, dtype=float)


def to_list(v: VEC3) -> list[float]:
    return v.tolist()


def from_yaw_pitch(yaw: float, pitch: float) -> VEC3:
    """
    Create a forward-looking unit vector from yaw and pitch angles (in radians).
    Yaw=0 means facing +Z, pitch=0 means level.
    """
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    x = cp * sy  # yaw right means +X
    y = sp  # pitch up means +Y
    z = cp * cy  # yaw=0 means facing +Z
    return normalize(from_list([x, y, z]))


# --- basic arithmetic ---------------------------------------------------------


def add(v1: VEC3, v2: VEC3) -> VEC3:
    "Returns a new vector that is the sum of v1 and v2"
    return v1 + v2


def subtract(v1: VEC3, v2: VEC3) -> VEC3:
    "Returns a new vector that is the difference of v1 and v2"
    return v1 - v2


def scale(v: VEC3, scalar: float) -> VEC3:
    "Returns a new vector that is v scaled by the given scalar"
    return v * scalar


# --- vector math --------------------------------------------------------------


def dot(v1: VEC3, v2: VEC3) -> float:
    return float(np.dot(v1, v2))


def cross(v1: VEC3, v2: VEC3) -> VEC3:
    return np.cross(v1, v2)  # type: ignore


def length(v: VEC3) -> float:
    return float(np.linalg.norm(v))


def normalize(v: VEC3) -> VEC3:
    "Returns a new vector that is the normalized version of v"
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def distance(v1: VEC3, v2: VEC3) -> float:
    return float(np.linalg.norm(v1 - v2))


def distance_squared(v1: VEC3, v2: VEC3) -> float:
    diff = v1 - v2
    return float(np.dot(diff, diff))


def copy(v: VEC3) -> VEC3:
    "Returns a new vector that is a copy of v"
    return np.copy(v)
