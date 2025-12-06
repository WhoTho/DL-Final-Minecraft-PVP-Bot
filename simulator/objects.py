import helpers.vec3 as vec3


class RectPrism:
    """
    A vertical rectangular prism, defined at (x,y,z) with given width, depth, and height.
    """

    def __init__(
        self,
        object_id: int,
        position: vec3.VEC3,
        width: float,
        depth: float,
        height: float,
        color=(200, 100, 50),
    ):
        self.object_id = object_id
        self.position = position
        self.width = width
        self.depth = depth
        self.height = height
        self.color = color

    def get_min_max_aabb(self) -> tuple[vec3.VEC3, vec3.VEC3]:
        """
        Returns the min and max corners of the axis-aligned bounding box (AABB) for this prism.
        """
        x, y, z = self.position
        w, d, h = self.width, self.depth, self.height
        aabb_min = vec3.from_list([x, y, z])
        aabb_max = vec3.from_list([x + w, y + h, z + d])
        return aabb_min, aabb_max


class Entity(RectPrism):
    """
    An entity in the world, with position, orientation, and size. (Used for players, NPCs, mobs, etc.)
    """

    def __init__(
        self,
        object_id: int,
        position: vec3.VEC3,
        velocity: vec3.VEC3 = vec3.from_list([0.0, 0.0, 0.0]),
        yaw: float = 0.0,
        pitch: float = 0.0,
        width: float = 0.6,
        depth: float = 0.6,
        height: float = 1.8,
        eye_height: float = 1.62,
        health: float = 20.0,
        max_health: float = 20.0,
        reach: float = 3.0,
        color=(200, 100, 50),
    ):
        super().__init__(object_id, position, width, depth, height, color)
        self.velocity = velocity
        self.yaw = yaw
        self.pitch = pitch
        self.eye_height = eye_height

        # Physics state
        self.on_ground = False
        self.is_sprinting = False

        # Combat state
        self.invulnerablility_ticks = 0
        self.health = health
        self.max_health = max_health
        self.reach = reach

    def get_min_max_aabb(self) -> tuple[vec3.VEC3, vec3.VEC3]:
        x, y, z = self.position
        w, d, h = self.width, self.depth, self.height
        half_w = w / 2
        half_d = d / 2
        aabb_min = vec3.from_list([x - half_w, y, z - half_d])
        aabb_max = vec3.from_list([x + half_w, y + h, z + half_d])
        return aabb_min, aabb_max

    def get_eye_position(self) -> vec3.VEC3:
        """
        Returns the world position of the entity's eyes.
        """
        return vec3.from_list(
            [
                self.position[0] + self.width / 2,
                self.position[1] + self.eye_height,
                self.position[2] + self.depth / 2,
            ]
        )


# class Player(RectPrism):
#     """
#     A player entity, subclass of RectPrism.
#     """

#     def __init__(
#         self,
#         position: vec3.VEC3,
#         yaw: float = 0.0,
#         pitch: float = 0.0,
#         width: float = 0.6,
#         depth: float = 0.6,
#         height: float = 1.8,
#         eye_height: float = 1.62,
#         color=(100, 100, 255),
#     ):
#         super().__init__(position, width, depth, height, color)
#         self.yaw = yaw
#         self.pitch = pitch
#         self.eye_height = eye_height
#         self.on_ground = False
#         self.is_sprinting = False
