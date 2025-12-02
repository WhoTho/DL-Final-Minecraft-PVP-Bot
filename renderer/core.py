# rendering/core.py


class RenderObject:
    """
    Base class for drawable objects.
    Each object implements draw(self, renderer, camera).
    """

    def draw(self, renderer, camera):
        raise NotImplementedError


class Scene:
    """
    Holds all objects and updates/draws them.
    """

    def __init__(self):
        self.objects = []

    def add(self, obj):
        self.objects.append(obj)

    def clear(self):
        self.objects = []

    def draw(self, renderer, camera):
        for obj in self.objects:
            obj.draw(renderer, camera)
