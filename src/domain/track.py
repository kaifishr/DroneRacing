import math

from Box2D import b2EdgeShape
from Box2D.Box2D import b2World

from src.utils.config import Config

class Gate:

    def __init__(self, x: float, y: float, rot: float) -> None:
        pass


class Track:

    def __init__(self, world: b2World, config: Config) -> None:
        # self.config = config.track.gates
        self.config = config.env.domain

        self.x_min = self.config.limit.x_min
        self.x_max = self.config.limit.x_max
        self.y_min = self.config.limit.y_min
        self.y_max = self.config.limit.y_max

        shapes = []
        shapes += self._get_boundary()
        shapes += self._get_track()
        world.CreateStaticBody(shapes=shapes)

    def _get_boundary(self) -> list:
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max

        domain_boundary = [
            b2EdgeShape(vertices=[(x_max, y_max), (x_min, y_max)]),
            b2EdgeShape(vertices=[(x_min, y_max), (x_min, y_min)]),
            b2EdgeShape(vertices=[(x_min, y_min), (x_max, y_min)]),
            b2EdgeShape(vertices=[(x_max, y_min), (x_max, y_max)]),
        ]

        return domain_boundary

    def _get_track(self) -> list:

        gates = ((-30.0, 0.0), (30.0, 0.0))
        theta = 0.5 * math.pi
        size = 5.0

        boundary = []

        for gate in gates:
            x_pos, y_pos = gate  # Center of gate
            x_0, y_0 = x_pos + 0.5 * size * math.cos(theta), y_pos + 0.5 * size * math.sin(theta)
            x_1, y_1 = x_pos + 0.5 * size * math.cos(theta + math.pi), y_pos + 0.5 * size * math.sin(theta + math.pi)
            boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        return boundary
