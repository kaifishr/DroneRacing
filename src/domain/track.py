import math

from Box2D import b2EdgeShape
from Box2D.Box2D import b2World
from Box2D.Box2D import b2Filter
from Box2D.Box2D import b2FixtureDef 
from Box2D.Box2D import b2PolygonShape, b2CircleShape


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

        shapes = self._get_boundary()
        world.CreateStaticBody(shapes=shapes)

        shapes = self._get_track(world)

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

    def _get_track(self, world) -> list:

        gates = ((-20.0, -10.0, 0.25 * math.pi), (20.0, -10.0, 1.75 * math.pi), (0.0, 10.0, 0.5 * math.pi))
        gate_size = 8.0
        radius = 0.5

        boundary = []

        # Create gates
        for gate in gates:
            x_pos, y_pos, theta = gate  # Center of gate

            # Left side
            x_0 = x_pos + 0.5 * gate_size * math.cos(theta)
            y_0 = y_pos + 0.5 * gate_size * math.sin(theta)
            gate = world.CreateStaticBody(position=(x_0, y_0))
            gate.CreateFixture(shape=b2CircleShape(radius=radius))

            # Right side
            x_1 = x_pos + 0.5 * gate_size * math.cos(theta + math.pi)
            y_1 = y_pos + 0.5 * gate_size * math.sin(theta + math.pi)
            gate = world.CreateStaticBody(position=(x_1, y_1))
            gate.CreateFixture(shape=b2CircleShape(radius=radius))

        return boundary
