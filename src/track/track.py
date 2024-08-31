import math

from typing import Union
from itertools import count

from Box2D import b2EdgeShape
from Box2D.Box2D import b2World
from Box2D.Box2D import b2CircleShape
from Box2D.Box2D import b2Filter
from Box2D.Box2D import b2Vec2

from src.utils.config import Config


class Target:

    _ids = count(0)

    def __init__(self):
        self.index = next(self._ids)


class Gate(Target):

    def __init__(
        self,
        world: b2World,
        x_pos: float,
        y_pos: float,
        theta: float,
        gate_size: float = 8.0,
        post_size: float = 0.5,
        is_interactive: bool = False,
    ) -> None:
        super().__init__()

        self.position = b2Vec2(x_pos, y_pos)
        self.gate_size = gate_size

        # Left side
        x_0 = x_pos + 0.5 * gate_size * math.cos(theta)
        y_0 = y_pos + 0.5 * gate_size * math.sin(theta)
        gate = world.CreateStaticBody(position=(x_0, y_0))
        gate.CreateFixture(
            shape=b2CircleShape(radius=post_size),
            filter=b2Filter(groupIndex=0 if is_interactive else -1),
        )

        # Right side
        x_1 = x_pos + 0.5 * gate_size * math.cos(theta + math.pi)
        y_1 = y_pos + 0.5 * gate_size * math.sin(theta + math.pi)
        gate = world.CreateStaticBody(position=(x_1, y_1))
        gate.CreateFixture(
            shape=b2CircleShape(radius=post_size),
            filter=b2Filter(groupIndex=0 if is_interactive else -1),
        )
        gate.userData = self.index


class Mark(Target):

    def __init__(
        self,
        world: b2World,
        x_pos: float,
        y_pos: float,
        gate_size: float = 3.0,
    ) -> None:
        super().__init__()
        self.position = b2Vec2(x_pos, y_pos)
        self.gate_size = gate_size

        gate = world.CreateStaticBody(position=(x_pos, y_pos))
        gate.CreateFixture(
            shape=b2CircleShape(radius=gate_size),
            filter=b2Filter(groupIndex=-1),  # Negative groups never collide.
        )
        gate.userData = self.index


class Track:

    def __init__(self, world: b2World, config: Config) -> None:

        self._get_boundary(world=world, config=config.env.domain)

        # track_1: square 
        gates = (
            (-15.0, -15.0, 0.5 * math.pi),
            (-12.0, 0.0, 0.5 * math.pi),
            (-15.0, 15.0, 0.5 * math.pi),
            (-7.0, 12.0, 0.5 * math.pi),
            (0.0, 5.0, 0.5 * math.pi),
            (15.0, 5.0, 0.5 * math.pi),
            (12.0, -5.0, 0.5 * math.pi),
            (15.0, -15.0, 0.5 * math.pi),
            (0.0, -10.0, 0.5 * math.pi),
        )

        self.gates = self._get_gates(world=world, gates=gates)

    def _get_boundary(self, world: b2World, config: Config) -> list:

        x_min = config.limit.x_min
        x_max = config.limit.x_max
        y_min = config.limit.y_min
        y_max = config.limit.y_max

        domain_boundary = [
            b2EdgeShape(vertices=[(x_max, y_max), (x_min, y_max)]),
            b2EdgeShape(vertices=[(x_min, y_max), (x_min, y_min)]),
            b2EdgeShape(vertices=[(x_min, y_min), (x_max, y_min)]),
            b2EdgeShape(vertices=[(x_max, y_min), (x_max, y_max)]),
        ]

        world.CreateStaticBody(shapes=domain_boundary)

    def _get_gates(self, world: b2World, gates: tuple[tuple[float]]) -> list[Union[Mark, Gate]]:

        gates_ = []

        for gate in gates:
            x_pos, y_pos, theta = gate
            gates_.append(
                Mark(world=world, x_pos=x_pos, y_pos=y_pos)
                # Gate(world=world, x_pos=x_pos, y_pos=y_pos, theta=theta)
            )

        return gates_
